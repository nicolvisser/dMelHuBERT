import copy
import json
import random
from dataclasses import dataclass, fields
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from .dmel import DMel
from .transformer import (
    RMSNorm,
    TransformerBlock,
    positions_from_sizes,
    precompute_freqs_cis,
)


@dataclass
class DMelHuBERTArgs:
    n_mels: int = 80
    amp2db_amin: float = 1e-6
    dmel_codebook_size: int = 16
    dmel_codebook_min_value: float = -157.5950
    dmel_codebook_max_value: float = -69.8588
    dmel_embedding_dim: int = 64
    n_label_embeddings: int = 100
    model_dim: int = 768
    n_heads: int = 12
    n_kv_heads: int = 12
    head_dim: int = 64
    dropout: float = 0.1
    n_layers: int = 12
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    proj_dim: int = 256
    mask_prob: float = 0.8
    mask_length: int = 10
    min_masks: int = 2
    temperature: float = 0.1
    targets: Literal["soft", "hard"] = "soft"

    @classmethod
    def load_json(cls, path: str) -> "DMelHuBERTArgs":
        with open(path, "r") as f:
            return cls(**json.load(f))

    def to_dict(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def save_json(self, path: str, indent: int = 4) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)


class DMelHuBERT(nn.Module):
    def __init__(self, args: DMelHuBERTArgs, mask: bool = True):
        super().__init__()
        self.args = args

        self._mask = mask
        self.dmel = DMel(
            n_mels=args.n_mels,
            amin=args.amp2db_amin,
            codebook_size=args.dmel_codebook_size,
            codebook_min_value=args.dmel_codebook_min_value,
            codebook_max_value=args.dmel_codebook_max_value,
        )
        self.dmel_embedding = DMelEmbedding(args)
        self.norm = RMSNorm(args.model_dim, eps=args.rms_norm_eps)
        self.dropout = nn.Dropout(args.dropout)
        self._freqs_cis = None  # set lazily
        self.encoder = TransformerEncoder(
            TransformerBlock(
                dim=args.model_dim,
                hidden_dim=4 * args.model_dim,
                n_heads=args.n_heads,
                n_kv_heads=args.n_kv_heads,
                head_dim=args.head_dim,
                dropout=args.dropout,
                norm_eps=args.rms_norm_eps,
            ),
            args.n_layers,
        )
        if self.args.targets == "soft":
            self.proj = nn.Linear(args.model_dim, args.proj_dim)
        else:
            self.proj = nn.Linear(args.model_dim, args.n_label_embeddings)

        self.masked_spec_embed = nn.Parameter(
            torch.FloatTensor(args.model_dim).uniform_()
        )
        if self.args.targets == "soft":
            self.label_embedding = nn.Embedding(args.n_label_embeddings, args.proj_dim)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def freqs_cis(self):
        # lazy init
        if self._freqs_cis is None:
            self._freqs_cis = precompute_freqs_cis(
                dim=self.args.head_dim,
                end=8_000,  # can change this later
                theta=self.args.rope_theta,
                device=self.device,
            )
        return self._freqs_cis

    def mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = None
        if self._mask:
            mask = _compute_mask(
                sequence_length=x.size(0),
                mask_prob=self.args.mask_prob,
                mask_length=self.args.mask_length,
                device=x.device,
                min_masks=self.args.min_masks,
            )
            x[mask] = self.masked_spec_embed.to(x.dtype)
        return x, mask

    def encode_from_dmel(
        self,
        dmel: torch.Tensor,  # (seqlen, n_mels)
        seqlens: List[int],  # len bsz
        layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dmel_embedding(dmel)  # (seqlen, model_dim)
        x, mask = self.mask(x)  # (seqlen, model_dim), (seqlen)
        x = self.dropout(self.norm(x))  # (seqlen, model_dim)

        positions = positions_from_sizes(seqlens, self.freqs_cis.device)  # (seqlen,)
        freqs_cis = self.freqs_cis[positions].to(device=x.device)  # (seqlen, model_dim)
        att_mask = BlockDiagonalMask.from_seqlens(seqlens)  # (seqlen, seqlen)

        x = self.encoder(
            src=x,
            freqs_cis=freqs_cis,
            att_mask=att_mask,
            output_layer=layer,
        )  # (seqlen, model_dim)

        if layer is not None and layer > self.args.n_layers:
            x = self.proj(x)

        return x, mask

    def encode_from_wav(
        self,
        wav: torch.Tensor,  # (seqlen)
        seqlens: List[int],  # len bsz
        layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dmel = self.dmel(wav)  # (seqlen, n_mels)
        return self.encode_from_dmel(dmel, seqlens, layer=layer)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.cosine_similarity(
            x.unsqueeze(1),  # (seqlen, 1, proj_dim)
            self.label_embedding.weight.unsqueeze(
                0
            ),  # (1, n_label_embeddings, proj_dim)
            dim=-1,
        )  # (seqlen, n_label_embeddings)
        return logits / self.args.temperature  # (seqlen, n_label_embeddings)

    def forward(
        self,
        dmel: torch.Tensor,  # (seqlen, n_mels)
        seqlens: List[int],  # len bsz
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.encode_from_dmel(
            dmel, seqlens, layer=None
        )  # (seqlen, model_dim)
        x = self.proj(x)  # (seqlen, proj_dim) or (seqlen, n_label_embeddings)
        if self.args.targets == "soft":
            logits = self.logits(x)  # (seqlen, n_label_embeddings)
        else:
            logits = x  # (seqlen, n_label_embeddings)
        return logits.float(), mask

    def save_pretrained_checkpoint(
        self,
        checkpoint_path: str,
    ):
        state_dict = self.state_dict()
        torch.save(state_dict, checkpoint_path)


class DMelEmbedding(nn.Module):
    def __init__(self, args: DMelHuBERTArgs):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=args.dmel_codebook_size,
            embedding_dim=args.dmel_embedding_dim,
        )
        self.linear = torch.nn.Linear(
            in_features=args.n_mels * args.dmel_embedding_dim,
            out_features=args.model_dim,
        )

    def forward(
        self,
        dmels: torch.Tensor,  # (seqlen, n_mels)
    ) -> torch.Tensor:
        E_ = self.embedding(dmels)  # (seqlen, n_mels, embedding_dim)
        E = self.linear(E_.view(E_.shape[0], -1))  # (seqlen, model_dim)
        return E


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: TransformerBlock, num_layers: int) -> None:
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
        self,
        src: torch.Tensor,
        freqs_cis: torch.Tensor,
        att_mask: torch.Tensor,
        output_layer: Optional[int] = None,
    ) -> torch.Tensor:
        output = src
        for layer in self.layers[:output_layer]:
            layer: TransformerBlock = layer
            output = layer(
                x=output,
                freqs_cis=freqs_cis,
                att_mask=att_mask,
            )
        return output


def _compute_mask(
    sequence_length: int,
    mask_prob: float,
    mask_length: int,
    device: torch.device,
    min_masks: int = 0,
) -> torch.Tensor:

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )

    # compute number of masked spans in batch
    num_masked_spans = int(mask_prob * sequence_length / mask_length + random.random())
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment mask to fill
    mask = torch.zeros((1, sequence_length), device=device, dtype=torch.bool)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = torch.ones((1, sequence_length - (mask_length - 1)), device=device)

    # get random indices to mask
    mask_indices = torch.multinomial(uniform_dist, num_masked_spans)

    # expand masked indices to masked spans
    mask_indices = (
        mask_indices.unsqueeze(dim=-1)
        .expand((1, num_masked_spans, mask_length))
        .reshape(1, num_masked_spans * mask_length)
    )
    offsets = (
        torch.arange(mask_length, device=device)[None, None, :]
        .expand((1, num_masked_spans, mask_length))
        .reshape(1, num_masked_spans * mask_length)
    )
    mask_idxs = mask_indices + offsets

    # scatter indices to mask
    mask = mask.scatter(1, mask_idxs, True)

    return mask[0]
