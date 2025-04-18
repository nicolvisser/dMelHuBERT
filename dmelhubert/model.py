import copy
import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from .transformer import (
    TransformerBlock,
    positions_from_sizes,
    precompute_freqs_cis,
    RMSNorm,
)


class DMelHuBERT(nn.Module):
    def __init__(self, num_label_embeddings: int = 100, mask: bool = True):
        super().__init__()
        self._mask = mask
        self.dmel_embedding = DMelEmbedding()
        self.norm = RMSNorm(768)
        self.dropout = nn.Dropout(0.1)
        self._freqs_cis = None  # set lazily
        self.encoder = TransformerEncoder(
            TransformerBlock(
                dim=768,
                hidden_dim=4 * 768,
                n_heads=12,
                n_kv_heads=12,
                head_dim=64,
                dropout=0.1,
                norm_eps=1e-6,
            ),
            12,
        )
        self.proj = nn.Linear(768, 256)

        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(768).uniform_())
        self.label_embedding = nn.Embedding(num_label_embeddings, 256)

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
                dim=64,
                end=8_000,  # can change this later
                theta=10_000.0,  # do not change this later
                device=self.device,
            )
        return self._freqs_cis

    def mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = None
        if self._mask:
            mask = _compute_mask(x.size(0), 0.8, 10, x.device, 2)
            x[mask] = self.masked_spec_embed.to(x.dtype)
        return x, mask

    def encode_from_dmel(
        self,
        x: torch.Tensor,  # (seqlen, n_mels)
        seqlens: List[int],  # len bsz
        layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dmel_embedding(x)  # (seqlen, 768)
        x, mask = self.mask(x)  # (seqlen, 768), (seqlen)
        x = self.dropout(self.norm(x))  # (seqlen, 768)

        positions = positions_from_sizes(seqlens, self.freqs_cis.device)  # (seqlen,)
        freqs_cis = self.freqs_cis[positions].to(device=x.device)  # (seqlen, 768)
        att_mask = BlockDiagonalMask.from_seqlens(seqlens)  # (seqlen, seqlen)

        x = self.encoder(
            src=x,
            freqs_cis=freqs_cis,
            att_mask=att_mask,
            output_layer=layer,
        )  # (seqlen, 768)

        return x, mask

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.cosine_similarity(
            x.unsqueeze(1),  # (seqlen, 1, 256)
            self.label_embedding.weight.unsqueeze(0),  # (1, 100, 256)
            dim=-1,
        )  # (seqlen, 100)
        return logits / 0.1  # (seqlen, 100)

    def forward(
        self,
        dmel: torch.Tensor,  # (seqlen, 39)
        seqlens: List[int],  # len bsz
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.encode_from_dmel(dmel, seqlens)  # (seqlen, 768)
        x = self.proj(x)  # (seqlen, 256)
        logits = self.logits(x)  # (seqlen, 100)
        return logits, mask

    def save_pretrained_checkpoint(
        self,
        checkpoint_path: str,
    ):
        state_dict = self.state_dict()
        torch.save(state_dict, checkpoint_path)


class DMelEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=16, embedding_dim=64)
        self.linear = torch.nn.Linear(in_features=80 * 64, out_features=768)

    def forward(
        self,
        dmels: torch.Tensor,  # seqlen x n_mels
    ) -> torch.Tensor:
        E_ = self.embedding(dmels)  # seqlen x n_mels x embedding_dim
        E = self.linear(E_.view(E_.shape[0], -1))  # seqlen x model_dim
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
