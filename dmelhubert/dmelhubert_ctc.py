import json
from dataclasses import dataclass, fields
from typing import List, Tuple

import torch
import torch.nn as nn
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from .dmel import DMel
from .dmelhubert import DMelEmbedding, TransformerEncoder
from .transformer import (
    RMSNorm,
    TransformerBlock,
    positions_from_sizes,
    precompute_freqs_cis,
)


@dataclass
class DMelHuBERTCTCArgs:
    n_mels: int = 80
    amp2db_amin: float = 1e-6
    dmel_codebook_size: int = 16
    dmel_codebook_min_value: float = -157.5950
    dmel_codebook_max_value: float = -69.8588
    dmel_embedding_dim: int = 64
    vocab_size: int = 28 + 1  # +1 for blank
    blank_idx: int = 28
    model_dim: int = 768
    n_heads: int = 12
    n_kv_heads: int = 12
    head_dim: int = 64
    dropout: float = 0.1
    n_layers: int = 12
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0

    @classmethod
    def load_json(cls, path: str) -> "DMelHuBERTCTCArgs":
        with open(path, "r") as f:
            return cls(**json.load(f))

    def to_dict(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def save_json(self, path: str, indent: int = 4) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)


class DMelHuBERTCTC(nn.Module):
    def __init__(self, args: DMelHuBERTCTCArgs):
        super().__init__()
        self.args = args

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

        self.proj = nn.Linear(args.model_dim, args.vocab_size)

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

    def forward(
        self,
        dmel: torch.Tensor,  # (seqlen, n_mels)
        seqlens: List[int],  # len bsz
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dmel_embedding(dmel)  # (seqlen, model_dim)
        x = self.dropout(self.norm(x))  # (seqlen, model_dim)

        positions = positions_from_sizes(seqlens, self.freqs_cis.device)  # (seqlen,)
        freqs_cis = self.freqs_cis[positions].to(device=x.device)  # (seqlen, model_dim)
        att_mask = BlockDiagonalMask.from_seqlens(seqlens)  # (seqlen, seqlen)

        x = self.encoder(
            src=x,
            freqs_cis=freqs_cis,
            att_mask=att_mask,
            output_layer=None,
        )  # (seqlen, model_dim)

        logits = self.proj(x)  # (seqlen, vocab_size)
        return logits.float()

    def save_pretrained_checkpoint(
        self,
        checkpoint_path: str,
    ):
        state_dict = self.state_dict()
        torch.save(state_dict, checkpoint_path)
