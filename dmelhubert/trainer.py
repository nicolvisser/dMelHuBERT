from pathlib import Path
from typing import List, Tuple

import lightning as L
import torch

from .model import DMelHuBERT


class MelHuBERTIteration1(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DMelHuBERT(num_label_embeddings=100, mask=True)

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        dmel, labels, seqlens = batch

        logits, mask = self.model(
            dmel=dmel,
            seqlens=seqlens,
        )  # (seqlen, 100)

        loss = torch.nn.functional.cross_entropy(
            input=logits[mask] if mask is not None else logits,
            target=labels[mask] if mask is not None else labels,
            reduction="mean",
        )
        return loss

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[int]],
        batch_idx: int,
    ) -> torch.Tensor:
        batch_size = len(batch[2])
        loss = self(batch)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[int]],
        batch_idx: int,
    ) -> None:
        # Check if we have an extra dimension and remove it if necessary
        mfccs, labels, seqlens = batch
        if mfccs.dim() > 2:  # If there's an extra dimension
            mfccs = mfccs.squeeze(0)
            labels = labels.squeeze(0)
            seqlens = seqlens[0] if isinstance(seqlens[0], list) else seqlens
            batch = (mfccs, labels, seqlens)

        batch_size = len(batch[2])
        loss = self(batch)
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )

    def configure_optimizers(self):
        return None  # optimizers will be configured externally

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str,
        checkpoint_filename: str = "best.ckpt",
    ):
        model_dir = Path(model_dir)
        checkpoint_path: Path = model_dir / checkpoint_filename
        assert checkpoint_path.exists(), checkpoint_path
        return cls.load_from_checkpoint(checkpoint_path=checkpoint_path)
