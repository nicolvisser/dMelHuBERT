from pathlib import Path
from typing import List, Tuple

import lightning as L
import torch

from .dmelhubert import DMelHuBERT, DMelHuBERTArgs


class DMelHuBERTLightningModule(L.LightningModule):
    def __init__(self, model_args: DMelHuBERTArgs):
        super().__init__()
        self.model = DMelHuBERT(model_args, mask=True)

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        dmel, labels, seqlens = batch

        logits, mask = self.model(
            dmel=dmel,
            seqlens=seqlens,
        )  # (seqlen, n_label_embeddings)

        loss = torch.nn.functional.cross_entropy(
            input=logits[mask],
            target=labels[mask],
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
