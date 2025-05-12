import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as L
import torch

from .dmelhubert_ctc import DMelHuBERTCTC, DMelHuBERTCTCArgs, DMelHuBERTCTCWithLora


class DMelHuBERTCTCLightningModule(L.LightningModule):
    def __init__(
        self,
        model_args: DMelHuBERTCTCArgs,
        use_lora: bool = False,
        r: Optional[int] = None,
        alpha: Optional[float] = None,
        dropout: Optional[float] = None,
        add_to_query: Optional[bool] = None,
        add_to_key: Optional[bool] = None,
        add_to_value: Optional[bool] = None,
        add_to_output: Optional[bool] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.model_args = model_args
        self.label_smoothing = label_smoothing

        self.model: Union[DMelHuBERTCTC, DMelHuBERTCTCWithLora] = None
        if use_lora:
            self.model = DMelHuBERTCTCWithLora(
                model_args,
                r=r,
                alpha=alpha,
                dropout=dropout,
                add_to_query=add_to_query,
                add_to_key=add_to_key,
                add_to_value=add_to_value,
                add_to_output=add_to_output,
            )
        else:
            self.model = DMelHuBERTCTC(model_args)

        # Load character mappings
        try:
            with open("checkpoints/ctc-char-mappings.json", "r") as f:
                self.char_mappings = json.load(f)
        except Exception:
            self.char_mappings = None

    def forward(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        dmel, seqlens, targets, tgtlens = batch

        logits: torch.Tensor = self.model(
            dmel=dmel,
            seqlens=seqlens.tolist(),
        )  # (total_seqlen, vocab_size)

        total_seqlen, vocab_size = logits.size()
        max_seqlen = max(seqlens)
        batch_size = len(seqlens)

        batched_logits = logits.new_full(
            (max_seqlen, batch_size, vocab_size), fill_value=0.0
        )  # (max_seqlen, batch_size, vocab_size)

        offset = 0
        for i, l in enumerate(seqlens):
            batched_logits[:l, i] = logits[offset : offset + l]
            offset += l

        if self.label_smoothing > 0:
            smooth_factor = self.label_smoothing / (vocab_size - 1)
            probs = torch.nn.functional.softmax(batched_logits, dim=2)
            smooth_probs = (1.0 - self.label_smoothing) * probs + smooth_factor
            log_probs = torch.log(smooth_probs)
        else:
            log_probs = torch.nn.functional.log_softmax(batched_logits, dim=2)

        loss = torch.nn.functional.ctc_loss(
            log_probs=log_probs,  # (max_seqlen, batch_size, vocab_size)
            targets=targets,  # (sum(tgtlens))
            input_lengths=seqlens,  # (batch_size)
            target_lengths=tgtlens,  # (batch_size)
            blank=self.model_args.blank_idx,
            reduction="mean",
        )

        return loss, log_probs

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        batch_size = len(batch[1])
        loss, _ = self(batch)
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
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        batch_size = len(batch[1])
        loss, log_probs = self(batch)
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        # Decode and log samples from the first batch only
        if batch_idx == 0 and self.char_mappings is not None:
            dmel, seqlens, targets, tgtlens = batch

            # Get predictions using greedy decoding
            predictions = log_probs.argmax(dim=-1).transpose(
                0, 1
            )  # (batch_size, max_seqlen)

            # Decode up to 4 samples to avoid excessive logging
            for i in range(min(4, batch_size)):
                # Decode prediction (remove repeated tokens and blanks)
                sample_len = seqlens[i].item()
                sample_pred = predictions[i, :sample_len].tolist()
                blank_idx = self.model_args.blank_idx

                # Convert prediction to text (merge repeated, remove blanks)
                decoded_text = self._ctc_decode(sample_pred, blank_idx)

                # Decode target for comparison
                target_start = sum(tgtlens[:i].tolist()) if i > 0 else 0
                target_end = target_start + tgtlens[i].item()
                sample_target = targets[target_start:target_end].tolist()
                target_text = self._ctc_decode(sample_target, blank_idx)

                print(f"Sample {i+1}:")
                print(f"Prediction: {decoded_text}")
                print(f"Target: {target_text}")
                print("\n")

    def _ctc_decode(self, pred_indices, blank_idx):
        # collapse repeated tokens
        result = [pred_indices[0]]
        for idx in pred_indices[1:]:
            if idx != result[-1]:
                result.append(idx)

        # remove blanks
        result = [i for i in result if i != blank_idx]

        # convert to text
        return self._indices_to_text(result)

    def _indices_to_text(self, indices):
        """Convert indices to text using character mappings"""
        if not self.char_mappings:
            return str(indices)

        return "".join([self.char_mappings[idx] for idx in indices])

    def configure_optimizers(self):
        return None  # optimizers will be configured externally
