import json
from dataclasses import dataclass, fields
from pathlib import Path

import lightning.pytorch as pl
import torch
from argparse_dataclass import ArgumentParser
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

import wandb
from dmelhubert.datasets import DMelHuBERTTrainDataset, train_collate_fn
from dmelhubert.dmelhubert import DMelHuBERTArgs
from dmelhubert.scheduler import LinearRampCosineDecayScheduler
from dmelhubert.trainer import DMelHuBERTLightningModule


@dataclass
class TrainArgs:
    project_name: str = "dMelHuBERT"
    run_name: str = "dmelhubert-iter1"
    train_dmels_dir: str = "/mnt/wsl/data/dmel/LibriSpeech"
    train_dmels_pattern: str = "train*/**/*.pt"
    train_labels_dir: str = "/mnt/wsl/data/mfcc-labels/k-100/LibriSpeech/"
    train_labels_pattern: str = "train*/**/*.pt"
    valid_dmels_dir: str = "/mnt/wsl/data/dmel/LibriSpeech"
    valid_dmels_pattern: str = "dev*/**/*.pt"
    valid_labels_dir: str = "/mnt/wsl/data/mfcc-labels/k-100/LibriSpeech/"
    valid_labels_pattern: str = "dev*/**/*.pt"
    max_duration: float = 10.0
    batch_size: int = 16
    num_workers: int = 32
    lr_init: float = 2e-7
    warmup_steps: int = 5000
    lr_max: float = 2e-4
    decay_steps: int = 95000
    lr_final: float = 0.0
    betas: tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 0.01
    eps: float = 1e-8
    accelerator: str = "gpu"
    strategy: str = "auto"
    devices: int = 1
    precision: str = "bf16-mixed"
    fast_dev_run: bool = False
    max_steps: int = 100000
    val_check_interval: float = 1000
    save_model_every_n_steps: int = 10000
    check_val_every_n_epoch: int = None
    log_every_n_steps: int = 1
    accumulate_grad_batches: int = 1
    gradient_clip_algorithm: str = "norm"
    gradient_clip_val: float = 1.0
    force: bool = False

    @classmethod
    def load_json(cls, path: str) -> "TrainArgs":
        with open(path, "r") as f:
            return cls(**json.load(f))

    def to_dict(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def save_json(self, path: str, indent: int = 4) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)


def train(model_args: DMelHuBERTArgs, train_args: TrainArgs) -> None:
    """Train the MelHuBERT model.

    Args:
        train_config_path: Path to the training configuration file
    """

    # DATASET SETUP
    train_dataset = DMelHuBERTTrainDataset(
        dmels_dir=train_args.train_dmels_dir,
        labels_dir=train_args.train_labels_dir,
        dmels_pattern=train_args.train_dmels_pattern,
        labels_pattern=train_args.train_labels_pattern,
        max_duration=train_args.max_duration,
    )
    valid_dataset = DMelHuBERTTrainDataset(
        dmels_dir=train_args.valid_dmels_dir,
        labels_dir=train_args.valid_labels_dir,
        dmels_pattern=train_args.valid_dmels_pattern,
        labels_pattern=train_args.valid_labels_pattern,
        max_duration=train_args.max_duration,
    )

    # DATALOADER SETUP
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_args.batch_size,
        shuffle=True,
        num_workers=train_args.num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=train_args.batch_size,
        shuffle=False,
        num_workers=train_args.num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # CHECKPOINT SETUP

    checkpoint_dir = Path("./checkpoints") / train_args.run_name
    train_args_path = checkpoint_dir / "train_args.json"

    if checkpoint_dir.exists() and not train_args.force:
        msg = (
            f"Checkpoint directory {checkpoint_dir} already exists. Aborting training."
        )
        print(msg)
        exit()

    train_args_path.parent.mkdir(parents=True, exist_ok=True)
    train_args.save_json(train_args_path)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last=True,
        save_weights_only=True,
        save_top_k=-1,
        every_n_train_steps=train_args.save_model_every_n_steps,
    )

    # LOGGER SETUP

    logger = pl.loggers.WandbLogger(
        log_model=True,
        project=train_args.project_name,
        name=train_args.run_name,
    )

    # Log hyperparameters
    logger.log_hyperparams(
        {
            "train_args": train_args.to_dict(),
        }
    )

    # LR MONITOR SETUP

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    # MODEL SETUP

    model = DMelHuBERTLightningModule(model_args)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_args.lr_max,
        betas=train_args.betas,
        weight_decay=train_args.weight_decay,
        eps=train_args.eps,
    )

    scheduler = {
        "scheduler": LinearRampCosineDecayScheduler(
            optimizer,
            n_linear_steps=train_args.warmup_steps,
            n_decay_steps=train_args.decay_steps,
            lr_init=train_args.lr_init,
            lr_max=train_args.lr_max,
            lr_final=train_args.lr_final,
        ),
        "frequency": 1,
        "interval": "step",
    }

    model.configure_optimizers = lambda: {
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
    }

    # TRAINER SETUP

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            checkpoint_callback,
            lr_monitor_callback,
        ],
        accelerator=train_args.accelerator,
        strategy=train_args.strategy,
        devices=train_args.devices,
        precision=train_args.precision,
        fast_dev_run=train_args.fast_dev_run,
        max_steps=train_args.max_steps,
        val_check_interval=train_args.val_check_interval,
        check_val_every_n_epoch=train_args.check_val_every_n_epoch,
        log_every_n_steps=train_args.log_every_n_steps,
        accumulate_grad_batches=train_args.accumulate_grad_batches,
        gradient_clip_algorithm=train_args.gradient_clip_algorithm,
        gradient_clip_val=train_args.gradient_clip_val,
    )

    # SAVE CURRENT SCRIPT

    wandb.save(str(Path(__file__).resolve()))

    # TRAINING

    torch.set_float32_matmul_precision("medium")  # Optimize for some NVIDIA GPUs

    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    model_args = DMelHuBERTArgs()
    train_args = ArgumentParser(TrainArgs).parse_args()

    train(model_args, train_args)
