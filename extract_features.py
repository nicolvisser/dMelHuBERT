from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dmelhubert.datasets import Wav2DMelDataset
from dmelhubert.dmelhubert import DMelHuBERT, DMelHuBERTArgs
from dmelhubert.trainer import DMelHuBERTLightningModule


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[int]]:
    relative_paths, dmels, seqlens = zip(*batch)
    dmels = torch.cat(dmels, dim=0).contiguous()
    return list(relative_paths), dmels, list(seqlens)


def extract_features(
    checkpoint_path: str,
    model_args_path: str,
    layer: int,
    waveforms_dir: str,
    waveforms_pattern: str,
    features_dir: str,
    batch_size: int = 1,
    num_workers: int = 1,
):
    waveforms_dir = Path(waveforms_dir)
    features_dir = Path(features_dir)

    model_args = DMelHuBERTArgs.load_json(model_args_path)

    trainer = DMelHuBERTLightningModule.load_from_checkpoint(
        checkpoint_path, model_args=model_args, mask=False
    )
    trainer.eval()
    model: DMelHuBERT = trainer.model.cuda()
    model.eval()

    dataset = Wav2DMelDataset(
        waveforms_dir=waveforms_dir,
        waveforms_pattern=waveforms_pattern,
        n_mels=model_args.n_mels,
        amin=model_args.amp2db_amin,
        codebook_size=model_args.dmel_codebook_size,
        codebook_min_value=model_args.dmel_codebook_min_value,
        codebook_max_value=model_args.dmel_codebook_max_value,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
    )

    features_dir.mkdir(parents=True, exist_ok=True)
    for relative_paths, dmels, seqlens in tqdm(loader):
        with torch.no_grad():
            features, mask = model.encode_from_dmel(dmels.cuda(), seqlens, layer=layer)
            assert mask is None
        i = 0
        for relative_path, seqlen in zip(relative_paths, seqlens):
            feats = features[i : i + seqlen]
            i += seqlen
            output_path: Path = (Path(features_dir) / relative_path).with_suffix(".pt")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(feats.cpu(), output_path)


if __name__ == "__main__":
    layer = 9
    extract_features(
        checkpoint_path="/mnt/wsl/nvme/code/dMelHuBERT/checkpoints/dmelhubert-iter1/epoch=34-step=100000.ckpt",
        model_args_path="/mnt/wsl/nvme/code/dMelHuBERT/checkpoints/dmelhubert-iter1/model_args.json",
        layer=layer,
        waveforms_dir="/mnt/wsl/nvme/datasets/LibriSpeech",
        waveforms_pattern="train-clean-100/**/*.flac",
        features_dir=f"/mnt/wsl/nvme/code/dMelHuBERT/output/iter1-features/layer-{layer}",
        batch_size=32,
        num_workers=32,
    )
