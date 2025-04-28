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


def extract_features_labels(
    checkpoint_path: str,
    model_args_path: str,
    layer: int,
    centroids_path: str,
    waveforms_dir: str,
    waveforms_pattern: str,
    labels_dir: str,
    batch_size: int = 1,
    num_workers: int = 1,
    dtype: torch.dtype = torch.int64,
):
    waveforms_dir = Path(waveforms_dir)
    labels_dir = Path(labels_dir)

    model_args = DMelHuBERTArgs.load_json(model_args_path)

    trainer = DMelHuBERTLightningModule.load_from_checkpoint(
        checkpoint_path, model_args=model_args, mask=False
    )
    trainer.eval()
    model: DMelHuBERT = trainer.model.cuda()
    model.eval()

    centroids = torch.load(centroids_path).cuda()

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

    labels_dir.mkdir(parents=True, exist_ok=True)
    for relative_paths, dmels, seqlens in tqdm(loader):
        with torch.no_grad():
            features, mask = model.encode_from_dmel(dmels.cuda(), seqlens, layer=layer)
            assert mask is None

        dists = torch.cdist(features, centroids)
        labels = torch.argmin(dists, dim=-1).to(dtype=dtype)
        i = 0
        for relative_path, seqlen in zip(relative_paths, seqlens):
            lab = labels[i : i + seqlen]
            i += seqlen
            output_path: Path = (Path(labels_dir) / relative_path).with_suffix(".pt")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(lab.cpu(), output_path)


if __name__ == "__main__":
    extract_features_labels(
        checkpoint_path="/mnt/wsl/nvme/code/dMelHuBERT/checkpoints/dmelhubert-iter2/epoch=34-step=100000.ckpt",
        model_args_path="/mnt/wsl/nvme/code/dMelHuBERT/checkpoints/dmelhubert-iter2/model_args.json",
        layer=13,
        centroids_path=f"/mnt/wsl/nvme/code/dMelHuBERT/output/kmeans-iter-2-layer-13-k-500/centroids-k-500.pt",
        waveforms_dir="/mnt/wsl/nvme/datasets/LibriSpeech",
        waveforms_pattern="**/*.flac",
        labels_dir=f"/mnt/wsl/data/dmelhubert-features-labels/iter-2/layer-13/k-500/LibriSpeech",
        batch_size=32,
        num_workers=32,
        dtype=torch.uint16,
    )
