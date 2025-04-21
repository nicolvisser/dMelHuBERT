from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dmelhubert.datasets import Wav2DMelDataset


def save_dmel_to_disk(
    dmels_dir: Path,
    relative_path: Path,
    dmel: torch.Tensor,
):
    dmel = dmel.to(device="cpu", dtype=torch.int8)
    dmel_path = (dmels_dir / relative_path).with_suffix(".pt")
    dmel_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dmel, dmel_path)


def extract_dmels(
    waveform_dir: str,
    dmels_dir: str,
    waveform_pattern: str = "**/*.flac",
    n_mels: int = 80,
    amin: float = 1e-6,
    codebook_size: int = 16,
    codebook_min_value: float = -157.5950,
    codebook_max_value: float = -69.8588,
    num_workers: int = 1,
):
    dataset = Wav2DMelDataset(
        waveforms_dir=waveform_dir,
        waveforms_pattern=waveform_pattern,
        n_mels=n_mels,
        amin=amin,
        codebook_size=codebook_size,
        codebook_min_value=codebook_min_value,
        codebook_max_value=codebook_max_value,
    )
    collate_fn = lambda x: x[0]
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    for relative_path, dmel, seqlen in tqdm(loader):
        save_dmel_to_disk(dmels_dir, relative_path, dmel)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "waveform_dir", type=str, help="Directory containing waveform files"
    )
    parser.add_argument("dmels_dir", type=str, help="Directory to save dMel features")
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.flac",
        help="Pattern to match waveform files",
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=80,
        help="Number of mel bins",
    )
    parser.add_argument(
        "--amin",
        type=float,
        default=1e-6,
        help="Minimum clamping value for dB conversion",
    )
    parser.add_argument(
        "--codebook_size",
        type=int,
        default=16,
        help="Number of codebook values",
    )
    parser.add_argument(
        "--codebook_min_value",
        type=float,
        default=-157.5950,
        help="Minimum value for codebook",
    )
    parser.add_argument(
        "--codebook_max_value",
        type=float,
        default=-69.8588,
        help="Maximum value for codebook",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=1,
        help="Number of workers for loading data",
    )
    args = parser.parse_args()
    extract_dmels(
        waveform_dir=args.waveform_dir,
        dmels_dir=args.dmels_dir,
        waveform_pattern=args.pattern,
        n_mels=args.n_mels,
        amin=args.amin,
        codebook_size=args.codebook_size,
        codebook_min_value=args.codebook_min_value,
        codebook_max_value=args.codebook_max_value,
        num_workers=args.njobs,
    )
