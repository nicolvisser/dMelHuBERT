from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dmelhubert.datasets import Wav2MFCCLabelsDataset


def save_mfcc_labels_to_disk(
    labels_dir: Path,
    relative_path: Path,
    labels: torch.Tensor,
    dtype: torch.dtype = torch.int64,
):
    labels = labels.to(device="cpu", dtype=dtype)
    labels_path = (labels_dir / relative_path).with_suffix(".pt")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(labels, labels_path)


def extract_mfcc_labels(
    waveform_dir: str,
    codebook_path: str,
    labels_dir: str,
    waveform_pattern: str = "**/*.flac",
    dtype: torch.dtype = torch.int64,
    num_workers: int = 1,
):
    dataset = Wav2MFCCLabelsDataset(
        waveforms_dir=waveform_dir,
        codebook_path=codebook_path,
        waveforms_pattern=waveform_pattern,
    )
    collate_fn = lambda x: x[0]
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    for relative_path, labels, seqlen in tqdm(loader):
        save_mfcc_labels_to_disk(labels_dir, relative_path, labels, dtype)


if __name__ == "__main__":
    import argparse

    dtype_lookup = {
        "uint8": torch.uint8,
        "uint16": torch.uint16,
        "uint32": torch.uint32,
        "uint64": torch.uint64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "long": torch.long,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "waveform_dir", type=str, help="Directory containing waveform files"
    )
    parser.add_argument(
        "codebook_path",
        type=str,
        help="Path to the codebook",
    )
    parser.add_argument("labels_dir", type=str, help="Directory to save labels")
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.flac",
        help="Pattern to match waveform files",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint8",
        help=f"Data type to save labels",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=1,
        help="Number of workers for loading data",
    )
    args = parser.parse_args()

    assert args.dtype in dtype_lookup, f"Invalid dtype: {args.dtype}"

    extract_mfcc_labels(
        waveform_dir=args.waveform_dir,
        codebook_path=args.codebook_path,
        labels_dir=args.labels_dir,
        waveform_pattern=args.pattern,
        dtype=dtype_lookup[args.dtype],
        num_workers=args.njobs,
    )
