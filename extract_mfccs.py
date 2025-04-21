from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dmelhubert.datasets import Wav2MFCCDataset


def save_mfcc_to_disk(
    mfccs_dir: Path,
    relative_path: Path,
    mfccs: torch.Tensor,
):
    mfccs = mfccs.to(device="cpu", dtype=torch.float32)
    mfccs_path = (mfccs_dir / relative_path).with_suffix(".pt")
    mfccs_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mfccs, mfccs_path)


def extract_mfccs(
    waveform_dir: str,
    mfccs_dir: str,
    waveform_pattern: str = "**/*.flac",
    num_workers: int = 1,
):
    dataset = Wav2MFCCDataset(
        waveforms_dir=waveform_dir,
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
    for relative_path, mfccs, seqlen in tqdm(loader):
        save_mfcc_to_disk(mfccs_dir, relative_path, mfccs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "waveform_dir", type=str, help="Directory containing waveform files"
    )
    parser.add_argument("mfccs_dir", type=str, help="Directory to save MFCCs")
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.flac",
        help="Pattern to match waveform files",
    )

    parser.add_argument(
        "--njobs",
        type=int,
        default=1,
        help="Number of workers for loading data",
    )
    args = parser.parse_args()
    extract_mfccs(
        waveform_dir=args.waveform_dir,
        mfccs_dir=args.mfccs_dir,
        waveform_pattern=args.pattern,
        num_workers=args.njobs,
    )
