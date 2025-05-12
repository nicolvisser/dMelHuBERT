from dataclasses import dataclass
from pathlib import Path

import torch
from argparse_dataclass import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm

from dmelhubert.datasets import Wav2DMelDataset


@dataclass
class ExtractDMelsArgs:
    waveform_dir: str
    dmels_dir: str
    waveform_pattern: str = "**/*.flac"
    n_mels: int = 80
    amin: float = 1e-6
    codebook_size: int = 16
    codebook_min_value: float = -157.5950
    codebook_max_value: float = -69.8588
    num_workers: int = 1


def extract_dmels(args: ExtractDMelsArgs):
    dataset = Wav2DMelDataset(
        waveforms_dir=args.waveform_dir,
        waveforms_pattern=args.waveform_pattern,
        n_mels=args.n_mels,
        amin=args.amin,
        codebook_size=args.codebook_size,
        codebook_min_value=args.codebook_min_value,
        codebook_max_value=args.codebook_max_value,
    )
    collate_fn = lambda x: x[0]
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    for relative_path, dmel, seqlen in tqdm(loader):
        dmel = dmel.to(device="cpu", dtype=torch.int8)
        dmel_path = (args.dmels_dir / relative_path).with_suffix(".pt")
        dmel_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dmel, dmel_path)


if __name__ == "__main__":
    parser = ArgumentParser(ExtractDMelsArgs)
    parser.description = "Extract dMel encodings from a directory of waveforms"
    parser.epilog = "See the implementation of extract_dmels.py for more details"
    args = parser.parse_args()
    extract_dmels(args)
