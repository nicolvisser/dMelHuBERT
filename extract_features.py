from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from argparse_dataclass import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm

from dmelhubert.datasets import Wav2DMelDataset
from dmelhubert.dmelhubert import DMelHuBERT, DMelHuBERTArgs
from dmelhubert.trainer import DMelHuBERTLightningModule


@dataclass
class ExtractFeaturesArgs:
    checkpoint_path: str
    model_args_path: str
    layer: int
    waveforms_dir: str
    waveforms_pattern: str
    features_dir: str
    batch_size: int = 1
    num_workers: int = 1


def extract_features(args: ExtractFeaturesArgs):
    waveforms_dir = Path(args.waveforms_dir)
    features_dir = Path(args.features_dir)

    model_args = DMelHuBERTArgs.load_json(args.model_args_path)

    trainer = DMelHuBERTLightningModule.load_from_checkpoint(
        args.checkpoint_path, model_args=model_args, mask=False
    )
    trainer.eval()
    model: DMelHuBERT = trainer.model.cuda()
    model.eval()

    dataset = Wav2DMelDataset(
        waveforms_dir=waveforms_dir,
        waveforms_pattern=args.waveforms_pattern,
        n_mels=model_args.n_mels,
        amin=model_args.amp2db_amin,
        codebook_size=model_args.dmel_codebook_size,
        codebook_min_value=model_args.dmel_codebook_min_value,
        codebook_max_value=model_args.dmel_codebook_max_value,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=args.num_workers,
    )

    features_dir.mkdir(parents=True, exist_ok=True)
    for relative_paths, dmels, seqlens in tqdm(loader):
        with torch.no_grad():
            features, mask = model.encode_from_dmel(
                dmels.cuda(), seqlens, layer=args.layer
            )
            assert mask is None
        i = 0
        for relative_path, seqlen in zip(relative_paths, seqlens):
            feats = features[i : i + seqlen]
            i += seqlen
            output_path: Path = (features_dir / relative_path).with_suffix(".pt")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(feats.cpu(), output_path)


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[int]]:
    relative_paths, dmels, seqlens = zip(*batch)
    dmels = torch.cat(dmels, dim=0).contiguous()
    return list(relative_paths), dmels, list(seqlens)


if __name__ == "__main__":
    parser = ArgumentParser(ExtractFeaturesArgs)
    parser.description = "Extract dMelHuBERT features from a directory of waveforms"
    parser.epilog = "See the implementation of extract_features.py for more details"
    args = parser.parse_args()
    extract_features(args)
