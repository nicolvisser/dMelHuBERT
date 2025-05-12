from dataclasses import dataclass
from pathlib import Path

import torch
from argparse_dataclass import ArgumentParser
from tqdm import tqdm


@dataclass
class ComputeBitrateArgs:
    labels_dir: str
    labels_pattern: str
    n_label_types: int
    labels_rate: float = 50.0


@dataclass
class ComputeBitrateResult:
    probs_duped: torch.Tensor
    probs_deduped: torch.Tensor
    entropy_duped: torch.Tensor
    entropy_deduped: torch.Tensor
    bitrate_duped: torch.Tensor
    bitrate_deduped: torch.Tensor
    avg_label_length: torch.Tensor


def compute_bitrate(args: ComputeBitrateArgs):
    msg = f"Labels directory: {args.labels_dir} does not exist"
    assert Path(args.labels_dir).exists(), msg
    msg = f"Labels directory: {args.labels_dir} is not a directory"
    assert Path(args.labels_dir).is_dir(), msg
    msg = f"labels_pattern: {args.labels_pattern} must match .pt files"
    assert args.labels_pattern.endswith(".pt"), msg

    labels_paths = sorted(list(Path(args.labels_dir).glob(args.labels_pattern)))
    msg = f"No label files found in {args.labels_dir} matching {args.labels_pattern}"
    assert len(labels_paths) > 0, msg

    n_labels_duped = 0
    n_labels_deduped = 0
    counts_duped = torch.zeros(args.n_label_types, dtype=torch.int64)
    counts_deduped = torch.zeros(args.n_label_types, dtype=torch.int64)
    durations = []
    for label_path in tqdm(labels_paths):
        labels_duped = torch.load(label_path).long()
        assert labels_duped.ndim == 1, "Label sequence must be 1D"
        assert labels_duped.dtype == torch.int64, "Label sequence must be of type int64"
        labels_deduped, durs = torch.unique_consecutive(
            labels_duped, return_counts=True
        )
        n_labels_duped += len(labels_duped)
        n_labels_deduped += len(labels_deduped)
        counts_duped += torch.bincount(labels_duped, minlength=args.n_label_types)
        counts_deduped += torch.bincount(labels_deduped, minlength=args.n_label_types)
        durations.extend([d / args.labels_rate for d in durs.tolist()])

    duration = n_labels_duped / args.labels_rate

    probs_duped = 1e-10 + counts_duped / n_labels_duped
    probs_deduped = 1e-10 + counts_deduped / n_labels_deduped

    entropy_duped = -torch.sum(probs_duped * torch.log2(probs_duped))
    entropy_deduped = -torch.sum(probs_deduped * torch.log2(probs_deduped))

    bitrate_duped = n_labels_duped * entropy_duped / duration
    bitrate_deduped = n_labels_deduped * entropy_deduped / duration

    avg_label_length = duration / n_labels_deduped

    return ComputeBitrateResult(
        probs_duped=probs_duped,
        probs_deduped=probs_deduped,
        entropy_duped=entropy_duped,
        entropy_deduped=entropy_deduped,
        bitrate_duped=bitrate_duped,
        bitrate_deduped=bitrate_deduped,
        avg_label_length=avg_label_length,
    )


if __name__ == "__main__":
    parser = ArgumentParser(ComputeBitrateArgs)
    parser.description = "Compute bitrate metrics from label sequences stored in files"
    parser.epilog = "See the implementation of compute_bitrate.py for more details"
    args = parser.parse_args()
    compute_bitrate(args)
