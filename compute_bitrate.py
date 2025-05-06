"""Compute bitrate metrics from label sequences stored in files.

This module provides utilities to compute and compare bitrate statistics (entropy, bitrate, average label length)
for duplicated and deduplicated label sequences. It defines a dataclass BitrateResult to store the results and
a function compute_bitrate to perform the computation given a directory of label files.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm


@dataclass
class BitrateResult:
    """Data class for storing bitrate computation results.

    Attributes:
        probs_duped (torch.Tensor): Probability distribution of labels for duplicated sequences.
        probs_deduped (torch.Tensor): Probability distribution of labels for deduplicated sequences.
        entropy_duped (torch.Tensor): Entropy in bits for duplicated labels.
        entropy_deduped (torch.Tensor): Entropy in bits for deduplicated labels.
        bitrate_duped (torch.Tensor): Bitrate in bits per second for duplicated labels.
        bitrate_deduped (torch.Tensor): Bitrate in bits per second for deduplicated labels.
        avg_label_length (torch.Tensor): Average duration in seconds of labels after deduplication.
    """

    probs_duped: torch.Tensor
    probs_deduped: torch.Tensor
    entropy_duped: torch.Tensor
    entropy_deduped: torch.Tensor
    bitrate_duped: torch.Tensor
    bitrate_deduped: torch.Tensor
    avg_label_length: torch.Tensor


def compute_bitrate(
    labels_dir: str,
    labels_pattern: str,
    n_label_types: int,
    labels_rate: float = 50.0,
):
    """Compute bitrate metrics from label files.

    Parameters:
        labels_dir (str or Path): Directory containing label files (.pt format).
        labels_pattern (str): Glob pattern to match label .pt files within labels_dir. (e.g. "**/*.pt")
        n_label_types (int): Number of distinct label types.
        labels_rate (float): Sampling rate (Hz) of the label sequences. Defaults to 50 Hz.

    Returns:
        BitrateResult: An object containing:
            probs_duped (torch.Tensor): Probability distribution of duplicated labels.
            probs_deduped (torch.Tensor): Probability distribution of deduplicated labels.
            entropy_duped (torch.Tensor): Entropy (bits) of duplicated labels.
            entropy_deduped (torch.Tensor): Entropy (bits) of deduplicated labels.
            bitrate_duped (torch.Tensor): Bitrate (bits/sec) for duplicated labels.
            bitrate_deduped (torch.Tensor): Bitrate (bits/sec) for deduplicated labels.
            avg_label_length (torch.Tensor): Average duration (sec) of labels after deduplication.

    This function loads label sequences from files, performs consecutive deduplication,
    accumulates counts, computes probabilities, entropies, bitrates, and average label length.
    """
    assert Path(labels_dir).exists(), f"Labels directory {labels_dir} does not exist"
    assert Path(
        labels_dir
    ).is_dir(), f"Labels directory {labels_dir} is not a directory"
    assert labels_pattern.endswith(".pt"), "labels_pattern must match .pt files"

    labels_paths = sorted(list(Path(labels_dir).glob(labels_pattern)))
    assert (
        len(labels_paths) > 0
    ), f"No label files found in {labels_dir} matching {labels_pattern}"

    n_labels_duped = 0
    n_labels_deduped = 0
    counts_duped = torch.zeros(n_label_types, dtype=torch.int64)
    counts_deduped = torch.zeros(n_label_types, dtype=torch.int64)
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
        counts_duped += torch.bincount(labels_duped, minlength=n_label_types)
        counts_deduped += torch.bincount(labels_deduped, minlength=n_label_types)
        durations.extend([d / labels_rate for d in durs.tolist()])

    duration = n_labels_duped / labels_rate

    probs_duped = 1e-10 + counts_duped / n_labels_duped
    probs_deduped = 1e-10 + counts_deduped / n_labels_deduped

    entropy_duped = -torch.sum(probs_duped * torch.log2(probs_duped))
    entropy_deduped = -torch.sum(probs_deduped * torch.log2(probs_deduped))

    bitrate_duped = n_labels_duped * entropy_duped / duration
    bitrate_deduped = n_labels_deduped * entropy_deduped / duration

    avg_label_length = duration / n_labels_deduped

    return BitrateResult(
        probs_duped=probs_duped,
        probs_deduped=probs_deduped,
        entropy_duped=entropy_duped,
        entropy_deduped=entropy_deduped,
        bitrate_duped=bitrate_duped,
        bitrate_deduped=bitrate_deduped,
        avg_label_length=avg_label_length,
    )
