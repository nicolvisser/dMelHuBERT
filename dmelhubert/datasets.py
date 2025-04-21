from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from .dmel import DMel
from .mfcc import mfcc


class Wav2DMelDataset(Dataset):
    def __init__(
        self,
        waveforms_dir: str,
        waveforms_pattern: str = "**/*.flac",
        n_mels: int = 80,
        amin: float = 1e-6,
        codebook_size: int = 16,
        codebook_min_value: float = -157.5950,
        codebook_max_value: float = -69.8588,
    ):
        self.waveforms_dir = Path(waveforms_dir)
        self.waveforms_paths = {
            p.stem: p for p in self.waveforms_dir.glob(waveforms_pattern)
        }
        self.stems = sorted(list(self.waveforms_paths.keys()))

        self.dmel = DMel(
            n_mels=n_mels,
            amin=amin,
            codebook_size=codebook_size,
            codebook_min_value=codebook_min_value,
            codebook_max_value=codebook_max_value,
        )

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx: int):
        # get paths to data
        stem = self.stems[idx]
        waveform_path = self.waveforms_paths[stem]
        relative_path = waveform_path.relative_to(self.waveforms_dir).with_suffix("")

        # load waveform and prepare dMel tokens
        waveform, sample_rate = torchaudio.load(waveform_path)
        assert sample_rate == 16000, "Sample rate must be 16000"
        assert waveform.shape[0] == 1, "Waveform must be mono"
        dmel = self.dmel.forward(waveform[0])

        seqlen = dmel.shape[0]

        return relative_path, dmel, seqlen


class Wav2MFCCDataset(Dataset):
    def __init__(
        self,
        waveforms_dir: str,
        waveforms_pattern: str = "**/*.flac",
    ):
        self.waveforms_dir = Path(waveforms_dir)
        self.waveforms_paths = {
            p.stem: p for p in self.waveforms_dir.glob(waveforms_pattern)
        }
        self.stems = sorted(list(self.waveforms_paths.keys()))

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx: int):
        # get paths to data
        stem = self.stems[idx]
        waveform_path = self.waveforms_paths[stem]
        relative_path = waveform_path.relative_to(self.waveforms_dir).with_suffix("")

        # load waveform and prepare dMel tokens
        waveform, sample_rate = torchaudio.load(waveform_path)
        assert sample_rate == 16000, "Sample rate must be 16000"
        assert waveform.shape[0] == 1, "Waveform must be mono"
        mfccs = mfcc(waveform)

        seqlen = mfccs.shape[0]

        return relative_path, mfccs, seqlen


class Wav2MFCCLabelsDataset(Dataset):
    def __init__(
        self,
        waveforms_dir: str,
        codebook_path: str,
        waveforms_pattern: str = "**/*.flac",
    ):
        self.mfcc_dataset = Wav2MFCCDataset(waveforms_dir, waveforms_pattern)
        self.codebook = torch.load(codebook_path)
        assert self.codebook.ndim == 2, "Codebook must be 2D"
        assert self.codebook.shape[1] == 39, "Codebook must have 39 dimensional vectors"

    def __len__(self):
        return len(self.mfcc_dataset)

    def __getitem__(self, idx: int):
        relative_path, mfccs, seqlen = self.mfcc_dataset[idx]
        # find the nearest codebook vector to the mfccs
        dists = torch.cdist(mfccs, self.codebook)
        labels = torch.argmin(dists, dim=-1)
        return relative_path, labels, seqlen


class DMelHuBERTTrainDataset(Dataset):
    """
    Dataset for training a DMel-based model.
    Loads the precomputed dMel tokens and target labels.
    """

    def __init__(
        self,
        dmels_dir: str,
        labels_dir: str,
        dmels_pattern: str = "**/*.pt",
        labels_pattern: str = "**/*.pt",
        max_duration: float = float("inf"),
        frame_rate: float = 50,
    ):
        self.max_duration = max_duration
        self.frame_rate = frame_rate

        # get paths to data
        self.dmels_paths = {p.stem: p for p in Path(dmels_dir).glob(dmels_pattern)}
        self.labels_paths = {p.stem: p for p in Path(labels_dir).glob(labels_pattern)}
        msg = "DMels and labels must have the same stems in their file names"
        assert set(self.dmels_paths.keys()) == set(self.labels_paths.keys()), msg
        self.stems = sorted(list(self.dmels_paths.keys()))

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx: int):
        # get paths to data
        stem = self.stems[idx]
        dmel_path = self.dmels_paths[stem]
        label_path = self.labels_paths[stem]

        # load waveform and prepare dMel tokens
        dmel = torch.load(dmel_path).long()  # [n_frames, n_mels]
        labels = torch.load(label_path).long()  # [n_frames]

        assert dmel.ndim == 2, "dmel files must be 2D tensors"
        assert labels.ndim == 1, "labels files must be 1D tensors"

        # check lengths and make sure they match
        assert (
            abs(dmel.shape[0] - labels.shape[0]) <= 2
        ), "dMel and labels must not differ by more than 2 frames. Check lengths of labels."
        seqlen = min(dmel.shape[0], labels.shape[0])
        dmel = dmel[:seqlen, :]
        labels = labels[:seqlen]

        # limit max length if specified
        max_num_frames = (
            int(self.max_duration * self.frame_rate)
            if self.max_duration < float("inf")
            else seqlen
        )
        if seqlen > max_num_frames:
            start = np.random.randint(0, seqlen - max_num_frames)
            dmel = dmel[start : start + max_num_frames, :]
            labels = labels[start : start + max_num_frames]
            seqlen = max_num_frames

        return dmel, labels, seqlen


def train_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    dmel, labels, seqlens = zip(*batch)
    dmel = torch.cat(dmel, dim=0).contiguous()
    labels = torch.cat(labels, dim=0).contiguous()
    return dmel, labels, list(seqlens)
