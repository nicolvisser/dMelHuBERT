from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

N_MELS = 80
CODEBOOK_SIZE = 16


class DMelWithLabelsDataset(Dataset):
    def __init__(
        self,
        waveforms_dir: str,
        labels_dir: str,
        dmel_min_value: float = -13.8155,
        dmel_max_value: float = 10.3219,
        waveforms_pattern: str = "**/*.flac",
        labels_pattern: str = "**/*.npy",
        max_duration: float = float("inf"),
    ):
        self.max_duration = max_duration

        # get paths to data
        self.waveforms_paths = {
            p.stem: p for p in Path(waveforms_dir).glob(waveforms_pattern)
        }
        self.labels_paths = {p.stem: p for p in Path(labels_dir).glob(labels_pattern)}
        msg = "Waveforms and labels must have the same stems in their file names"
        assert set(self.waveforms_paths.keys()) == set(self.labels_paths.keys()), msg
        self.stems = sorted(list(self.waveforms_paths.keys()))

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=400,
            hop_length=320,
            f_min=0,
            f_max=8000,
            pad=0,
            n_mels=N_MELS,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=False,
            wkwargs=None,
            center=False,
            pad_mode="reflect",
            onesided=None,
            norm=None,
            mel_scale="htk",
        )
        self.codebook = torch.linspace(dmel_min_value, dmel_max_value, CODEBOOK_SIZE)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx: int):
        # get paths to data
        stem = self.stems[idx]
        waveform_path = self.waveforms_paths[stem]
        label_path = self.labels_paths[stem]

        # load waveform and prepare dMel tokens
        waveform, sample_rate = torchaudio.load(waveform_path)
        assert sample_rate == 16000, "Sample rate must be 16000"
        assert waveform.shape[0] == 1, "Waveform must be mono"
        mel_spec = self.mel(waveform[0]).T  # T x N
        log_mel_spec = torch.log(mel_spec + 1e-6)  # T x N
        dmel = torch.argmin(
            (log_mel_spec[:, :, None] - self.codebook[None, None, :]) ** 2, dim=2
        )  # T x N

        # load labels
        labels = torch.from_numpy(np.load(label_path)).long()  # [T]

        # check lengths and make sure they match
        assert (
            abs(dmel.shape[0] - labels.shape[0]) <= 2
        ), "Melspec and labels must not differ by more than 2 frames. Check lengths of labels."
        seqlen = min(dmel.shape[0], labels.shape[0])
        dmel = dmel[:seqlen, :]
        labels = labels[:seqlen]

        # limit max length if specified
        max_num_features = (
            int(self.max_duration * 50) if self.max_duration < float("inf") else seqlen
        )
        if seqlen > max_num_features:
            start = np.random.randint(0, seqlen - max_num_features)
            dmel = dmel[start : start + max_num_features, :]
            labels = labels[start : start + max_num_features]
            seqlen = max_num_features

        return dmel, labels, seqlen


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    dmel, labels, seqlens = zip(*batch)
    dmel = torch.cat(dmel, dim=0).contiguous()
    labels = torch.cat(labels, dim=0).contiguous()
    return dmel, labels, list(seqlens)
