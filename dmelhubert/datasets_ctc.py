from pathlib import Path
from typing import List, Tuple

import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset


def insert_blank_ids(ids: List[int], blank_id: int) -> List[int]:
    new_ids = [ids[0]]
    for i in range(1, len(ids)):
        if ids[i] == new_ids[-1]:
            new_ids.append(blank_id)
        new_ids.append(ids[i])
    return new_ids


class DMelHuBERTCTCTrainDataset(Dataset):

    def __init__(
        self,
        dmels_dir: str,
        labels_dir: str,
        blank_idx: int,
        dmels_pattern: str = "**/*.pt",
        labels_pattern: str = "**/*.pt",
        add_blank_ids: bool = True,
    ):
        self.blank_idx = blank_idx
        self.add_blank_ids = add_blank_ids

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
        labels = torch.load(label_path).long()
        if self.add_blank_ids:
            labels = insert_blank_ids(labels.tolist(), self.blank_idx)
            labels = torch.tensor(labels, dtype=torch.long)

        assert dmel.ndim == 2, "dmel files must be 2D tensors"
        assert labels.ndim == 1, "labels files must be 1D tensors"

        seqlen = dmel.shape[0]
        tgtlen = labels.shape[0]

        return dmel, seqlen, labels, tgtlen


class DMelHuBERTCTCTokenizerTrainDataset(Dataset):

    def __init__(
        self,
        dmels_dir: str,
        text_dir: str,
        tokenizer: Tokenizer,
        blank_idx: int,
        dmels_pattern: str = "**/*.pt",
        text_pattern: str = "**/*.txt",
    ):
        self.tokenizer = tokenizer
        self.blank_idx = blank_idx
        # get paths to data
        self.dmels_paths = {p.stem: p for p in Path(dmels_dir).glob(dmels_pattern)}
        self.texts_paths = {p.stem: p for p in Path(text_dir).glob(text_pattern)}
        msg = "DMels and text must have the same stems in their file names"
        assert set(self.dmels_paths.keys()) == set(self.texts_paths.keys()), msg
        self.stems = sorted(list(self.dmels_paths.keys()))

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx: int):
        # get paths to data
        stem = self.stems[idx]
        dmel_path = self.dmels_paths[stem]
        text_path = self.texts_paths[stem]

        # load waveform and prepare dMel tokens
        dmel = torch.load(dmel_path).long()  # [n_frames, n_mels]
        text = text_path.read_text()

        assert dmel.ndim == 2, "dmel files must be 2D tensors"

        ids = self.tokenizer.encode(text).ids
        ids = insert_blank_ids(ids, self.blank_idx)
        ids = torch.tensor(ids, dtype=torch.long)

        seqlen = dmel.shape[0]
        tgtlen = len(ids)

        return dmel, seqlen, ids, tgtlen


def train_ctc_collate_fn(
    batch: List[Tuple[torch.Tensor, int, torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dmel, seqlens, labels, tgtlens = zip(*batch)
    dmel = torch.cat(dmel, dim=0).contiguous()
    seqlens = torch.tensor(seqlens, dtype=torch.long)
    labels = torch.cat(labels, dim=0).contiguous()
    tgtlens = torch.tensor(tgtlens, dtype=torch.long)
    return dmel, seqlens, labels, tgtlens
