import torch
import torch.nn as nn
import torchaudio


class Mel(nn.Module):
    def __init__(self, n_mels: int = 80, amin: float = 1e-10):
        super().__init__()
        self.wav2mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=400,
            hop_length=320,
            f_min=0,
            f_max=8000,
            pad=0,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=False,
            wkwargs=None,
            center=True,
            pad_mode="reflect",
            onesided=None,
            norm=None,
            mel_scale="htk",
        )
        self.amin = amin

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.wav2mel(waveform)
        log_mel = torchaudio.functional.amplitude_to_DB(
            mel,
            multiplier=10.0,
            amin=self.amin,
            db_multiplier=10.0,
            top_db=None,
        )
        return log_mel.transpose(-2, -1)


class DMel(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        amin: float = 1e-6,
        codebook_size: int = 16,
        codebook_min_value: float = -157.5950,
        codebook_max_value: float = -69.8588,
    ):
        super().__init__()
        self.mel = Mel(n_mels=n_mels, amin=amin)
        self.codebook = torch.linspace(
            codebook_min_value, codebook_max_value, codebook_size
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        assert waveform.ndim <= 2, "Waveform must be 1D or 2D"
        unbatched = waveform.ndim == 1
        if unbatched:
            waveform = waveform.unsqueeze(0)  # 1 x n_samples
        mel = self.mel.forward(waveform)  # bsz x n_feats x n_mels
        mel = mel.unsqueeze(-1)  # bsz x n_feats x n_mels x 1
        codebook = (
            self.codebook.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )  # 1 x 1 x 1 x n_codes
        dmel = torch.argmin((mel - codebook) ** 2, dim=-1)  # bsz x n_feats x n_mels
        if unbatched:
            dmel = dmel.squeeze(0)  # n_feats x n_mels
        return dmel
