import torch
import torchaudio


def mfcc(waveform: torch.Tensor) -> torch.Tensor:
    mfccs = torchaudio.compliance.kaldi.mfcc(
        waveform=waveform,
        sample_frequency=16000,
        use_energy=False,
        frame_shift=20,
    )
    mfccs = mfccs.transpose(0, 1)  # (freq, time)
    deltas = torchaudio.functional.compute_deltas(mfccs)
    ddeltas = torchaudio.functional.compute_deltas(deltas)
    concat = torch.cat([mfccs, deltas, ddeltas], dim=0)
    concat = concat.transpose(0, 1).contiguous()  # (time, freq)
    return concat
