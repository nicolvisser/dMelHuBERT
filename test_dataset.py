from dmelhubert.dataset import DMelWithLabelsDataset
import matplotlib.pyplot as plt

dataset = DMelWithLabelsDataset(
    waveforms_dir="/mnt/wsl/nvme/datasets/LibriSpeech",
    labels_dir="/mnt/wsl/data/mfcc-labels/k-100/LibriSpeech",
    dmel_min_value=-13.8155,
    dmel_max_value=10.3219,
    waveforms_pattern="dev-clean/**/*.flac",
    labels_pattern="dev-clean/**/*.npy",
    max_duration=float("inf"),
)

dmel, label, seqlen = dataset[0]

print(dmel.shape)
print(label.shape)
print(seqlen)

plt.figure(figsize=(10, 5))
plt.imshow(dmel.T, aspect="auto", origin="lower", interpolation="none", vmin=0, vmax=15)
plt.colorbar(label="DMel value")
plt.xlabel("Frame")
plt.ylabel("Mel bin")
plt.title("DMel spectrogram")
plt.tight_layout()
plt.show()
