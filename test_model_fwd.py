from torch.utils.data import DataLoader

from dmelhubert.dataset import DMelWithLabelsDataset, collate_fn
from dmelhubert.model import DMelHuBERT
import torch

dataset = DMelWithLabelsDataset(
    waveforms_dir="/mnt/wsl/nvme/datasets/LibriSpeech",
    labels_dir="/mnt/wsl/data/mfcc-labels/k-100/LibriSpeech",
    dmel_min_value=-13.8155,
    dmel_max_value=10.3219,
    waveforms_pattern="dev-clean/**/*.flac",
    labels_pattern="dev-clean/**/*.npy",
    max_duration=float("inf"),
)

model = DMelHuBERT(num_label_embeddings=100, mask=True).cuda()

dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

for batch in dataloader:
    dmel, labels, seqlens = batch
    print(dmel.shape)
    print(labels.shape)
    print(seqlens)
    output, mask = model.forward(dmel=dmel.cuda(), seqlens=seqlens)
    print(output.shape)
    loss = torch.nn.functional.cross_entropy(output, labels.cuda())
    print(loss)
    break

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.imshow(dmel.cpu().numpy().T, aspect="auto", origin="lower", interpolation="none")
plt.colorbar()
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(
    mask.unsqueeze(0).cpu().numpy(), aspect="auto", origin="lower", interpolation="none"
)
plt.colorbar()
plt.show()
