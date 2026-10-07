import torch

from eeg.gesture2hand import EEGNet

model = EEGNet()

x = torch.randn(1, 1, 64, 400)

x = model(x)
print(x.shape)
