import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(RNN).__init__(*args, **kwargs)