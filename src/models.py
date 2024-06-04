import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression

device = ("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size,
                            hidden_size, 
                            num_stacked_layers, 
                            batch_first = True)
        self.fc = nn.Linear(hidden_size,1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x,(h0, c0))
        out = self.fc(out[:, -1, :])
        return out