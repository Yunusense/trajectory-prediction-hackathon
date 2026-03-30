import torch
import torch.nn as nn

class BaselineGRU(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, future_steps=12):
        super().__init__()
        self.future_steps = future_steps
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, future_steps * 2)

    def forward(self, x):
        _, h = self.gru(x)
        h_last = h[-1]
        out = self.fc(h_last)
        out = out.view(x.size(0), self.future_steps, 2)
        return out
