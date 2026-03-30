import torch
import torch.nn as nn

class SocialTrajectoryModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, future_steps=12, num_modes=3):
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes

        self.agent_gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.neighbor_gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.mode_head = nn.Linear(hidden_size, num_modes)
        self.traj_head = nn.Linear(hidden_size, num_modes * future_steps * 2)

    def forward(self, history, neighbors):
        _, h_agent = self.agent_gru(history)
        agent_feat = h_agent[-1]

        batch_size = neighbors.size(0)
        num_neighbors = neighbors.size(1)
        seq_len = neighbors.size(2)

        neighbors = neighbors.view(batch_size * num_neighbors, seq_len, 2)
        _, h_nei = self.neighbor_gru(neighbors)
        nei_feat = h_nei[-1].view(batch_size, num_neighbors, -1).mean(dim=1)

        fused = torch.cat([agent_feat, nei_feat], dim=-1)
        fused = torch.relu(self.fusion(fused))

        mode_logits = self.mode_head(fused)
        trajs = self.traj_head(fused)
        trajs = trajs.view(batch_size, self.num_modes, self.future_steps, 2)

        return trajs, mode_logits
