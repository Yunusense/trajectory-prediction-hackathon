import numpy as np
import torch
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        history = torch.tensor(sample["history"], dtype=torch.float32)
        future = torch.tensor(sample["future"], dtype=torch.float32)

        neighbors = sample.get("neighbors", [])
        if len(neighbors) == 0:
            neighbors = np.zeros((1, history.shape[0], 2), dtype=np.float32)
        neighbors = torch.tensor(neighbors, dtype=torch.float32)

        return {
            "history": history,
            "future": future,
            "neighbors": neighbors
        }
