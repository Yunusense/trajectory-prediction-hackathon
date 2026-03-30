import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import CONFIG
from src.dataset import TrajectoryDataset
from src.model_baseline import BaselineGRU
from src.model_social import SocialTrajectoryModel
from src.utils import ade, fde

def evaluate_baseline(samples, checkpoint_path):
    loader = DataLoader(TrajectoryDataset(samples), batch_size=1, shuffle=False)

    model = BaselineGRU(
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        future_steps=CONFIG["future_steps"]
    ).to(CONFIG["device"])
    model.load_state_dict(torch.load(checkpoint_path, map_location=CONFIG["device"]))
    model.eval()

    ades, fdes = [], []

    with torch.no_grad():
        for batch in loader:
            history = batch["history"].to(CONFIG["device"])
            future = batch["future"].cpu().numpy()[0]
            pred = model(history).cpu().numpy()[0]

            ades.append(ade(pred, future))
            fdes.append(fde(pred, future))

    return {
        "ADE": float(np.mean(ades)),
        "FDE": float(np.mean(fdes))
    }

def evaluate_social(samples, checkpoint_path):
    loader = DataLoader(TrajectoryDataset(samples), batch_size=1, shuffle=False)

    model = SocialTrajectoryModel(
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        future_steps=CONFIG["future_steps"],
        num_modes=3
    ).to(CONFIG["device"])
    model.load_state_dict(torch.load(checkpoint_path, map_location=CONFIG["device"]))
    model.eval()

    ades, fdes = [], []

    with torch.no_grad():
        for batch in loader:
            history = batch["history"].to(CONFIG["device"])
            future = batch["future"].cpu().numpy()[0]
            neighbors = batch["neighbors"].to(CONFIG["device"])

            trajs, mode_logits = model(history, neighbors)
            pred_modes = trajs.cpu().numpy()[0]

            sample_ades = [ade(p, future) for p in pred_modes]
            sample_fdes = [fde(p, future) for p in pred_modes]

            ades.append(min(sample_ades))
            fdes.append(min(sample_fdes))

    return {
        "minADE@3": float(np.mean(ades)),
        "minFDE@3": float(np.mean(fdes))
    }
