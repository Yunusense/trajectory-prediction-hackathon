import torch
import numpy as np

from src.config import CONFIG
from src.model_social import SocialTrajectoryModel

def predict_sample(history, neighbors, checkpoint_path):
    model = SocialTrajectoryModel(
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        future_steps=CONFIG["future_steps"],
        num_modes=3
    ).to(CONFIG["device"])

    model.load_state_dict(torch.load(checkpoint_path, map_location=CONFIG["device"]))
    model.eval()

    history = torch.tensor(history, dtype=torch.float32).unsqueeze(0).to(CONFIG["device"])
    neighbors = torch.tensor(neighbors, dtype=torch.float32).unsqueeze(0).to(CONFIG["device"])

    with torch.no_grad():
        trajs, mode_logits = model(history, neighbors)
        probs = torch.softmax(mode_logits, dim=-1).cpu().numpy()[0]
        trajs = trajs.cpu().numpy()[0]

    return {
        "mode_probabilities": probs.tolist(),
        "predicted_trajectories": trajs.tolist()
    }
