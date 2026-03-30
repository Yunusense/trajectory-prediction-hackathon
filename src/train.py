import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CONFIG
from src.utils import ensure_dir, set_seed
from src.dataset import TrajectoryDataset
from src.model_baseline import BaselineGRU
from src.model_social import SocialTrajectoryModel

def train_baseline(train_samples, val_samples, checkpoint_path):
    set_seed(CONFIG["random_seed"])

    train_loader = DataLoader(TrajectoryDataset(train_samples), batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(TrajectoryDataset(val_samples), batch_size=CONFIG["batch_size"], shuffle=False)

    model = BaselineGRU(
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        future_steps=CONFIG["future_steps"]
    ).to(CONFIG["device"])

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.MSELoss()

    best_val = float("inf")
    ensure_dir(os.path.dirname(checkpoint_path))

    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Baseline Epoch {epoch+1}"):
            history = batch["history"].to(CONFIG["device"])
            future = batch["future"].to(CONFIG["device"])

            optimizer.zero_grad()
            pred = model(history)
            loss = criterion(pred, future)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                history = batch["history"].to(CONFIG["device"])
                future = batch["future"].to(CONFIG["device"])
                pred = model(history)
                loss = criterion(pred, future)
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: val_loss={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), checkpoint_path)

def train_social(train_samples, val_samples, checkpoint_path):
    set_seed(CONFIG["random_seed"])

    train_loader = DataLoader(TrajectoryDataset(train_samples), batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(TrajectoryDataset(val_samples), batch_size=CONFIG["batch_size"], shuffle=False)

    model = SocialTrajectoryModel(
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        future_steps=CONFIG["future_steps"],
        num_modes=3
    ).to(CONFIG["device"])

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.MSELoss()

    best_val = float("inf")
    ensure_dir(os.path.dirname(checkpoint_path))

    for epoch in range(CONFIG["epochs"]):
        model.train()
        for batch in tqdm(train_loader, desc=f"Social Epoch {epoch+1}"):
            history = batch["history"].to(CONFIG["device"])
            future = batch["future"].to(CONFIG["device"])
            neighbors = batch["neighbors"].to(CONFIG["device"])

            optimizer.zero_grad()
            trajs, mode_logits = model(history, neighbors)
            pred = trajs[:, 0]
            loss = criterion(pred, future)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                history = batch["history"].to(CONFIG["device"])
                future = batch["future"].to(CONFIG["device"])
                neighbors = batch["neighbors"].to(CONFIG["device"])

                trajs, mode_logits = model(history, neighbors)
                pred = trajs[:, 0]
                loss = criterion(pred, future)
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: val_loss={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), checkpoint_path)
