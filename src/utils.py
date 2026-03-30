import os
import json
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def ade(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)
    return np.mean(np.linalg.norm(pred - gt, axis=-1))

def fde(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)
    return np.linalg.norm(pred[-1] - gt[-1])
