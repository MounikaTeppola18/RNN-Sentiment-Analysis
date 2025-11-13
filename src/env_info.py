import torch
import random
import numpy as np
import psutil

# -----------------------------
# Fix seeds
# -----------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
print("Random seeds fixed: torch, numpy, random")

# -----------------------------
# Report hardware info
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ram_gb = round(psutil.virtual_memory().total / 1e9, 2)

print(f"Device: {device}") 
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print(f"RAM Size: {ram_gb} GB")
