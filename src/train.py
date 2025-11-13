import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import os
import random

from models import SimpleRNNModel, LSTMModel, BiLSTMModel

# -----------------------------
# Reproducibility
# -----------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -----------------------------
# Global CSV setup
# -----------------------------
RESULTS_DIR = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)
results_path = os.path.join(RESULTS_DIR, "metrics.csv")

metric_columns = [
    "Model", "Activation", "Optimizer", "SeqLen", "GradClip",
    "Epoch", "Loss", "Accuracy", "F1", "EpochTime(s)", "TotalTime(s)"
]

if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
    pd.DataFrame(columns=metric_columns).to_csv(results_path, index=False)
else:
    with open(results_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
    if not first_line.startswith("Model,Activation"):
        existing_data = pd.read_csv(results_path, header=None, names=metric_columns)
        existing_data.to_csv(results_path, index=False)


# -----------------------------
# Training function
# -----------------------------
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device,
                       architecture, act, opt_name, seq_len, grad_clip=False,
                       results_path="../results/metrics.csv"):
    model.to(device)
    num_epochs = 5
    model.train()

    epoch_losses, epoch_accs, epoch_f1s, epoch_times = [], [], [], []
    start_time_total = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0

        # -------- Training Loop --------
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # -------- Evaluation per epoch --------
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = (outputs.cpu().numpy() > 0.5).astype(int)
                all_preds.extend(preds.flatten())
                all_labels.extend(y_batch.numpy())

        acc_epoch = accuracy_score(all_labels, all_preds)
        f1_epoch = f1_score(all_labels, all_preds, average='macro')
        epoch_time = time.time() - epoch_start

        epoch_losses.append(avg_loss)
        epoch_accs.append(acc_epoch)
        epoch_f1s.append(f1_epoch)
        epoch_times.append(epoch_time)

        print(f"  Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f} | Acc: {acc_epoch:.4f} | F1: {f1_epoch:.4f} | Time: {epoch_time:.2f}s")
        model.train()

    total_time = time.time() - start_time_total

    # -------- Save epoch-level metrics --------
    for i, (loss, acc_epoch, f1_epoch, epoch_time) in enumerate(zip(epoch_losses, epoch_accs, epoch_f1s, epoch_times), start=1):
        pd.DataFrame([{
            "Model": architecture.upper(),
            "Activation": act,
            "Optimizer": opt_name,
            "SeqLen": seq_len,
            "GradClip": "Yes" if grad_clip else "No",
            "Epoch": i,
            "Loss": round(loss, 4),
            "Accuracy": round(acc_epoch, 4),
            "F1": round(f1_epoch, 4),
            "EpochTime(s)": round(epoch_time, 2),
            "TotalTime(s)": round(total_time, 2)
        }]).to_csv(results_path, mode='a', header=False, index=False)

    print(f"Completed: Loss={epoch_losses[-1]:.4f}, Acc={epoch_accs[-1]:.4f}, F1={epoch_f1s[-1]:.4f}, TotalTime={total_time:.2f}s")

    per_epoch_time = total_time / num_epochs
    return epoch_losses[-1], epoch_accs[-1], epoch_f1s[-1], total_time, per_epoch_time, epoch_losses


# -----------------------------
# Main script
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, required=True,
                        choices=["rnn", "lstm", "bilstm"])
    args = parser.parse_args()

    architecture = args.architecture.lower()
    activations = ["relu", "tanh", "sigmoid"]
    optimizers_list = ["adam", "sgd", "rmsprop"]
    seq_lengths = [25, 50, 100]
    grad_clipping = [False, True]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running compact experiments for: {architecture.upper()}")

    # --- Select ~18 runs per architecture (so total ~54) ---
    selected_combos = []
    for act in activations:
        for opt_name in optimizers_list:
            for seq_len in seq_lengths:
                for clip in grad_clipping:
                    selected_combos.append((act, opt_name, seq_len, clip))
                    if len(selected_combos) >= 18:
                        break
                if len(selected_combos) >= 18:
                    break
            if len(selected_combos) >= 18:
                break
        if len(selected_combos) >= 18:
            break

    for run_id, (act, opt_name, seq_len, clip) in enumerate(selected_combos, start=1):
        print(f"\n[Run {run_id}/{len(selected_combos)}] Model={architecture.upper()} | Act={act} | Opt={opt_name} | Seq={seq_len} | Clip={clip}")

        # -------- Load Data --------
        data = np.load(f"../data/imdb_seq{seq_len}.npz")
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long),
                                      torch.tensor(y_train, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.long),
                                     torch.tensor(y_test, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # -------- Model Setup --------
        vocab_size = 10000
        if architecture == "rnn":
            model = SimpleRNNModel(vocab_size=vocab_size, activation=act)
        elif architecture == "lstm":
            model = LSTMModel(vocab_size=vocab_size, activation=act)
        else:
            model = BiLSTMModel(vocab_size=vocab_size, activation=act)

        # -------- Optimizer --------
        if opt_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        elif opt_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        else:
            optimizer = optim.RMSprop(model.parameters(), lr=0.001)

        criterion = nn.BCELoss()

        # -------- Train --------
        final_loss, acc, f1, total_time, per_epoch_time, epoch_losses = train_and_evaluate(
            model, train_loader, test_loader, criterion, optimizer, device,
            architecture=architecture, act=act, opt_name=opt_name, seq_len=seq_len,
            grad_clip=clip, results_path=results_path
        )

        # Save loss curve for later plotting
        loss_path = os.path.join(RESULTS_DIR, f"loss_{architecture}_{act}_{opt_name}_{seq_len}_{'clip' if clip else 'noclip'}.csv")
        pd.DataFrame({"Epoch": np.arange(1, len(epoch_losses) + 1), "Loss": epoch_losses}).to_csv(loss_path, index=False)

    print(f"\nAll compact runs completed for {architecture.upper()}.")


if __name__ == "__main__":
    main()
