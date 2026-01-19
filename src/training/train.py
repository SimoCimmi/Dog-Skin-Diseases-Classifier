"""Training script for P1 with AMP, loss tracking, and best model saving."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from torch import amp, nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.common import DEVICE, get_loader, get_model


def validate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module
) -> float:
    """Calculate average loss on the validation set.

    Args:
        model: The neural network model.
        loader: DataLoader for the validation set.
        criterion: Loss function.

    Returns:
        The average loss.

    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in loader:
            # Soluzione PLW2901: nomi diversi per evitare l'overwriting
            imgs, lbls = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler,
) -> float:
    """Perform one training epoch using Mixed Precision.

    Args:
        model: The neural network model.
        loader: DataLoader for the training set.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        scaler: AMP GradScaler for stable training.

    Returns:
        The average training loss for the epoch.

    """
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)

    for images, targets in pbar:
        # Soluzione PLW2901: nomi diversi per le variabili inviate al device
        imgs, lbls = images.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with amp.autocast(device_type=device_type):
            outputs = model(imgs)
            loss = criterion(outputs, lbls)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / len(loader.dataset)


def save_loss_plot(history: Dict[str, List[float]], out_dir: Path) -> None:
    """Generate and save the Training vs Validation Loss plot.

    Args:
        history: Dictionary containing loss lists.
        out_dir: Directory where the plot will be saved.

    """
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Training Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Validation Loss", linewidth=2, linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training Dynamics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "loss_plot.png", dpi=300)
    plt.close()


def main() -> None:
    """Entry point for the training script."""
    parser = argparse.ArgumentParser(description="P1 Training Script")
    parser.add_argument("--model", type=str, required=True, help="Model architecture")
    parser.add_argument("--train", type=str, required=True, help="Path to train folder")
    parser.add_argument("--val", type=str, required=True, help="Path to val folder")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading data: {args.train} (Train) / {args.val} (Val)")
    train_loader = get_loader(Path(args.train), args.batch, shuffle=True, is_train=True)
    val_loader = get_loader(Path(args.val), args.batch, shuffle=False, is_train=False)

    classes = train_loader.dataset.classes
    print(f"[*] Detected {len(classes)} classes: {classes}")

    model = get_model(args.model, len(classes))

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = amp.GradScaler()

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    start_time = time.time()

    print(f"[*] Starting training for {args.epochs} epochs on {DEVICE}...")

    for epoch in range(args.epochs):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        v_loss = validate(model, val_loader, criterion)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), out_dir / "model.pth")
            saved_msg = "(*)"
        else:
            saved_msg = ""

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"Train Loss: {t_loss:.4f} | "
            f"Val Loss: {v_loss:.4f} {saved_msg}"
        )

    total_time = (time.time() - start_time) / 60
    print(f"[*] Training complete in {total_time:.1f} min. "
          f"Best Val Loss: {best_val_loss:.4f}")

    save_loss_plot(history, out_dir)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=4)


if __name__ == "__main__":
    main()
