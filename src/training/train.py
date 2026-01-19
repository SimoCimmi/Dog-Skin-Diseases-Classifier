"""Training script for P1/P2 with AMP, loss tracking, and best model saving."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sized, cast

import matplotlib.pyplot as plt
import torch
from torch import amp, nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.training.common import DEVICE, get_loader, get_model


def validate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module
) -> float:
    """Calculate average loss on the validation set."""
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in loader:
            imgs, lbls = images.to(DEVICE), targets.to(DEVICE)

            # Autocast anche in validazione per coerenza
            with amp.autocast(device_type="cuda" if torch.cuda.is_available()
            else "cpu"):
                outputs = model(imgs)
                loss = criterion(outputs, lbls)

            running_loss += loss.item() * images.size(0)

    # Casting per evitare warning IDE "Expected Sized"
    num_samples = len(cast(Sized, cast(object, loader.dataset)))
    return running_loss / num_samples


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler,
) -> float:
    """Perform one training epoch using Mixed Precision."""
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)

    for images, targets in pbar:
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

    # Casting per evitare warning IDE
    num_samples = len(cast(Sized, cast(object, loader.dataset)))
    return running_loss / num_samples


def save_loss_plot(history: Dict[str, List[float]], out_dir: Path) -> None:
    """Generate and save the Training vs. Validation Loss plot."""
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
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--model", type=str, required=True,
                        help="Model architecture")
    parser.add_argument("--train", type=str, required=True,
                        help="Path to train folder")
    parser.add_argument("--val", type=str, required=True,
                        help="Path to val folder")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory")
    # NUOVO ARGOMENTO (Opzionale, default None)
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to initial weights (for Fine-Tuning)")

    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading data: {args.train} (Train) / {args.val} (Val)")
    train_loader = get_loader(Path(args.train), args.batch, shuffle=True, is_train=True)
    val_loader = get_loader(Path(args.val), args.batch, shuffle=False, is_train=False)

    # Casting esplicito per evitare warning su .classes
    train_dataset = cast(ImageFolder, train_loader.dataset)
    classes = train_dataset.classes
    print(f"[*] Detected {len(classes)} classes: {classes}")

    # 1. Inizializza Modello
    model = get_model(args.model, len(classes))

    # 2. (NUOVO) Carica pesi se specificati (Logica per Fase 2)
    if args.weights:
        print(f"[*] Loading Fine-Tuning weights from: {args.weights}")
        state_dict = torch.load(args.weights, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)

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
