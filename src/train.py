"""Script di training P1 con AMP, loss tracking e salvataggio del miglior modello."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from torch import amp, nn, optim
from tqdm import tqdm

from src.common import DEVICE, get_loader, get_model


def validate(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module) -> float:
    """Calcola la loss media sul validation set."""
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def train_one_epoch(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scaler: amp.GradScaler,
) -> float:
    """Esegue un'epoca di training con Mixed Precision."""
    model.train()
    running_loss = 0.0

    # Tqdm per barra di avanzamento
    pbar = tqdm(loader, desc="Training", leave=False)

    for images, targets in pbar:
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)  # set_to_none è più veloce di zero_grad

        with amp.autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / len(loader.dataset)


def save_loss_plot(history: Dict[str, List[float]], out_dir: Path) -> None:
    """Genera il grafico Training vs Validation Loss."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    # Ora accettiamo percorsi distinti per train e validation
    parser.add_argument("--train", type=str, required=True, help="Path to training folder")
    parser.add_argument("--val", type=str, required=True, help="Path to validation folder")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup Dataloaders
    print(f"[*] Loading data from: {args.train} (Train) / {args.val} (Val)")
    train_loader = get_loader(Path(args.train), args.batch, shuffle=True, is_train=True)
    val_loader = get_loader(Path(args.val), args.batch, shuffle=False, is_train=False)

    classes = train_loader.dataset.classes
    print(f"[*] Detected {len(classes)} classes: {classes}")

    # 2. Setup Model & Training Components
    model = get_model(args.model, len(classes))

    # Label Smoothing aiuta con dataset rumorosi/medici
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # AdamW è meglio di Adam standard
    scaler = amp.GradScaler()  # Fondamentale per 8GB VRAM

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    start_time = time.time()

    # 3. Training Loop
    print(f"[*] Starting training for {args.epochs} epochs on {DEVICE}...")

    for epoch in range(args.epochs):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        v_loss = validate(model, val_loader, criterion)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        # Logic: Save BEST model only
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), out_dir / "model.pth")
            saved_msg = "(*)"  # Indicatore visuale
        else:
            saved_msg = ""

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"Train Loss: {t_loss:.4f} | "
            f"Val Loss: {v_loss:.4f} {saved_msg}"
        )

    total_time = (time.time() - start_time) / 60
    print(f"[*] Training complete in {total_time:.1f} min. Best Val Loss: {best_val_loss:.4f}")

    # 4. Save Artifacts
    save_loss_plot(history, out_dir)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=4)


if __name__ == "__main__":
    main()
