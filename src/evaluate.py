"""Evaluation script for P1: calculates detailed metrics and confusion matrix."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.common import DEVICE, get_loader, get_model


def evaluate(
    model: torch.nn.Module, loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute inference and return true and predicted labels.

    Args:
        model: The trained neural network model.
        loader: DataLoader for the test set.

    Returns:
        A tuple containing (true_labels, predicted_labels).

    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            # Soluzione PLW2901: usa un nome diverso per l'input spostato sul device
            imgs = inputs.to(DEVICE)
            outputs = model(imgs)

            # Ottiene la classe con probabilità maggiore
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], out_path: Path
) -> None:
    """Generate and save an aesthetic confusion matrix plot.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels from the model.
        classes: List of class names for axis labels.
        out_path: Full path where the image will be saved.

    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    """Run the evaluation pipeline and save metrics and plots."""
    parser = argparse.ArgumentParser(description="P1 Evaluation Stage")
    parser.add_argument("--model", type=str, required=True, help="Model architecture")
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to model.pth"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to test folder"
    )
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    loader = get_loader(Path(args.data), args.batch, shuffle=False, is_train=False)

    # Cast esplicito per rassicurare il linter sul tipo di dataset
    dataset = cast(ImageFolder, loader.dataset)
    classes = dataset.classes

    # 2. Load Model
    print(f"[*] Loading model weights from {args.weights}")
    model = get_model(args.model, len(classes))
    state_dict = torch.load(args.weights, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)

    # 3. Evaluate
    y_true, y_pred = evaluate(model, loader)

    # 4. Calculate Metrics
    report: Dict[str, Any] = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True
    )

    metrics = {
        "accuracy": report["accuracy"],
        "f1_score_macro": report["macro avg"]["f1-score"],
        "f1_score_weighted": report["weighted avg"]["f1-score"],
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "per_class": report
    }

    print(
        f"[*] Accuracy: {metrics['accuracy']:.4f} | "
        f"F1 (Weighted): {metrics['f1_score_weighted']:.4f}"
    )

    # 5. Save Artifacts
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    plot_confusion_matrix(y_true, y_pred, classes, out_dir / "confusion_matrix.png")
    print(f"[+] Evaluation artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
