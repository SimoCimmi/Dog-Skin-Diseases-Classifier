"""Script di valutazione P1: calcola metriche dettagliate e matrice di confusione."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.common import DEVICE, get_loader, get_model


def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader):
    """Esegue l'inferenza e restituisce etichette reali e predette."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            # Ottiene la classe con probabilità maggiore
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, classes: List[str], out_path: Path):
    """Genera e salva una matrice di confusione estetica."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True, help="Path to model.pth")
    parser.add_argument("--data", type=str, required=True, help="Path to test folder")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    loader = get_loader(Path(args.data), args.batch, shuffle=False, is_train=False)

    # Diciamo all'IDE che il dataset dentro il loader è sicuramente un ImageFolder
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
    # Questo risolve "Unexpected type(s):(str)..." dicendo che report è un dizionario
    report: Dict[str, Any] = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True
    )

    metrics = {
        "accuracy": report["accuracy"],
        "f1_score_macro": report["macro avg"]["f1-score"],
        "f1_score_weighted": report["weighted avg"]["f1-score"],
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "per_class": report  # Salviamo tutto il dettaglio
    }

    print(f"[*] Accuracy: {metrics['accuracy']:.4f} | F1 (Weighted): {metrics['f1_score_weighted']:.4f}")

    # 5. Save Artifacts
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    plot_confusion_matrix(y_true, y_pred, classes, out_dir / "confusion_matrix.png")
    print(f"[+] Evaluation artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
