"""Script di valutazione P1: calcola metriche, matrici e tabelle riassuntive."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pandas.plotting import table
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.training.common import DEVICE, get_loader, get_model


def evaluate(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader
) -> Tuple[np.ndarray, np.ndarray]:
    """Esegue l'inferenza e restituisce etichette reali e predette."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            # Soluzione PLW2901: rinominato inputs in imgs
            imgs = inputs.to(DEVICE)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], out_path: Path
) -> None:
    """Genera e salva una matrice di confusione estetica."""
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


def save_metrics_table(report: Dict[str, Any], out_path: Path) -> None:
    """Genera una tabella PNG professionale e un CSV usando Pandas Plotting."""
    # 1. Preparazione DataFrame
    df = pd.DataFrame(report).transpose()

    # Rimuoviamo righe non necessarie per la tabella
    if "accuracy" in df.index:
        acc_val = df.loc["accuracy", "f1-score"]
        df = df.drop("accuracy")
    else:
        acc_val = report.get("accuracy", 0.0)

    # Ordine e selezione colonne
    cols = ["precision", "recall", "f1-score", "support"]
    df = df[cols]

    # Salvataggio CSV
    csv_path = out_path.with_suffix(".csv")
    df.to_csv(csv_path)

    # 2. Creazione Plot
    # Calcolo altezza dinamica basata sul numero di righe
    h = len(df) * 0.5 + 2
    # Soluzione RUF059: aggiunto underscore a fig
    _fig, ax = plt.subplots(figsize=(10, h))

    # Nascondiamo gli assi (vogliamo solo la tabella)
    ax.axis("off")

    # Soluzione PLC0415: import spostato in alto
    tbl = table(
        ax,
        df.round(4),  # Passiamo direttamente il DataFrame arrotondato
        loc="center",
        cellLoc="center"
    )

    # Styling
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.5)

    plt.title(f"Classification Report (Global Accuracy: {acc_val:.2%})", fontsize=14)

    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"[+] Table saved to: {out_path}")


def main() -> None:
    """Entry point per l'esecuzione dello script di valutazione."""
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

    # Casting esplicito
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
        "per_class": report
    }

    # 5. Save Artifacts
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    plot_confusion_matrix(y_true, y_pred, classes, out_dir / "confusion_matrix.png")

    save_metrics_table(report, out_dir / "classification_report.png")

    print(f"\n[*] Global Accuracy: {metrics['accuracy']:.2%}")
    print(f"[*] Artifacts saved in: {out_dir}")


if __name__ == "__main__":
    main()
