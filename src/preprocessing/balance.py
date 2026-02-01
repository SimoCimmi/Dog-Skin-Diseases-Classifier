"""Script per il bilanciamento del dataset (K-Means Undersampling)."""

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from torchvision import models, transforms
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_feature_extractor() -> nn.Module:
    """Prepara una ResNet18 per estrarre feature (rimuove l'ultimo layer fc)."""
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    # Rimuoviamo il layer di classificazione (FC) per avere le feature pure
    modules = list(model.children())[:-1]
    feature_extractor = nn.Sequential(*modules)
    feature_extractor.to(DEVICE)
    feature_extractor.eval()
    return feature_extractor


def extract_features(
        image_paths: List[Path], model: nn.Module, batch_size: int = 32
) -> np.ndarray:
    """Estrae i vettori delle feature per una lista di immagini."""
    # Preprocessing standard per ImageNet/ResNet (Feature Extraction)
    # Usiamo 256 per coerenza con i pesi di ResNet
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features_list = []

    # Processiamo in batch per velocità
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i: i + batch_size]
        batch_tensors = []

        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            batch_tensors.append(preprocess(img))

        if not batch_tensors:
            continue

        batch_stack = torch.stack(batch_tensors).to(DEVICE)

        with torch.no_grad():
            output = model(batch_stack)
            output = output.flatten(start_dim=1)
            features_list.append(output.cpu().numpy())

    if not features_list:
        return np.array([])

    return np.vstack(features_list)


def get_class_counts(data_dir: Path) -> Dict[str, int]:
    """Conta le immagini per ogni classe."""
    counts = {}
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            files = [
                f
                for f in class_dir.glob("*")
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            counts[class_dir.name] = len(files)
    return counts


def undersample_dataset(input_dir: Path, output_dir: Path, seed: int = 42) -> None:
    """Crea una copia bilanciata usando K-Means Undersampling."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Analisi classi
    counts = get_class_counts(input_dir)
    if not counts:
        print("[!] Nessuna classe trovata o cartella vuota.")
        return

    min_count = min(counts.values())
    print(f"[*] Classi rilevate: {counts}")
    print(f"[*] Target bilanciamento (N): {min_count} immagini per classe")

    # Inizializza modello per estrazione feature
    print(f"[*] Caricamento modello feature extraction su {DEVICE}...")
    model = get_feature_extractor()

    # Elaborazione classi
    for class_name, count in counts.items():
        src_class_dir = input_dir / class_name
        dst_class_dir = output_dir / class_name
        dst_class_dir.mkdir(exist_ok=True)

        all_files = list(src_class_dir.glob("*"))
        images = [f for f in all_files if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]

        # SELEZIONE TRAMITE CLUSTERING
        if len(images) <= min_count:
            # Se la classe è già la minoritaria (o uguale), prendiamo tutto
            selected_images = images
        else:
            print(f"    -> Analisi K-Means per '{class_name}' ({count} -> {min_count})...")

            # Estrazione Feature
            features = extract_features(images, model)

            # K-Means Clustering
            kmeans = KMeans(n_clusters=min_count, random_state=seed, n_init=10)
            kmeans.fit(features)

            # Trova l'immagine più vicina al centro di ogni cluster
            closest_indices, _ = pairwise_distances_argmin_min(
                kmeans.cluster_centers_, features
            )

            # Recupera i path corrispondenti
            selected_indices = sorted(list(set(closest_indices)))

            # Se per caso K-Means ha prodotto meno centroidi unici del richiesto
            if len(selected_indices) < min_count:
                remaining_indices = list(set(range(len(images))) - set(selected_indices))
                needed = min_count - len(selected_indices)
                random.seed(seed)
                extra = random.sample(remaining_indices, needed)
                selected_indices.extend(extra)

            selected_images = [images[i] for i in selected_indices]

        # Copia file
        print(f"    -> Copia file per '{class_name}'...")
        for img_path in tqdm(selected_images, desc=class_name, leave=False):
            shutil.copy2(img_path, dst_class_dir / img_path.name)

    print(f"[+] Dataset bilanciato (K-Means) creato in: {output_dir}")


def main() -> None:
    """Esegue il processo di undersampling basato sugli argomenti da riga di comando."""
    parser = argparse.ArgumentParser(
        description="K-Means Undersampling per Two-Phase Learning"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path dataset originale"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path dataset bilanciato"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed per riproducibilità")

    args = parser.parse_args()

    undersample_dataset(Path(args.input), Path(args.output), args.seed)


if __name__ == "__main__":
    main()