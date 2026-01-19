"""Script per il bilanciamento del dataset (Random Undersampling)."""

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict

from tqdm import tqdm


def get_class_counts(data_dir: Path) -> Dict[str, int]:
    """Conta le immagini per ogni classe."""
    counts = {}
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            # Conta solo file immagine validi
            files = [f for f in class_dir.glob("*") if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
            counts[class_dir.name] = len(files)
    return counts


def undersample_dataset(input_dir: Path, output_dir: Path, seed: int = 42):
    """Crea una copia bilanciata del dataset usando Random Undersampling."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # 1. Analisi classi
    counts = get_class_counts(input_dir)
    min_count = min(counts.values())
    print(f"[*] Classi rilevate: {counts}")
    print(f"[*] Target bilanciamento (N): {min_count} immagini per classe")

    random.seed(seed)

    # 2. Copia dei file (o creazione dataset)
    for class_name, count in counts.items():
        src_class_dir = input_dir / class_name
        dst_class_dir = output_dir / class_name
        dst_class_dir.mkdir(exist_ok=True)

        all_files = list(src_class_dir.glob("*"))

        # Filtriamo solo estensioni immagini
        images = [f for f in all_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]

        # Selezioniamo casualmente N immagini (dove N = min_count)
        selected_images = random.sample(images, min_count)

        print(f"    -> Copia {len(selected_images)}/{count} immagini per '{class_name}'...")

        for img_path in tqdm(selected_images, desc=class_name, leave=False):
            shutil.copy2(img_path, dst_class_dir / img_path.name)

    print(f"[+] Dataset bilanciato creato in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Random Undersampling per Two-Phase Learning")
    parser.add_argument("--input", type=str, required=True, help="Path dataset originale/deduplicato")
    parser.add_argument("--output", type=str, required=True, help="Path dataset bilanciato")
    parser.add_argument("--seed", type=int, default=42, help="Seed per riproducibilità")

    args = parser.parse_args()

    undersample_dataset(Path(args.input), Path(args.output), args.seed)


if __name__ == "__main__":
    main()