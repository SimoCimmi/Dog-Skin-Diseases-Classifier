"""Data augmentation script for Pipeline 2: Single random transformation per image."""

import argparse
import random
import shutil
from pathlib import Path
from typing import Callable, List

import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm


def add_noise(image: Image.Image) -> Image.Image:
    """Add random Gaussian noise to the image."""
    img_array = np.array(image).astype(float)
    # Intensità del rumore casuale tra 2% e 5%
    intensity = random.uniform(0.02, 0.05)
    noise = np.random.normal(loc=0, scale=255 * intensity, size=img_array.shape)
    noisy_img = img_array + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def adjust_saturation(image: Image.Image) -> Image.Image:
    """Adjust color saturation randomly."""
    enhancer = ImageEnhance.Color(image)
    # Fattore tra 1.2 (vivido) e 1.8 (molto saturo)
    factor = random.uniform(1.2, 1.8)
    return enhancer.enhance(factor)


def flip_horizontal(image: Image.Image) -> Image.Image:
    """Apply horizontal flip."""
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def flip_vertical(image: Image.Image) -> Image.Image:
    """Apply vertical flip."""
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def rotate_small(image: Image.Image) -> Image.Image:
    """Rotate image by ± 5 degrees."""
    angle = random.choice([-5, 5])
    return image.rotate(angle, expand=False)


def rotate_90(image: Image.Image) -> Image.Image:
    """Rotate image by ± 90 degrees."""
    angle = random.choice([-90, 90])
    return image.rotate(angle, expand=True)


def get_random_transform() -> Callable[[Image.Image], Image.Image]:
    """Return a random transformation function from the available list.

    Returns:
        A function that takes an Image and returns an Image.

    """
    transforms: List[Callable[[Image.Image], Image.Image]] = [
        add_noise,
        adjust_saturation,
        flip_horizontal,
        flip_vertical,
        rotate_small,
        rotate_90,
    ]
    return random.choice(transforms)


def main() -> None:
    """Execute the augmentation pipeline."""
    parser = argparse.ArgumentParser(description="P2: Random Augmentation Script")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input training folder (e.g., data/deduplicated/train)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output training folder (e.g., data/augmented/train)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Pulizia preliminare
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Troviamo le immagini (supporto per jpg, png, bmp)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [x for x in input_dir.rglob("*") if x.suffix.lower() in valid_exts]

    print(f"[*] Trovate {len(images)} immagini in {input_dir}")
    print("[*] Generazione dataset aumentato (Originale + 1 Variante Casuale)...")

    for img_path in tqdm(images, desc="Augmenting"):
        # Struttura cartelle:
        # data/deduplicated/train/melanoma -> data/augmented/train/melanoma
        rel_path = img_path.relative_to(input_dir)
        target_path = output_dir / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Copia l'originale
        shutil.copy2(img_path, target_path)

        # 2. Genera e salva la variante aumentata
        try:
            with Image.open(img_path) as raw_img:
                # Correzione PLW2901: raw_img non viene sovrascritta
                img = raw_img.convert("RGB")

                # Seleziona UNA trasformazione casuale
                transform_func = get_random_transform()
                augmented_img = transform_func(img)

                # Salva con suffisso _aug
                stem = target_path.stem
                suffix = target_path.suffix
                aug_name = target_path.parent / f"{stem}_aug{suffix}"

                augmented_img.save(aug_name)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"\n✅ Augmentation completata. Dataset salvato in: {output_dir}")


if __name__ == "__main__":
    main()
