import argparse
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path


def get_image_hash(filepath: Path) -> str:
    """Genera un hash MD5 per identificare contenuti identici leggendo a chunk."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_all_duplicates(root_dir: Path) -> dict[str, list[Path]]:
    """Scansiona tutte le sottocartelle e mappa gli hash."""
    hashes = defaultdict(list)
    # Estensioni comuni
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    print(f"🔍 Scansione totale in corso su: {root_dir}...")

    # rglob trova i file indipendentemente dalla profondità
    for file_path in root_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            file_hash = get_image_hash(file_path)
            hashes[file_hash].append(file_path)

    return hashes


def run_deduplication(input_dir: str, output_dir: str) -> None:
    """Esegue la deduplicazione applicando logiche severe per evitare data leakage."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Statistiche per il report
    stats = {
        "total_images_scanned": 0,
        "unique_images_kept": 0,
        "removed_same_split": 0,  # Duplicati ridondanti (es. copia in train)
        "removed_cross_class": 0,  # Ambiguità (es. img1 in Healthy e in Ringworm)
        "removed_cross_split": 0,  # DATA LEAKAGE (es. img1 in train e in test)
    }

    all_entries = find_all_duplicates(input_path)

    if not all_entries:
        print("❌ Dataset vuoto o percorso errato.")
        return

    stats["total_images_scanned"] = sum(len(paths) for paths in all_entries.values())

    print(f"Immagini trovate: {stats['total_images_scanned']}. Inizio elaborazione...")

    for h, paths in all_entries.items():
        classes_involved = {p.parent.name for p in paths}
        splits_involved = {p.parent.parent.name for p in paths}

        # CASO 1: Inconsistenza di Classe
        if len(classes_involved) > 1:
            stats["removed_cross_class"] += len(paths)
            continue

        # CASO 2: Inconsistenza di Split (Data Leakage)
        if len(splits_involved) > 1:
            stats["removed_cross_split"] += len(paths)
            print(
                f"⚠️ LEAKAGE: Hash {h} in {splits_involved}. Rimossi tutti."
            )
            continue

        # CASO 3: Duplicati "sicuri"
        keep_path = paths[0]
        duplicates_count = len(paths) - 1
        stats["removed_same_split"] += duplicates_count
        stats["unique_images_kept"] += 1

        rel_path = keep_path.relative_to(input_path)
        dest_file = output_path / rel_path

        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(keep_path, dest_file)

    # --- REPORT FINALE PER IL PROGETTO ---
    print("\n" + "=" * 40)
    print("📊 REPORT DEDUPLICAZIONE (Pipeline 1)")
    print("=" * 40)
    print(f"Totale immagini analizzate:      {stats['total_images_scanned']}")
    print(f"Immagini valide mantenute:       {stats['unique_images_kept']}")
    print("-" * 40)
    print(
        f"🗑️  Duplicati interni rimossi:    {stats['removed_same_split']} "
        "(Ridondanza)"
    )
    print(
        f"⛔ Conflict Cross-Class rimossi:  {stats['removed_cross_class']} "
        "(Etichetta ambigua)"
    )
    print(
        f"☢️  Conflict Cross-Split rimossi:  {stats['removed_cross_split']} "
        "(DATA LEAKAGE evitato)"
    )
    print("=" * 40)
    print(f"✅ Dataset pulito salvato in: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw",
        help="Input dataset root"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/deduplicated",
        help="Output directory",
    )
    args = parser.parse_args()

    run_deduplication(args.input, args.output)
