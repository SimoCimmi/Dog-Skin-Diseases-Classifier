import argparse
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path


def get_image_hash(filepath: Path) -> str:
    """Genera un hash MD5 per identificare contenuti identici."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_all_duplicates(root_dir: Path):
    """Scansiona tutte le sottocartelle (train, test, valid) e mappa gli hash.
    """
    hashes = defaultdict(list)
    # Estensioni supportate
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    print(f"🔍 Scansione totale in corso su: {root_dir}...")

    # rglob("*") cerca ricorsivamente in tutte le sottocartelle
    for file_path in root_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            file_hash = get_image_hash(file_path)
            hashes[file_hash].append(file_path)

    return hashes


def run_deduplication(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Report stats
    stats = {
        "same_class_removed": 0,
        "cross_class_removed": 0,
        "total_kept": 0
    }

    all_entries = find_all_duplicates(input_path)

    if not all_entries:
        print("Empty dataset.")
        return

    for h, paths in all_entries.items():
        # Estraiamo le classi (nomi delle cartelle genitrici immediate)
        # Es: data/raw/train/healthy/img1.jpg -> classe è 'healthy'
        classes_involved = {p.parent.name for p in paths}

        if len(classes_involved) > 1:
            # CASO 1: CROSS-CLASS o CROSS-SPLIT DUPLICATES
            # Eliminiamo tutto per evitare confusione o data leakage
            stats["cross_class_removed"] += len(paths)
            print(f"⚠️ Rimosso duplicato inconsistente (Cross-Class): {h}")
            continue

        else:
            # CASO 2: DUPLICATI NELLA STESSA CLASSE
            # Ne teniamo solo uno e copiamo quello nella cartella di output
            keep_path = paths[0]
            stats["same_class_removed"] += (len(paths) - 1)
            stats["total_kept"] += 1

            # Ricostruiamo il percorso relativo per l'output (mantenendo train/test/valid e classe)
            rel_path = keep_path.relative_to(input_path)
            dest_file = output_path / rel_path

            # Crea le cartelle necessarie e copia
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(keep_path, dest_file)

    # Stampa riassunto per il report
    print("\n" + "=" * 30)
    print("STATISTICHE DEDUPLICAZIONE")
    print("=" * 30)
    print(f"Immagini uniche salvate: {stats['total_kept']}")
    print(f"Duplicati stessa classe rimossi: {stats['same_class_removed']}")
    print(f"Inconsistenze cross-class eliminate totalmente: {stats['cross_class_removed']}")
    print(f"Output salvato in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path alla cartella data/raw")
    parser.add_argument("--output", type=str, required=True, help="Path alla cartella data/interim/deduplicated")
    args = parser.parse_args()

    run_deduplication(args.input, args.output)
