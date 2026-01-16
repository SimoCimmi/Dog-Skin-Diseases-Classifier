import os
import platform
import shutil
import sys
from pathlib import Path

# Fix per Windows: Cache breve
os.environ["KAGGLEHUB_CACHE"] = "C:\\k_data"

import kagglehub

# --- CALCOLO PERCORSI ASSOLUTI ---
# Ottiene la cartella dove si trova QUESTO file (src/)
SCRIPT_DIR = Path(__file__).resolve().parent
# Ottiene la cartella padre (root del progetto: Dog-Skin-Diseases-Classifier/)
PROJECT_ROOT = SCRIPT_DIR.parent
# Definisce il target finale in modo assoluto
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / "raw"


def get_long_path(path_str: str) -> str:
    """Gestisce i percorsi lunghi (Long Paths) su Windows."""
    path = Path(path_str).resolve()
    path_str = str(path)
    if platform.system() == "Windows":
        if not path_str.startswith("\\\\?\\") and not path_str.startswith("\\\\"):
            return f"\\\\?\\{path_str}"
    return path_str


def download_dataset() -> None:
    """Scarica il dataset da Kaggle e lo organizza nella cartella raw del progetto."""
    dest_path = DEFAULT_DESTINATION

    # Verifica se esiste già
    if dest_path.exists() and any(dest_path.iterdir()):
        print(f"⚠️  La cartella '{dest_path}' non è vuota. Download annullato.")
        return

    print(f"📍 Destinazione fissata a: {dest_path}")
    print("⬇️  Inizio download dataset da Kaggle...")

    try:
        # Download in cache temporanea
        cache_path_str = kagglehub.dataset_download(
            "youssefmohmmed/dogs-skin-diseases-image-dataset"
        )

        # Creazione cartella (assicurati che i genitori esistano)
        dest_path.mkdir(parents=True, exist_ok=True)
        print("📦 Spostamento file...")

        cache_path = Path(cache_path_str)

        # Copia file per file
        for item in cache_path.rglob("*"):
            if item.is_file():
                relative_path = item.relative_to(cache_path)
                target_file = dest_path / relative_path
                target_file.parent.mkdir(parents=True, exist_ok=True)

                safe_src = get_long_path(str(item))
                safe_dst = get_long_path(str(target_file))

                shutil.copy2(safe_src, safe_dst)

        print(f"✅ Dataset scaricato correttamente in:\n   {dest_path}")

        # Pulizia cache
        try:
            shutil.rmtree(r"C:\k_data", ignore_errors=True)
        except Exception:
            pass

    except Exception as e:
        print(f"❌ Errore: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_dataset()
