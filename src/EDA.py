from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm


class SkinDetectorEDA:
    """Gestisce l'Analisi Esplorativa (EDA) professionale per dataset divisi in split.

    Analizza distribuzione, metadati e geometria distinguendo tra
    Train, Test e Validation.
    """

    def __init__(self, data_path: Path, report_path: Path) -> None:
        """Inizializza l'EDA con i percorsi necessari.

        Args:
            data_path: Path alla cartella 'raw' (contenente train/test/valid).
            report_path: Path dove salvare i grafici e i CSV.

        """
        self.data_path: Final[Path] = data_path
        self.report_path: Final[Path] = report_path
        self.valid_exts: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".bmp")
        self.splits: Final[list[str]] = ["train", "test", "valid"]

        # Crea la cartella reports/eda se non esiste
        self.report_path.mkdir(parents=True, exist_ok=True)

    def run_full_analysis(self) -> None:
        """Esegue l'analisi completa scansionando tutti gli split."""
        print(f"[*] Starting EDA on dataset at: {self.data_path}")

        # 1. Estrazione Dati (Unica scansione per efficienza)
        df_full = self._scan_dataset()

        if df_full.empty:
            print("[-] Error: Dataset is empty or path structure is incorrect.")
            return

        # 2. Export CSV
        csv_path = self.report_path / "dataset_metadata.csv"
        df_full.to_csv(csv_path, index=False)
        print(f"[+] Metadata CSV exported to: {csv_path}")

        # 3. Generazione Grafici
        self._plot_class_distribution(df_full)
        self._plot_split_distribution(df_full)
        self._plot_geometry(df_full)

        # 4. Summary Terminale
        self._print_executive_summary(df_full)

    def _scan_dataset(self) -> pd.DataFrame:
        """Scansiona ricorsivamente train/test/valid e estrae metadati."""
        data = []

        # Itera su ogni split (train, test, valid)
        for split in self.splits:
            split_path = self.data_path / split
            if not split_path.exists():
                print(f"[!] Warning: Split folder '{split}' not found in raw data.")
                continue

            # Itera su ogni classe dentro lo split
            for class_dir in split_path.iterdir():
                if class_dir.is_dir():
                    files = [
                        f
                        for f in class_dir.iterdir()
                        if f.suffix.lower() in self.valid_exts
                    ]

                    for img_path in tqdm(
                        files, desc=f"Scanning {split}/{class_dir.name}"
                    ):
                        try:
                            with Image.open(img_path) as img:
                                w, h = img.size
                                data.append(
                                    {
                                        "Filename": img_path.name,
                                        "Split": split,  # Colonna fondamentale
                                        "Class": class_dir.name,
                                        "Width": w,
                                        "Height": h,
                                        "Aspect_Ratio": round(w / h, 2),
                                        "Pixels": w * h,
                                    }
                                )
                        except (OSError, ValueError):
                            continue

        return pd.DataFrame(data)

    def _plot_class_distribution(self, df: pd.DataFrame) -> None:
        """Confronta la distribuzione delle classi nei vari split."""
        plt.figure(figsize=(14, 8))
        sns.set_theme(style="whitegrid")

        # Conta occorrenze per Split e Classe
        sns.countplot(
            data=df, x="Class", hue="Split", palette="viridis", edgecolor="black"
        )

        plt.title("Class Distribution across Train/Test/Valid", fontsize=16, pad=20)
        plt.xlabel("Disease Class")
        plt.ylabel("Number of Images")
        plt.legend(title="Dataset Split")

        output = self.report_path / "class_distribution_by_split.png"
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_split_distribution(self, df: pd.DataFrame) -> None:
        """Grafico a torta per mostrare la proporzione totale (es. 70/20/10)."""
        split_counts = df["Split"].value_counts()

        plt.figure(figsize=(8, 8))
        plt.pie(
            split_counts,
            labels=split_counts.index,
            autopct="%1.1f%%",
            colors=sns.color_palette("pastel"),
            startangle=140,
        )
        plt.title("Overall Dataset Split Ratio", fontsize=16)

        output = self.report_path / "split_ratio_pie.png"
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_geometry(self, df: pd.DataFrame) -> None:
        """Analizza risoluzione e aspect ratio."""
        sns.set_theme(style="white")

        # Scatter plot Width vs. Height con colore per Split
        grid = sns.jointplot(
            data=df,
            x="Width",
            y="Height",
            hue="Split",  # Vediamo se uno split ha immagini diverse
            kind="scatter",
            alpha=0.6,
            palette="deep",
            height=10,
        )

        # Linee di riferimento
        max_dim = max(df["Width"].max(), df["Height"].max())
        x_ref = np.linspace(0, max_dim, 100)
        grid.ax_joint.plot(x_ref, x_ref, "r--", alpha=0.5, label="1:1 (Square)")
        grid.ax_joint.plot(x_ref, x_ref * 0.75, "g--", alpha=0.5, label="4:3")

        grid.figure.suptitle(
            "Image Geometry Analysis (Resolution Check)", y=1.02, fontsize=16
        )

        output = self.report_path / "geometry_analysis.png"
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()

    def _print_executive_summary(self, df: pd.DataFrame) -> None:
        """Stampa un report scientifico per il terminale."""
        print("\n" + "=" * 60)
        print("                EXECUTIVE EDA SUMMARY REPORT")
        print("=" * 60)

        total_imgs = len(df)
        print(f"Total Samples:      {total_imgs}")

        # Statistiche per Split
        print("\n--- Split Breakdown ---")
        for split in self.splits:
            count = len(df[df["Split"] == split])
            pct = (count / total_imgs) * 100
            print(f"{split.capitalize()}: {count:5d} images ({pct:.1f}%)")

        # Controllo Bilanciamento (Imbalance Ratio nel Train)
        df_train = df[df["Split"] == "train"]
        if not df_train.empty:
            class_counts = df_train["Class"].value_counts()
            ir = class_counts.max() / class_counts.min()
            print(f"\nTraining Imbalance Ratio: {ir:.2f} (1.00 is perfectly balanced)")

        # Controllo Risoluzione
        avg_w = df["Width"].mean()
        avg_h = df["Height"].mean()
        print(f"\nAverage Resolution: {int(avg_w)}x{int(avg_h)} px")

        print("-" * 60)
        print(f"[✓] Analysis complete. Check graphs in: {self.report_path}")
        print("=" * 60)


if __name__ == "__main__":
    # Setup dinamico dei path
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # Sale da src/ a root

    raw_data_path = project_root / "data" / "raw"
    reports_path = project_root / "reports" / "eda"  # Creiamo una sottocartella eda

    if not raw_data_path.exists():
        print(f"[-] Error: Could not find data at {raw_data_path}")
        print("    Ensure you are running this from the project root or src folder.")
    else:
        eda = SkinDetectorEDA(raw_data_path, reports_path)
        eda.run_full_analysis()
