import os
import sys
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional
from PIL import Image
from torch import nn
from torchvision import transforms

try:
    from src.training.common import get_model
except ImportError:
    print(
        "ERRORE: Esegui questo script dalla root del progetto "
        "(dove c'è la cartella 'src')"
    )
    sys.exit(1)

# --- CONFIGURAZIONE ---
MODEL_NAME = "effnet_v2_s"

# Calcola la root del progetto automaticamente
BASE_DIR = Path(__file__).resolve().parent.parent

# Costruisci il percorso partendo dalla root
WEIGHTS_PATH = (
    BASE_DIR
    / "pipelines"
    / "p4_aug_two_phase"
    / "results"
    / "phase2"
    / "effnet_v2_s"
    / "model.pth"
)

print(f"[*] Sto cercando il modello qui: {WEIGHTS_PATH}")

CLASSES = [
    "Dermatitis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity",
    "demodicosis",
    "ringworm",
]


# --- CARICAMENTO MODELLO ---
def load_trained_model() -> tuple[nn.Module, torch.device]:
    """Carica il modello e i pesi dal disco."""
    print(f"[*] Caricamento modello {MODEL_NAME}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # architettura
    model = get_model(MODEL_NAME, len(CLASSES))

    # pesi
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Non trovo il file dei pesi in: {WEIGHTS_PATH}")

    # map_location=device garantisce che funzioni anche se hai trainato
    # su GPU e testi su CPU
    state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model, device


model, device = load_trained_model()

# --- PREPROCESSING ---
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict(image: Image.Image) -> dict[str, float] | None:
    """Esegue la predizione sull'immagine fornita."""
    if image is None:
        return None

    # Trasforma l'immagine
    img_tensor = transform(image).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        # Applica Softmax per ottenere percentuali
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]

    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Carica foto della lesione"),
    outputs=gr.Label(num_top_classes=3, label="Diagnosi AI"),
    title="Vet-Skin AI Assistant",
    description=(
        f"Modello caricato: {MODEL_NAME}. "
        "Carica un'immagine per rilevare patologie cutanee."
    ),
)

if __name__ == "__main__":
    print("[*] Avvio interfaccia... Clicca sul link che apparirà sotto.")
    interface.launch(share=True)
