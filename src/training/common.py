"""Utility comuni per caricamento dati e inizializzazione modelli."""

import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Rilevamento automatico del device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_loader(
        data_dir: Path, batch_size: int, shuffle: bool = False, is_train: bool = False
) -> DataLoader:
    """Crea un DataLoader per una specifica directory.

    Args:
        data_dir: Path alla cartella contenente le classi (es. data/train).
        batch_size: Dimensione del batch.
        shuffle: Se mescolare i dati.
        is_train: Se True, applica augmentation leggere al volo.

    Returns:
        DataLoader configurato.

    """
    # Risoluzione standard ImageNet (224).
    resize_dim = 224

    stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if is_train:
        # Augmentation base "on-the-fly" per evitare overfitting durante il train
        tfs = transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ])
    else:
        # Nessuna augmentation per validazione/test
        tfs = transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ])

    # Ottimizzazioni per GPU NVIDIA
    # num_workers=4 è un buon compromesso per non saturare la CPU
    dataset = datasets.ImageFolder(str(data_dir), transform=tfs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4 if os.name != 'nt' else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )


def get_model(model_name: str, num_classes: int) -> nn.Module:
    """Inizializza l'architettura scelta con pesi pre-addestrati.

    Args:
        model_name: Nome del modello (effnet_v2_s, convnext).
        num_classes: Numero di classi in output.

    """
    if "effnet" in model_name:
        # EfficientNet V2 Small (Ottimo bilanciamento speed/acc)
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        model = models.efficientnet_v2_s(weights=weights)
        # Sostituzione della testa di classificazione
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif "convnext" in model_name:
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    else:
        raise ValueError(f"Modello {model_name} non supportato.")

    return model.to(DEVICE)
