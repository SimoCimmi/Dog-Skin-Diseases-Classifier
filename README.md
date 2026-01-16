# Dog Skin Diseases Classifier 🐶🩺

Progetto di Machine Learning - A.A. 2025-26 (Prof. Polese, Prof.ssa Caruccio).

## 1. Scenario e Obiettivo
Sviluppo di un sistema di classificazione automatica per 6 patologie cutanee canine.
L'obiettivo è supportare la diagnosi veterinaria tramite Deep Learning.
Il modello esamina le foto caricate e identifica se appartengono a una delle classi presenti nel dataset, specificando la confidence.

## 2. Dataset
Il dataset comprende 10.000 immagini (640x640px) suddivise in 6 classi (dermatite, ipersensibilità, infezioni fungine, demodicosi, tigna, cute sana).

## 3. Metodologia (Pipeline)
Il progetto confronta 2 modelli (EfficientNetV2 S/M/L, ConvNeXt) attraverso 4 pipeline sperimentali:
1. **Raw**: partiamo dal raw dataset, rimuoviamo i duplicati e addestriamo i modelli.
2. **Augmented**: applichiamo tecniche di data augmentation sul raw dataset prima di addestrare i modelli..
3. **Balanced**: bilanciamo le classi del raw dataset, addestriamo i modelli, poi facciamo fine-tuning.
4. **Hybrid**: mescoliamo le due tecniche citate sopra, facendo prima data augmentation, poi data balancing e fine tuning.
   
## 4. Qualità del Codice
Per garantire la leggibilità e la manutenibilità del progetto, utilizziamo **Ruff**.
- Per controllare il codice: `ruff check .`
- Per formattare il codice: `ruff format .`
