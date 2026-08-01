# Minecraft Structure Generator

PyTorch-basierte Pipeline zur Generierung von Minecraft-Strukturen aus Text-Prompts.
Unterstützt mehrere Architekturen: Shared-Weight-Transformer, Diffusion-Modelle und
pre-trained Text-Encoder (Phi-3.5, Gemma 2/3, Flan-T5).

## Features

- **Text-zu-Bauwerk**: Generiere Minecraft-Strukturen aus natürlicher Sprache
- **Multi-Architektur**: Wähle zwischen Transformer, Diffusion oder TF-Diffusion (hybride Modelle)
- **GUI-Anwendung**: Benutzerfreundliche Oberfläche mit CustomTkinter
- **3D-Vorschau**: Echtzeit-3D-Ansicht generierter Bauwerke (pyglet/OpenGL)
- **Training**: Integriertes Training mit Augmentierung, Sampling-Gewichten und mehr
- **Kaggle-Export**: Erstelle Kaggle-Notebooks für Cloud-Training
- **Portable Build**: Erstelle eine portable Windows-Distribution mit `build_portable.py`

## Schnellstart

### GUI starten

```powershell
.\start_app.bat
```

Oder direkt:

```powershell
python -m app.main_app
```

### Bauwerk generieren (Kommandozeile)

```powershell
python generate.py "small medieval wooden cottage with stone foundation" --run-dir runs/voxel_transformer --output generated.schem
```

### Training

```powershell
python train.py --data-dir "Trainingsdaten good thoroughly analyzed" --model-size 16 --epochs 10 --lr 1.5e-3
```

## Projektstruktur

```
├── app/                          # GUI-Anwendung (CustomTkinter)
│   ├── main_app.py               # Hauptfenster mit Training, Generierung, Vorschau
│   ├── config.py                 # Konfigurations-Dataclasses
│   ├── model_manager.py          # Modell-Registry und -Verwaltung
│   ├── diffusion_model.py        # Voxel-Diffusionsmodell & TF-Diffusion
│   ├── transformer_encoder.py    # Pre-trained Text-Encoder (Phi-3.5, Gemma 2/3, Flan-T5)
│   ├── hidden_state_cache.py     # Cache für Encoder-Hidden-States
│   ├── voxel_preview.py          # 2D-Voxel-Vorschau (Pillow)
│   └── voxel_viewer_3d.py        # 3D-Viewer (pyglet/OpenGL)
├── dataset.py                    # .schem/.schematic Import, Tokenizer, Augmentierung
├── model.py                      # Shared-Weight-Voxel-Transformer (Single-Pass)
├── train.py                      # Kommandozeilen-Training (Transformer)
├── generate.py                   # Kommandozeilen-Generierung (Transformer)
├── augment_data.py               # Datenaugmentierung (Spiegelung, Rotation)
├── filter_dataset.py             # Dataset-Filterung nach Größe
├── kaggle_export.py              # Export für Kaggle-Notebooks
├── build_portable.py             # Portable-Windows-Builder
├── config.json                   # GUI-Konfiguration
├── requirements.txt              # Python-Abhängigkeiten
├── start_app.bat                 # GUI-Starter
└── tests/
    └── test_pipeline.py          # Pipeline-Tests (pytest)
```

## Architekturen

### Shared-Weight-Transformer (`model.py`)
Kompakter Single-Pass Transformer, der im Forward Pass direkt das gesamte Voxel-Grid
generiert. Verwendet Cross-Attention über den encodierten Prompt.

Training mit gewichteter Cross-Entropy und adaptiven per-structure Gewichten.

### Diffusion Model (`app/diffusion_model.py`)
Denoising-Diffusion-Probabilistic-Model (DDPM) für Voxelgitter.
Verwendet feste Architektur: d_model=128, d_text=64, channels=64, (1,2,2).

### Transformer Diffusion (`app/diffusion_model.py`)
Hybrides Modell mit gefrorenem pre-trained Text-Encoder und 3D UNet
mit Cross-Attention. Unterstützt Phi-3.5, Gemma 2/3 und Flan-T5.

### Pre-trained Text Encoder (`app/transformer_encoder.py`)
Unterstützte Modelle (alle Gewichte frozen):
- **Phi-3.5-mini** (Microsoft)
- **Gemma 2** (2B–27B, Google)
- **Gemma 3** (1B–27B, Google)
- **Flan-T5** (small–XXL, Google)

## GUI-Funktionen

- **Modellauswahl**: Transformer, Diffusion oder TF-Diffusion, verschiedene Checkpoints
- **Grid-Größe**: 16×16×16, 32×32×32, 48×48×48, 64×64×64, 96×96×96, 128×128×128, 256×256×256 (experimentell)
- **Temperatur & Top-K**: Steuerung der Generierungs-Zufälligkeit
- **3D-Vorschau**: Orbit-Kamera (Maus ziehen = rotieren, Scrollen = zoomen)
- **Training**: Direkt aus der GUI mit Augmentierungs-Optionen
- **Export**: Speichern als `.schem` (WorldEdit/Litematica-kompatibel)
- **Kaggle-Export**: Erstelle Kaggle-Notebooks für Cloud-Training

## Training (GUI)

1. Modellarchitektur wählen (Transformer, Diffusion oder TF-Diffusion)
2. Trainingsdaten-Ordner werden automatisch erkannt
3. Grid-Größe, Epochen, Lernrate einstellen
4. Augmentierung aktivieren (Rotation, Verschiebung, vertikale Bewegung)
5. "Training starten" klicken

## Abhängigkeiten

Siehe `requirements.txt`:
- torch>=2.0.0
- nbtlib>=1.12.0
- customtkinter>=5.0.0
- Pillow>=10.0.0
- numpy>=1.24.0

Für den 3D-Viewer zusätzlich: `pip install pyglet`
Für Text-Encoder: `pip install transformers`

## Portable Distribution

```powershell
python build_portable.py
```

Erzeugt eine portable Windows-Distribution mit embedded Python im `dist/`-Ordner.

## Tests

```powershell
python -m pytest tests -q -p no:cacheprovider
```

## Lizenz

MIT