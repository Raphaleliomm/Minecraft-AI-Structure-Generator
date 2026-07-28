# Minecraft Structure Generator

PyTorch-basierte Pipeline zur Generierung von Minecraft-Strukturen aus Text-Prompts.
Unterstützt mehrere Architekturen: Shared-Weight-Transformer, Diffusion-Modelle und
pre-trained Text-Encoder (Phi-3.5, Gemma, Flan-T5).

## Features

- **Text-zu-Bauwerk**: Generiere Minecraft-Strukturen aus natürlicher Sprache
- **Multi-Architektur**: Wähle zwischen Transformer, Diffusion oder hybriden Modellen
- **GUI-Anwendung**: Benutzerfreundliche Oberfläche mit CustomTkinter
- **3D-Vorschau**: Echtzeit-3D-Ansicht generierter Bauwerke (pyglet/OpenGL)
- **Training**: Integriertes Training mit Augmentierung, Sampling-Gewichten und mehr
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
python train.py --data-dir "Trainingsdaten good thoroughly analyzed" --grid-size 16x16x16 --epochs 10 --alpha 0.25
```

## Projektstruktur

```
├── app/                          # GUI-Anwendung (CustomTkinter)
│   ├── main_app.py               # Hauptfenster mit Training, Generierung, Vorschau
│   ├── config.py                 # Konfigurations-Dataclasses
│   ├── model_manager.py          # Modell-Registry und -Verwaltung
│   ├── diffusion_model.py        # Voxel-Diffusionsmodell
│   ├── transformer_encoder.py    # Pre-trained Text-Encoder (Phi-3.5, Gemma, T5)
│   ├── hidden_state_cache.py     # Cache für Encoder-Hidden-States
│   ├── voxel_preview.py          # 2D-Voxel-Vorschau (Pillow)
│   └── voxel_viewer_3d.py        # 3D-Viewer (pyglet/OpenGL)
├── dataset.py                    # .schem/.schematic Import, Tokenizer, Augmentierung
├── model.py                      # Shared-Weight-Voxel-Transformer
├── train.py                      # Kommandozeilen-Training
├── generate.py                   # Kommandozeilen-Generierung
├── gui_app.py                    # Legacy-GUI (einfachere Version)
├── augment_data.py               # Datenaugmentierung (Spiegelung, Rotation)
├── filter_dataset.py             # Dataset-Filterung nach Größe
├── kaggle_export.py              # Export für Kaggle-Notebooks
├── build_portable.py             # Portable-Windows-Builder
├── config.json                   # GUI-Konfiguration
├── requirements.txt              # Python-Abhängigkeiten
├── start_app.bat                 # GUI-Starter (Haupt-App)
├── start_gui.bat                 # GUI-Starter (Legacy)
└── tests/
    └── test_pipeline.py          # Pipeline-Tests (pytest)
```

## Architekturen

### Shared-Weight-Transformer (`model.py`)
Kompakter Transformer, der im Forward Pass zweimal aufgerufen wird:
1. **First Pass**: Erzeugt eine Draft-Struktur `Y1`
2. **Second Pass**: Nutzt Prompt + `Y1` als Kontext für finale Struktur `Y2`

Training mit gewichteter Cross-Entropy: `alpha * CE(Y1, Ygt) + beta * CE(Y2, Ygt)`

### Diffusion Model (`app/diffusion_model.py`)
Denoising-Diffusion-Probabilistic-Model (DDPM) für Voxelgitter.
Unterstützt konditionierte Generierung mit Text-Encodern.

### Pre-trained Text Encoder (`app/transformer_encoder.py`)
Unterstützte Modelle (alle Gewichte frozen):
- **Phi-3.5-mini** (Microsoft)
- **Gemma 2** (1B–27B, Google)
- **Gemma 3** (1B–27B, Google)
- **Flan-T5** (small–XXL, Google)

## GUI-Funktionen

- **Modellauswahl**: Transformer oder Diffusion, verschiedene Checkpoints
- **Grid-Größe**: 16×16×16, 32×32×32 oder 48×48×48
- **Temperatur & Top-K**: Steuerung der Generierungs-Zufälligkeit
- **3D-Vorschau**: Orbit-Kamera (Maus ziehen = rotieren, Scrollen = zoomen)
- **Training**: Direkt aus der GUI mit Augmentierungs-Optionen
- **Export**: Speichern als `.schem` (WorldEdit/Litematica-kompatibel)

## Training (GUI)

1. Modellarchitektur wählen (Transformer oder Diffusion)
2. Trainingsdaten-Ordner auswählen
3. Grid-Größe, Epochen, Lernrate einstellen
4. Augmentierung aktivieren (Spiegelung, Rotation, vertikale Verschiebung)
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