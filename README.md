# Minecraft AI Structure Generator

PyTorch-basierte Pipeline zur Generierung von Minecraft-Strukturen aus Text-Prompts.
Unterstützt mehrere Architekturen: Shared-Weight-Transformer, Diffusion-Modelle und
pre-trained Text-Encoder (Phi-3.5, Gemma 2/3, Flan-T5).

## Features

- **Text-zu-Bauwerk**: Generiere Minecraft-Strukturen aus natürlicher Sprache
- **Multi-Architektur**: Wähle zwischen Transformer, Diffusion oder TF-Diffusion (Empfohlen: TF-Diffusion)
- **GUI-Anwendung**: Benutzerfreundliche Oberfläche mit CustomTkinter
- **3D-Vorschau**: Echtzeit-3D-Ansicht generierter Bauwerke (pyglet/OpenGL)
- **Training**: Integriertes Training mit Augmentierung, Sampling-Gewichten und mehr
- **Kaggle-Export**: Erstelle Kaggle-Notebooks für Cloud-Training

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
├── filter_dataset.py             # Dataset-Filterung nach Größe
├── scrape_phantommarket.py       # Dataset-Scraper (market.phantom-node.com)
├── scrape_all.bat                # Starter für den Volllauf (resume-fähig)
├── validate_schematics.py        # Validierung von .schem-Dateien (ohne nbtlib)
├── kaggle_export.py              # Export für Kaggle-Notebooks
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
- **Grid-Größe**: 16×16×16, 32×32×32, 48×48×48, 64×64×64, 96×96×96, 128×128×128, 256×256×256 (96×96×96+ ist experimentell)
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

## Trainingsdaten von PhantomMarket laden (`scrape_phantommarket.py`)

Lädt alle Builds von [market.phantom-node.com](https://market.phantom-node.com) als
Trainingsdaten. Die **Build-Beschreibung wird dabei als Prompt** gespeichert.

```powershell
# Alle Builds als .schem + Prompt-Dateien (Standard-Ziel: ../Trainingsdaten/phantommarket)
python scrape_phantommarket.py

# oder als Doppelklick-Starter mit den Standardwerten für den Volllauf
.\scrape_all.bat

# Erst testen: 50 Builds, langsamer
python scrape_phantommarket.py --max-builds 50 --delay 1.5

# Original-Format zusätzlich zum konvertierten .schem
python scrape_phantommarket.py --formats litematic,schem --max-file-mb 100

# Nur Beschreibungen/Metadaten (schnell, keine Downloads)
python scrape_phantommarket.py --metadata-only

# prompts.json nachträglich aus builds_meta.jsonl neu ableiten
python scrape_phantommarket.py --rebuild-prompts
```

Ausgabe-Layout (direkt als `--data-dir` nutzbar):

```
Trainingsdaten/phantommarket/
├── builds_meta.jsonl              # alle Metadaten je Build (Resume-Grundlage)
├── prompts.json                   # {"<datei>.schem": "<prompt>"}
├── <slug>__Name.schem             # konvertierte Struktur (Sponge v2/v3)
├── <slug>__Name.litematic         # optionales Original
└── <slug>__Name.txt               # Prompt-Sidecar (name:/author:/source:/url: + ---)
```

Wichtige Optionen:

| Option | Wirkung |
| --- | --- |
| `--out-dir` | Zielordner (Standard `../Trainingsdaten/phantommarket`) |
| `--formats` | `litematic,schem,nbt,mcstructure,mcpack,java_world` (Standard: `schem`) |
| `--metadata-only` | keine Struktur-Dateien, nur Beschreibungen/Metadaten |
| `--max-builds` / `--order` | Umfang und Reihenfolge (`newest`/`oldest`/`shuffle`) |
| `--limit` | Lauf nach N verarbeiteten Builds beenden (Resume-freundlich, für lange Vollläufe) |
| `--delay` / `--jitter` / `--retries` | Rate-Limit und Wiederholungen (429/5xx) |
| `--max-file-mb` | große Builds überspringen (Standard 250 MB) |
| `--prompt-mode` | `description` (Standard), `title` oder `title+description` |
| `--max-prompt-chars` | Prompt kappen (Standard 0 = Beschreibung unverändert) |
| `--no-meta-lines` | `.txt` ohne Herkunftszeilen (Altformat) |
| `--dedupe-by-hash` / `--redownload` / `--refresh-meta` | Duplikate bzw. Neuladen erzwingen |
| `--split-by-category` | Strukturen in Unterordner je Kategorie legen |

Der Download ist **unterbrechbar und wiederaufsetzbar**: Ein abgebrochener Lauf wird
mit demselben Befehl fortgesetzt, bereits vollständige Builds werden übersprungen
(`builds_meta.jsonl`, Status `ok`). Fehlgeschlagene Builds bleiben mit `ok: false`
in der Datei und werden beim nächsten Lauf erneut versucht.

> **Rechtlicher Hinweis:** Die Seite untersagt in ihrer `robots.txt`
> (`Content-Signal: ai-train=no`) die Nutzung der Inhalte als KI-Trainingsdaten,
> und der Download-Endpoint liegt unter `Disallow: /api/`. Das Skript ist ein
> lokales Werkzeug – die Verantwortung für Nutzung und Weitergabe der Daten
> liegt beim Ausführenden.

## Abhängigkeiten

Siehe `requirements.txt`:
- torch>=2.0.0
- nbtlib>=1.12.0
- customtkinter>=5.0.0
- Pillow>=10.0.0
- numpy>=1.24.0
- transformers>=4.30.0
- pyglet>=2.0.0
- requests>=2.28.0

Installation aller Abhängigkeiten:
```powershell
pip install -r requirements.txt
```

## Tests

```powershell
python -m pytest tests -q -p no:cacheprovider
```
