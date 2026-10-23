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
python train.py --data-dir "..\Trainingsdaten\Trainingsdaten good thoroughly analyzed" --model-size 16 --epochs 10 --lr 1.5e-3
```

Ohne `--data-dir` werden automatisch alle passenden Ordner unter `..\Trainingsdaten\`
verwendet (rekursiv, also auch verschachtelte Datensätze).

## Projektstruktur

```
├── app/                          # GUI-Anwendung (CustomTkinter)
│   ├── main_app.py               # Hauptfenster mit Training, Generierung, Vorschau
│   ├── config.py                 # Konfigurations-Dataclasses (validiert config.json)
│   ├── paths.py                  # Zentrale Pfade (Projektwurzel, runs/, exports/, Datenordner)
│   ├── constants.py              # Grid-Größen (gemeinsam für GUI/CLI)
│   ├── model_manager.py          # Modell-Registry und -Verwaltung (Metadaten-Cache)
│   ├── diffusion_model.py        # Voxel-Diffusionsmodell & TF-Diffusion
│   ├── transformer_encoder.py    # Pre-trained Text-Encoder (Phi-3.5, Gemma 2/3, Flan-T5)
│   ├── hidden_state_cache.py     # Cache für Encoder-Hidden-States
│   ├── training_memory.py        # VRAM-Abschätzung, Chunk-/Batch-Planung (Training & Generierung)
│   ├── voxel_preview.py          # 2D-Voxel-Vorschau (Pillow)
│   ├── voxel_viewer_3d.py        # 3D-Viewer (pyglet/OpenGL, eigener Prozess)
│   └── litematic.py              # .litematic/Blueprint -> Sponge-.schem
├── dataset.py                    # .schem/.schematic Import, Tokenizer, Augmentierung
├── model.py                      # Shared-Weight-Voxel-Transformer (Single-Pass)
├── train.py                      # Kommandozeilen-Training (Transformer)
├── generate.py                   # Kommandozeilen-Generierung (Transformer)
├── filter_dataset.py             # Dataset-Filterung nach Größe
├── scrape_phantommarket.py       # Dataset-Scraper (market.phantom-node.com)
├── scrape_structures.py          # Mehrquellen-Scraper (Abfielder, Hugging Face)
├── scrape_all.bat                # Starter für den Volllauf (resume-fähig)
├── validate_schematics.py        # Validierung von .schem-Dateien (ohne nbtlib)
├── kaggle_export.py              # Export für Kaggle-Notebooks
├── config.json                   # GUI-Konfiguration
├── requirements.txt              # Python-Abhängigkeiten
├── start_app.bat                 # GUI-Starter
└── tests/
    ├── conftest.py               # Sorgt dafür, dass die Tests aus jedem Ordner laufen
    ├── test_pipeline.py          # Pipeline-Tests (pytest)
    ├── test_rotation_direction.py# Rotationsrichtung von Blockstates
    ├── test_dataset_startup.py   # Startzeit/Speicherbedarf der Datensatz-Suche
    ├── test_preview_render.py    # 2D-Vorschau (Rendering, Threading)
    ├── test_prompt_sidecars.py   # Prompt-Dateien (.txt/prompts.json) des Scrapers
    ├── test_structure_conversion.py # Litematic/Blueprint -> .schem + Quellen-Parser
    ├── test_training_memory.py   # VRAM-Planung & gechunkte Logits (Training)
    ├── test_generation_memory.py # VRAM-Planung & gechunkte Logits (Generierung)
    ├── test_encoder_fallback.py  # Text-Encoder: VRAM oder System-RAM (Rückfrage)
    └── test_regressions.py       # Regressionstests für behobene Fehler
```

> **Pfade:** Alle Laufzeit-Pfade (`config.json`, `runs/`, `exports/`, `Blocktextures/`,
> Trainingsdaten) werden über `app/paths.py` relativ zum Projektordner aufgelöst.
> Das Programm darf daher aus jedem Arbeitsverzeichnis gestartet werden.
> Die Strukturen der beigelegten Minecraft-Texturen (`Blocktextures/`) sind
> urheberrechtlich Mojang/Microsoft zuzuordnen und nur für die lokale Vorschau gedacht.

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
- **Batch-Groesse halten**: Passt der Batch nicht in den VRAM, wird er nicht mehr
  still verkleinert, sondern in Mikro-Batches geteilt und die Gradienten
  aufsummiert (Gradient Accumulation, Schalter im Training-Tab, aus per
  `MINECRAFT_AI_GRAD_ACCUM=0` bzw. `--no-grad-accum`): gleiche effektive
  Batch-Groesse, nur etwas langsamer
- **Weiter trainieren**: Im Model-Tab setzt "Weiter trainieren" einen vorhandenen
  Checkpoint fort - Architektur, Vokabular und Text-Encoder kommen aus dem
  Checkpoint, die neuen Epochen werden auf die bereits trainierten addiert
- **Export**: Speichern als `.schem` (WorldEdit/Litematica-kompatibel)
- **Kaggle-Export**: Erstelle Kaggle-Notebooks für Cloud-Training

## Wo liegen die Trainingsdaten?

Ablageort (relativ zum Programmordner, also eine Ebene über `app/`):

```
Minecraft_AI_Structure_Generator/
├── MInecraft build generator Program/    <- Programm + Start (start_app.bat)
└── Trainingsdaten/                       <- hier gehoeren die Trainingsdaten hin
    ├── Meine Sammlung/                   <- beliebiger Name / beliebige Verschachtelung
    │   ├── haus.schem
    │   └── haus.txt
    └── phantommarket/                    <- vom Scraper erzeugt (schem + txt + prompts.json)
```

Regeln:

* Jeder Unterordner von `Trainingsdaten/` wird **rekursiv** durchsucht – Kategorien,
  Unterordner und verschachtelte Sammlungen sind kein Problem.
* Zu jeder Struktur (`*.schem` / `*.schematic`) muss es einen Prompt geben:
  * eine gleichnamige `.txt`-Datei daneben (`haus.schem` → `haus.txt`), oder
  * einen Eintrag in einer `prompts.json` (im selben Ordner oder in einem übergeordneten).
* In der `.txt` steht **ein Prompt pro Block**; mehrere Prompts werden mit `---` getrennt.
  Jeder Block wird ein eigenes Trainingsbeispiel, z. B. Name der Struktur und
  Beschreibung:

  ```
  Medieval House
  ---
  A cozy medieval house for survival
  ```

  Zeilen, die mit `name:`, `author:`, `source:`, `url:`, `aug:` oder `id:` beginnen,
  gelten als Metadaten und werden ignoriert (Altformat des Scrapers).
* In einer `prompts.json` darf pro Struktur ein einzelner Prompt **oder eine Liste**
  von Prompts stehen – die Liste erzeugt ebenfalls ein Beispiel je Prompt:

  ```json
  {
    "haus.schem": ["Medieval House", "A cozy medieval house for survival"],
    "turm.schem": "a tall tower"
  }
  ```
* Ordner ohne Prompts werden **übersprungen** (z. B. `in progress`, `Trainingsdaten_ohne_prompt`).
  Strukturen über `--max-voxels` (Standard 400.000) werden ebenfalls übersprungen –
  die Größe wird dabei **nur aus dem NBT-Header** gelesen, die Blöcke werden also
  gar nicht erst dekodiert (siehe „Startzeit und Speicherbedarf“).
* Mit `--data-dir` (mehrfach angebbar) kann man die Automatik übersteuern:
  `python train.py --data-dir "..\Trainingsdaten\Meine Sammlung"`.

Die Automatik ist in `app/paths.py::default_data_dirs()` implementiert; bevorzugt werden
zuerst `Trainingsdaten good thoroughly analyzed` und `phantommarket`, danach alle weiteren
Ordner mit Prompts (alphabetisch).

## Startzeit und Speicherbedarf

Beim Start des Trainings wird jede Struktur genau **einmal** geladen und ausgewertet;
danach entscheidet ein Budget, was im RAM bleibt:

* **Größenprüfung vor dem Dekodieren.** Ein Build mit z. B. 1019x276x1002 Voxeln
  (281 Mio.) belegt beim Dekodieren Minuten und mehrere GB – obwohl er danach wegen
  `--max-voxels` verworfen wird. `dataset.read_schematic_size()` liest nur den Header;
  zu große Strukturen enden in `SchematicTooLarge`, ohne dass Blockdaten entstehen.
* **Blockzustände & Voxel-Kodierung laufen vektorisiert.** Varint-Dekodierung,
  Bounding-Box (Trimmen) und Vokabular-Aufbau arbeiten mit numpy bzw. memoisierten
  Namens-Abbildungen statt mit Python-Schleifen über jeden Voxel.
* **Begrenzter Schematic-Cache.** Dekodierte Strukturen werden nur bis zu einem
  Budget gehalten (Standard 6 Mio. Voxel ≈ 150-200 MB); anpassbar über die
  Umgebungsvariable `MINECRAFT_AI_SCHEMATIC_CACHE_VOXELS` oder den Parameter
  `schematic_cache_voxels` von `MultiSourceSchematicDataset`.
* **Hidden-State-Cache wird gemappt statt kopiert.** Der vorberechnete Cache ist
  mehrere GB groß (1,7 GB bei 6367 Prompts und `hidden_dim=3072`); er wird per
  `torch.load(..., mmap=True)` eingebunden und nicht mehr vollständig in den RAM
  kopiert. Abschaltbar mit `MINECRAFT_AI_CACHE_MMAP=0`.
* **Fortschritt in der Statuszeile.** Der Aufbau meldet „Lade Trainingsdaten: n/m
  Beispiele…“, damit der Start nicht wie ein Hänger aussieht.

## Training auf kleinen Grafikkarten (VRAM-Planung)

Vor dem ersten Trainingsschritt rechnet `app/training_memory.py` aus, ob Logits,
Aktivierungen und Modellzustände in den **freien** Speicher passen (85 % davon,
`DEFAULT_SAFETY_FRACTION`), und protokolliert das Ergebnis:

```
VRAM-Plan: Logits 34.55 GiB (64 Chunks a 512 Voxel), Batch 16, geschaetzt 4.90 von 5.10 GiB nutzbar
```

Der Auslöser war ein realer Absturz auf einer GTX 1060 (6 GB) mit
`Cuda out of memory. Tried to allocate 34.55 GiB.`: Die Logits des
Diffusionsmodells sind `[batch, num_blocks, X*Y*Z]`, bei Grid 32³, 17.690
Blockzuständen und Batch 16 also `16 * 17690 * 32768 * 4 Byte = 34,55 GiB`.
Seit `app/diffusion_model.py` die Logits **in Voxel-Chunks** berechnet (der
Ausgabekopf ist punktweise und daher beliebig aufteilbar), liegt nur noch ein
Chunk gleichzeitig im Speicher – gemessen 3,76 GiB statt 34,55 GiB, bei
identischem Loss und identischen Gradienten (`tests/test_training_memory.py`
vergleicht beide Rechenwege numerisch).

Was der Plan tut:

* **Chunk-Größe wählen.** 768 MiB Logits pro Chunk treffen den Sweet Spot
  (gemessen auf der GTX 1060, Grid 32³/Batch 16: 512 Voxel → 5,7 s pro Batch,
  1024 → 16,7 s, 2048 → 44,6 s). Der Plan kennt die Restbudget-Rechnung und
  wählt die passende Zweierpotenz.
* **Batch-Größe reduzieren.** Passt der Wunsch-Batch nicht, wird der größte
  passende genommen und der Schritt im Protokoll gemeldet statt abzustürzen.
* **Klare Meldung bei „passt nie“.** Grid 256³ passt auch mit Batch 1 nicht;
  der Plan meldet dann den größten Verursacher (Aktivierungen oder Parameter)
  samt Vorschlag für ein kleineres Grid.
* **Attention-Limit beim Transformer.** Beim Shared-Weight-Transformer
  (`model.py`) läuft die Self-Attention über **alle** Voxel und ist nicht
  chunkbar; bei 32³ sind das 6 · 32768² · 4 Byte = 24 GiB pro Matrix und Layer.
  Statt eines CUDA-Fehlers kommt dort ein Hinweis mit dem größten noch
  passenden Grid.

Abschaltbar bzw. übersteuerbar per Umgebungsvariable:

| Variable | Wirkung |
| --- | --- |
| `MINECRAFT_AI_VRAM_GUARD=0` | Planung aus – Training läuft wie vorher (ungechunkt, ohne Batch-Reduktion) |
| `MINECRAFT_AI_LOGITS_CHUNK_VOXELS=<n>` | feste Chunk-Größe erzwingen (Debugging/Vergleichsmessungen) |

## Generierung auf kleinen Grafikkarten (VRAM-Planung)

Dieselbe Rechnung wie beim Training gilt für den Tab **Generate**: die
Ausgabeköpfe (`out_proj` des UNet bzw. `model.output` des Transformers) sind
punktweise und werden deshalb ebenfalls **in Voxel-Chunks** gerechnet. Der
Planer (`app/training_memory.plan_generation`) legt die Chunk-Größe vor dem
Start aus dem freien Speicher fest und schreibt sie in die Statuszeile:

```
VRAM-Plan (Generierung): Logits 2.16 GiB (16 Chunks a 2048 Voxel), geschaetzt 0.67 von 0.71 GiB nutzbar
```

Der Auslöser war der Fehler

```
Cuda out of memory. Tried to allocate 2.16 GiB. GPU 0 has a total capacity of 6.00 GiB
```

beim Klick auf „Generieren“ **nach** einem Trainingslauf: bei Grid 32³ und
17.690 Blockzuständen ist *ein* Logits-Tensor 2,16 GiB groß, und wegen
Temperatur-, Top-K-, Softmax- und Permute-Kopie liegen kurzzeitig rund 8,6 GiB
im Speicher (gemessen: 8,71 GiB Spitzenwert auf einer GTX 1060 – also mehr als
der gesamte VRAM). Zusätzlich hielt der PyTorch-Caching-Allocator den
Speicher-Peak des Trainings fest, und `torch.cuda.mem_get_info()` (Grundlage der
Planung) meldet solche Blöcke als *belegt*.

Was deshalb passiert:

* **Chunk-Größe aus dem freien Speicher.** Wie beim Training (85 % der freien
  Bytes, `DEFAULT_SAFETY_FRACTION`), gedeckelt auf 768 MiB Logits pro Chunk.
  Gemessen mit dem echten Checkpoint (Grid 32³, 17.690 Blöcke): Spitze 0,67 GiB
  statt OOM.
* **Speicher wird vor jeder Aktion freigegeben.** Nach dem Training, nach einem
  fehlgeschlagenen Modellaufbau und vor der Messung des freien Speichers läuft
  `torch.cuda.empty_cache()` (`release_cuda_cache()` in `app/main_app.py`).
* **Klare Meldung statt Absturz.** Passt selbst der kleinste Chunk nicht
  (z. B. Grid 64³ mit der quadratischen Self-Attention des Transformers), nennt
  die Statuszeile den größten Verursacher und ein Grid, das passen würde.
* **Das Sampling bleibt unverändert.** Chunking erzeugt bitgleiche Logits
  (getestet mit deterministischem Sampling über `top_k=1`/Argmax,
  `tests/test_generation_memory.py`); nur die konkreten Zufallszahlen liegen an
  einer anderen Stelle des Zufallsgenerators.

Für die CLI:

```powershell
python generate.py "small medieval cottage" --model-size 32            # plant automatisch
python generate.py "small medieval cottage" --model-size 32 --chunk-voxels 512   # feste Chunk-Größe
```

> **Text-Encoder brauchen eigenen VRAM – oder den System-RAM.** TF-Diffusion lädt
> zum Generieren den Encoder (z. B. `Phi-3.5-mini`) mit: in fp16 sind das ~7,1 GiB.
> Passt er nicht in den freien VRAM, fragt die App einmal je Encoder:
>
> ```
> Text-Encoder passt nicht in den VRAM
> Phi-3.5-mini braucht ca. 7.1 GiB VRAM, frei sind nur 4.19 GiB.
>
> Den Encoder stattdessen in den System-RAM laden?
> Er rechnet dann auf der CPU: die Generierung funktioniert, dauert aber laenger.
> Freier RAM: 7.4 GiB
> ```
>
> Bei „Ja" wird der Encoder in fp16 in den RAM geladen und läuft auf der CPU
> (gemessen: 45 s Ladezeit, 7-12 s pro Prompt; die Generierung selbst bleibt auf
> der GPU). Reicht auch der RAM nicht, kommt ein Hinweis statt eines Absturzes.
> Die Encoder-Zeile im Training-Tab zeigt dann `..., CPU, frozen`. Für
> Dauerbetrieb auf kleinen Karten lohnt trotzdem ein kleinerer Encoder
> (Flan-T5-small/base, Gemma-3-1B), der komplett in den VRAM passt.
> Der Hidden-State-Cache beschleunigt nur das Training – beim Generieren wird
> der Encoder für den fertigen Prompt live gebraucht.

## Training (GUI)

1. Modellarchitektur wählen (Transformer, Diffusion oder TF-Diffusion)
2. Trainingsdaten-Ordner werden automatisch erkannt (siehe unten), rekursiv
3. Grid-Größe, Epochen, Lernrate einstellen
4. Augmentierung aktivieren (Rotation, Verschiebung, vertikale Bewegung)
5. "Training starten" klicken

> Die Sampling-Schritte werden automatisch auf die Anzahl der Timesteps begrenzt,
> mit denen das Modell trainiert wurde (der Slider erlaubt bewusst mehr).

## Trainingsdaten von PhantomMarket laden (`scrape_phantommarket.py`)

Lädt alle Builds von [market.phantom-node.com](https://market.phantom-node.com) als
Trainingsdaten. Pro Build entstehen **zwei Prompts**: der Build-Name und die
Build-Beschreibung – jeweils in der `.txt` und in `prompts.json` (also zwei
Trainingsbeispiele pro Struktur).

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

# Vorhandene .txt-Sidecars offline auf das aktuelle Format umschreiben
python scrape_phantommarket.py --rebuild-sidecars

# prompts.json nachträglich aus builds_meta.jsonl neu ableiten
python scrape_phantommarket.py --rebuild-prompts
```

Ausgabe-Layout (direkt als `--data-dir` nutzbar):

```
Trainingsdaten/phantommarket/
├── builds_meta.jsonl              # alle Metadaten je Build (Resume-Grundlage, mit Autor/URL)
├── prompts.json                   # {"<datei>.schem": ["<name>", "<beschreibung>"]}
├── <slug>__Name.schem             # konvertierte Struktur (Sponge v2/v3)
├── <slug>__Name.litematic         # optionales Original
└── <slug>__Name.txt               # Prompt-Sidecar (Name + --- + Beschreibung)
```

> Autor, Quelle und URL des Builds stehen weiterhin in `builds_meta.jsonl`;
> `--rebuild-sidecars` ergänzt sie nicht mehr in die `.txt` (dort stehen nur die
> Prompts, sonst würde die Herkunftszeile als eigener Prompt gelten).

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
| `--name-prompt` / `--no-name-prompt` | Name des Builds als ersten Prompt-Block schreiben (Standard: an) |
| `--rebuild-sidecars` | vorhandene `.txt` offline auf `Name + --- + Beschreibung` umschreiben |
| `--rebuild-prompts` | `prompts.json` neu aus `builds_meta.jsonl` ableiten |
| `--dedupe-by-hash` / `--redownload` / `--refresh-meta` | Duplikate bzw. Neuladen erzwingen |
| `--split-by-category` | Strukturen in Unterordner je Kategorie legen (Dataset-Suche ist rekursiv) |

Der Download ist **unterbrechbar und wiederaufsetzbar**: Ein abgebrochener Lauf wird
mit demselben Befehl fortgesetzt, bereits vollständige Builds werden übersprungen
(`builds_meta.jsonl`, Status `ok`). Fehlgeschlagene Builds bleiben mit `ok: false`
in der Datei und werden beim nächsten Lauf erneut versucht.

> **Rechtlicher Hinweis:** Die Seite untersagt in ihrer `robots.txt`
> (`Content-Signal: ai-train=no`) die Nutzung der Inhalte als KI-Trainingsdaten,
> und der Download-Endpoint liegt unter `Disallow: /api/`. Das Skript ist ein
> lokales Werkzeug – die Verantwortung für Nutzung und Weitergabe der Daten
> liegt beim Ausführenden.

## Weitere Quellen laden (`scrape_structures.py`)

`scrape_structures.py` ergänzt PhantomMarket um weitere **frei zugängliche**
Quellen. Die großen Portale sind entweder login-pflichtig, kostenpflichtig oder
sperren KI-Crawler in der `robots.txt`; der Scraper respektiert `robots.txt`
standardmäßig und lädt nur frei erreichbare Dateien.

| Quelle | Inhalt | Umfang | Hinweis |
| --- | --- | --- | --- |
| `abfielder` | abfielder.com, `.litematic` | mehrere Tausend freie Builds (Premium wird übersprungen) | kein Login; Creator-Rechte beachten |
| `huggingface` | `Hack337/Minecraft-Schematics` | 873 Rohdateien + 669 JSON-Blueprints (ca. 1.358 Einträge) | MIT-Datensatz, `.schem`/`.litematic`/Blueprint |

```powershell
# Quellen und bewusst nicht eingebaute Seiten anzeigen
python scrape_structures.py --list-sources

# Abfielder: erst 20 Browse-Seiten testen, max. 200 neue Builds
python scrape_structures.py --source abfielder --abfielder-pages 20 --max-items 200 --delay 2

# Kompletter Abfielder-Lauf (resume-fähig, kann mehrere Stunden dauern)
python scrape_structures.py --source abfielder --delay 2.5 --jitter 1.0

# Hugging Face: Rohdateien + Blueprints (Standard)
python scrape_structures.py --source huggingface

# Nur die 669 JSON-Blueprints rekonstruieren
python scrape_structures.py --source huggingface --blueprints-only

# Beide Quellen, insgesamt max. 500 neue Einträge
python scrape_structures.py --source all --max-items 500

# prompts.json nachträglich aus structures_meta.jsonl neu schreiben
python scrape_structures.py --source abfielder --rebuild-prompts
```

Ausgabe-Layout (pro Quelle ein eigener Ordner, direkt als `--data-dir` nutzbar):

```
Trainingsdaten/
├── abfielder/
│   ├── structures_meta.jsonl        # Herkunft, Titel, Beschreibung, Dateien, Status
│   ├── prompts.json                 # {"<datei>.schem": ["<name>", "<prompt>"]}
│   ├── abfielder-<id>__<name>.schem
│   ├── abfielder-<id>__<name>.txt   # Prompt-Sidecar (Name / '---' / Beschreibung)
│   └── abfielder-<id>__<name>.litematic   # optional via --keep-litematic
└── huggingface/
    ├── structures_meta.jsonl
    ├── prompts.json
    ├── huggingface-<id>__<name>.schem
    └── huggingface-<id>__<name>.txt
```

### Litematica-Konvertierung

`app/litematic.py` wandelt `.litematic` lokal in Sponge-`.schem` um – ohne
externe Tools. Unterstützt werden:

* beide Litematica-Bitpacking-Varianten (klassisch mit Restbits pro Long und
  das kontinuierliche Packing neuerer Versionen),
* mehrere Regionen, negative `Size`-Werte und daraus resultierende
  Bounding-Boxen,
* BlockEntities (Positionen werden korrekt auf die Schematic-Min-Ecke
  umgerechnet) und Entities,
* das JSON-Blueprint-Format aus `Hack337/Minecraft-Schematics`
  (`size`/`palette`/`layers`, Luft = `.`).

Die erzeugten Dateien werden vor dem Schreiben validiert (gzip, Sponge-Root,
Dimensionen, Voxel-Limit). Zu große Strukturen werden mit `--max-voxels`
übersprungen.

### Wichtige Optionen

| Option | Wirkung |
| --- | --- |
| `--source` | `abfielder`, `huggingface` oder `all` (Komma-Liste möglich) |
| `--out-root` | Zielordner (Standard `../Trainingsdaten`, je Quelle ein Unterordner) |
| `--max-items` / `--limit` | Pro Quelle bzw. insgesamt N neue Einträge verarbeiten |
| `--metadata-only` | nur Metadaten/Prompts, keine Strukturdateien |
| `--abfielder-pages` / `--abfielder-start-page` | Browse-Seiten begrenzen bzw. verschieben |
| `--blueprints-only` / `--no-blueprints` | Hugging Face: nur Blueprints bzw. nur Rohdateien |
| `--max-voxels` | Strukturen über N Voxel überspringen (Standard 20 Mio.) |
| `--max-file-mb` | Downloads über N MB überspringen (Standard 250) |
| `--keep-litematic` | Original `.litematic` neben der `.schem` behalten |
| `--dedupe-by-hash` | identische Zieldateien überspringen |
| `--redownload` / `--refresh-meta` | Dateien bzw. Metadaten erneut laden |
| `--order` | `source`, `shuffle` oder `reverse` |
| `--ignore-robots` | robots.txt ignorieren (nicht empfohlen) |
| `--dry-run` | nur Discovery, keine Downloads |

Der Lauf ist **unterbrechbar und resume-fähig**: Bereits vollständige Einträge
werden übersprungen, fehlgeschlagene bleiben mit `ok: false` in
`structures_meta.jsonl` und werden beim nächsten Lauf erneut versucht.

> **Rechtlicher Hinweis:** Die Rechte an den Builds liegen bei den jeweiligen
> Creator:innen. Herkunft, Autor und Lizenz stehen in `structures_meta.jsonl`.
> Der Scraper umgeht keine Logins, Paywalls oder `robots.txt`-Sperren. Bewusst
> nicht eingebaut sind u. a. Planet Minecraft, minecraft-schematics.com,
> MineSchematic, Shulkr, Bloxelizer und GrabCraft (Login, Paywall,
> KI-Crawler-Verbot oder gesperrte `/api`-Pfade). Die Verantwortung für
> Nutzung, Weitergabe und Veröffentlichung der Daten liegt beim Ausführenden.

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

## Vorschau-Performance

Die 2D-Vorschau (`app/voxel_preview.py`) rendert auf einem Worker-Thread, damit
die Oberfläche beim Drehen (Ziehen mit der Maus, Zoom-Slider, Mausrad) nicht
blockiert:

* Beim Ziehen werden nur Frames in reduzierter Auflösung (60 %) gezeichnet und
  sofort hochskaliert; ein Frame in voller Qualität folgt ~0,16 s nach dem Ende
  der Interaktion (`PREVIEW_*`-Konstanten in `app/main_app.py`).
* Kommen während eines Rendervorgangs weitere Anfragen, werden die alten
  verworfen ("frame dropping") statt aufgestaut - die Vorschau hinkt dadurch
  nie hinterher.
* Intern wird eine `VoxelScene` einmal je Bauwerk vorbereitet (Positionen,
  Farben, Nachbar-Belegung). Pro Frame werden nur sichtbare Oberflächen-Flächen
  gezeichnet (von festen Nachbarn verdeckte Flächen entfallen) und die
  Projektion läuft vektorisiert in numpy.
* Transparente Blöcke (Glas, Wasser) werden von Pillow direkt beim Zeichnen
  geblendet, ohne Zwischenbild.
* Für sehr große Strukturen (mehrere hunderttausend Blöcke, z. B. einzelne
  importierte Projekte) bleibt der 3D-Viewer die bessere Wahl - dort rendert die
  GPU.
