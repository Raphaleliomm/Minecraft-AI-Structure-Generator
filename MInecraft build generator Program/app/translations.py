"""UI translations for Minecraft Structure Generator.
Supports German (de) and English (en). Default is English."""
from __future__ import annotations

# All UI strings keyed by a stable identifier.
# Add new strings here, then use self._tr("key") in the app.
TRANSLATIONS = {
    # ─── Status messages ───
    "status_ready": {"en": "✅ Ready", "de": "✅ Bereit"},
    "status_loading_models": {"en": "⏳ Loading models...", "de": "⏳ Lade Modelle..."},
    "status_no_model": {"en": "⚠️ No model loaded", "de": "⚠️ Kein Modell geladen"},
    "status_done": {"en": "✅ Done", "de": "✅ Fertig"},
    "status_generating": {"en": "⏳ Generating structure...", "de": "⏳ Generiere Struktur..."},
    "status_enter_prompt": {"en": "❌ Please enter a prompt", "de": "❌ Bitte Prompt eingeben"},
    "status_project_saved": {"en": "💾 Project saved: {name}", "de": "💾 Projekt gespeichert: {name}"},
    "status_exported": {"en": "📤 Exported: {path}", "de": "📤 Exportiert: {path}"},
    "status_settings_saved": {"en": "✅ Settings saved", "de": "✅ Einstellungen gespeichert"},
    "status_training_stopping": {"en": "⏹ Stopping and saving training...", "de": "⏹ Training wird gestoppt und gespeichert..."},
    "status_training_started": {"en": "⏳ Training started ({type}, {grid})...", "de": "⏳ Training gestartet ({type}, {grid})..."},
    "status_transformer_done": {"en": "✅ Transformer training complete", "de": "✅ Transformer Training abgeschlossen"},
    "status_diffusion_done": {"en": "✅ Diffusion training complete", "de": "✅ Diffusion Training abgeschlossen"},
    "status_tf_diffusion_done": {"en": "✅ Transformer Diffusion training complete", "de": "✅ Transformer Diffusion Training abgeschlossen"},
    "status_tf_training_started": {"en": "⏳ TF-Diffusion training started ({mode})...", "de": "⏳ TF-Diffusion Training gestartet ({mode})..."},
    "status_unet_preset": {"en": "✅ UNet Preset: {label}", "de": "✅ UNet Preset: {label}"},
    "status_encoder_loading": {"en": "⏳ Loading {name}...", "de": "⏳ Lade {name}..."},
    "status_encoder_loaded": {"en": "✅ {name} loaded", "de": "✅ {name} geladen"},
    "status_encoder_error": {"en": "❌ Encoder error: {msg}", "de": "❌ Encoder Fehler: {msg}"},
    "status_model_not_found": {"en": "❌ Model {name} not found", "de": "❌ Modell {name} nicht gefunden"},
    "status_model_loading": {"en": "📥 Loading model: {name}...", "de": "📥 Lade Modell: {name}..."},
    "status_model_default_set": {"en": "👑 {name} is now default {type}", "de": "👑 {name} ist jetzt Standard-{type}"},
    "status_model_default_failed": {"en": "❌ Could not set {name} as default", "de": "❌ Konnte {name} nicht als Standard setzen"},
    "status_model_renamed": {"en": "✏️ {old} → {new}", "de": "✏️ {old} → {new}"},
    "status_model_rename_failed": {"en": "❌ Rename failed (does {name} exist?)", "de": "❌ Umbenennen fehlgeschlagen (existiert {name}?)"},
    "status_model_deleted": {"en": "🗑 {name} deleted", "de": "🗑 {name} gelöscht"},
    "status_invalid_params": {"en": "❌ Invalid training parameters", "de": "❌ Ungültige Trainingsparameter"},
    "status_no_encoder": {"en": "❌ Please load the Text Encoder first or enable Pre-Computed HS", "de": "❌ Bitte zuerst den Text Encoder laden oder Pre-Computed HS aktivieren"},
    "status_no_cache": {"en": "❌ Please select a cache", "de": "❌ Bitte einen Cache auswählen"},
    "status_cache_invalid": {"en": "❌ Cache not current: {msg}", "de": "❌ Cache nicht aktuell: {msg}"},
    "status_cache_warning": {"en": "⚠️ Cache not current: {msg}", "de": "⚠️ Cache nicht aktuell: {msg}"},
    "status_no_encoder_loaded": {"en": "❌ Please load the Text Encoder first", "de": "❌ Bitte zuerst den Text Encoder laden"},
    "status_no_training_data": {"en": "❌ No training data found", "de": "❌ Keine Trainingsdaten gefunden"},
    "status_computing_hs": {"en": "⏳ Computing Hidden States for {count} structures...", "de": "⏳ Berechne Hidden States für {count} Strukturen..."},
    "status_hs_saved": {"en": "✅ Hidden States saved: {name}", "de": "✅ Hidden States gespeichert: {name}"},
    "status_no_project": {"en": "⚠️ No project selected", "de": "⚠️ Kein Projekt ausgewählt"},
    "status_no_schem": {"en": "❌ No .schem file in project", "de": "❌ Keine .schem Datei im Projekt"},
    "status_project_loaded": {"en": "✅ Project loaded: {name}", "de": "✅ Projekt geladen: {name}"},
    "status_project_deleted": {"en": "🗑 Deleted: {name}", "de": "🗑 Gelöscht: {name}"},
    "status_project_exported": {"en": "📤 Exported to: {dest}", "de": "📤 Exportiert nach: {dest}"},
    "status_load_error": {"en": "❌ Error loading: {err}", "de": "❌ Fehler beim Laden: {err}"},
    "status_pyglet_missing": {"en": "❌ pyglet not installed", "de": "❌ pyglet nicht installiert"},
    "status_kaggle_creating": {"en": "⏳ Creating Kaggle export...", "de": "⏳ Erstelle Kaggle-Export..."},
    "status_kaggle_done": {"en": "✅ Kaggle export created: {name}", "de": "✅ Kaggle-Export erstellt: {name}"},
    "status_kaggle_failed": {"en": "❌ Export failed: {err}", "de": "❌ Export fehlgeschlagen: {err}"},

    # ─── Tab names ───
    "tab_generate": {"en": "🎨 Generate", "de": "🎨 Generieren"},
    "tab_models": {"en": "🧠 Models", "de": "🧠 Modelle"},
    "tab_training": {"en": "🎯 Training", "de": "🎯 Training"},
    "tab_projects": {"en": "📂 Projects", "de": "📂 Projekte"},
    "tab_settings": {"en": "⚙️ Settings", "de": "⚙️ Einstellungen"},
    "tab_about": {"en": "ℹ️ About", "de": "ℹ️ Über"},

    # ─── Generate tab ───
    "model_type": {"en": "Model Type:", "de": "Modell-Typ:"},
    "version": {"en": "Version:", "de": "Version:"},
    "build_description": {"en": "Build Description:", "de": "Baubeschreibung:"},
    "parameters": {"en": "Parameters:", "de": "Parameter:"},
    "temperature": {"en": "Temperature:", "de": "Temperatur:"},
    "top_k": {"en": "Top-K:", "de": "Top-K:"},
    "diff_steps": {"en": "Diff. Steps:", "de": "Diff. Steps:"},
    "btn_generate": {"en": "✦ Generate", "de": "✦ Generieren"},
    "btn_generating": {"en": "⏳ Generating...", "de": "⏳ Generiere..."},
    "btn_save": {"en": "💾 Save", "de": "💾 Speichern"},
    "btn_export_schem": {"en": "📤 Export .schem", "de": "📤 Export .schem"},
    "rotation": {"en": "↔ Rotation:", "de": "↔ Drehung:"},
    "preview_hint": {"en": "🖱 Drag horizontally to rotate\nScroll to zoom", "de": "🖱 Horizontal ziehen zum Drehen\nScroll zum Zoomen"},
    "btn_3d_viewer": {"en": "🎮 Open 3D Viewer (OpenGL)", "de": "🎮 3D Viewer öffnen (OpenGL)"},
    "btn_3d_viewer_missing": {"en": "🎮 3D Viewer (pyglet missing)", "de": "🎮 3D Viewer (pyglet fehlt)"},
    "pyglet_missing": {"en": "⚠️ pyglet missing", "de": "⚠️ pyglet fehlt"},

    # ─── Models tab ───
    "model_manager": {"en": "🧠 Model Manager", "de": "🧠 Modell-Manager"},
    "btn_scan": {"en": "🔄 Scan", "de": "🔄 Scannen"},
    "btn_back_editor": {"en": "⬅ Back to Editor", "de": "⬅ Zurück zum Editor"},
    "tf_diffusion_models": {"en": "🤖 Transformer Diffusion Models", "de": "🤖 Transformer Diffusion Modelle"},
    "transformer_models": {"en": "⚡ Transformer Models", "de": "⚡ Transformer Modelle"},
    "diffusion_models": {"en": "🌀 Diffusion Models", "de": "🌀 Diffusion Modelle"},
    "no_tf_diffusion": {"en": "  (No Transformer Diffusion models found)", "de": "  (Keine Transformer Diffusion Modelle gefunden)"},
    "no_transformer": {"en": "  (No Transformer models found)", "de": "  (Keine Transformer-Modelle gefunden)"},
    "no_diffusion": {"en": "  (No Diffusion models found)", "de": "  (Keine Diffusion-Modelle gefunden)"},
    "btn_load": {"en": "► Load", "de": "► Laden"},
    "btn_set_default": {"en": "👑 Set Default", "de": "👑 Als Standard"},
    "btn_train_more": {"en": "🎯 Train More", "de": "🎯 Weiter trainieren"},
    "btn_rename": {"en": "✏️ Rename", "de": "✏️ Umbenennen"},
    "btn_delete": {"en": "🗑 Delete", "de": "🗑 Löschen"},
    "default_badge": {"en": "  👑 Default", "de": "  👑 Standard"},
    "blocks_label": {"en": "Blocks", "de": "Blöcke"},
    "epochs_label": {"en": "Epochs", "de": "Epochen"},

    # ─── Training tab ───
    "training_title": {"en": "🎯 Training", "de": "🎯 Training"},
    "tf_diffusion_settings": {"en": "🤖 TF-Diffusion — Settings", "de": "🤖 TF-Diffusion — Einstellungen"},
    "transformer_settings": {"en": "⚡ Transformer — Settings", "de": "⚡ Transformer — Einstellungen"},
    "diffusion_settings": {"en": "🌀 Diffusion — Settings", "de": "🌀 Diffusion — Einstellungen"},
    "text_encoder": {"en": "🧠 Text Encoder:", "de": "🧠 Text Encoder:"},
    "encoder_not_loaded": {"en": "⏳ Not loaded", "de": "⏳ Nicht geladen"},
    "btn_load_encoder": {"en": "📥 Load Encoder", "de": "📥 Encoder laden"},
    "hidden_states": {"en": "💾 Pre-Computed Hidden States", "de": "💾 Pre-Computed Hidden States"},
    "use_cached_hs": {"en": "Use Pre-Computed Hidden States (faster)", "de": "Pre-Computed Hidden States verwenden (schneller)"},
    "select_cache": {"en": "Select cache:", "de": "Cache auswählen:"},
    "no_cache_loaded": {"en": "⏳ No cache loaded", "de": "⏳ Kein Cache geladen"},
    "no_cache_selected": {"en": "⏳ No cache selected", "de": "⏳ Kein Cache ausgewählt"},
    "select_cache_first": {"en": "⏳ Please select a cache first", "de": "⏳ Bitte erst einen Cache auswählen"},
    "no_cache_used": {"en": "⏳ No cache used", "de": "⏳ Kein Cache verwendet"},
    "btn_precompute": {"en": "📥 Pre-compute", "de": "📥 Vorberechnen"},
    "btn_check_cache": {"en": "🔍 Check cache", "de": "🔍 Cache prüfen"},
    "unet_preset": {"en": "🏗️ UNet Preset:", "de": "🏗️ UNet-Preset:"},
    "model_size": {"en": "Model Size (Parameters):", "de": "Modellgröße (Parameter):"},
    "target_arch": {"en": "🔍 Target: {val}M → ✅ ~{params}M parameters\n📐 d_model={d_model}, nhead={nhead}, layers={layers}, FFN={ffn}", "de": "🔍 Ziel: {val}M → ✅ ~{params}M Parameter\n📐 d_model={d_model}, nhead={nhead}, layers={layers}, FFN={ffn}"},
    "manual_arch": {"en": "✅ Manual: ~{params}M parameters\n📐 d_model={d_model}, nhead={nhead}, layers={layers}, FFN={ffn}", "de": "✅ Manuell: ~{params}M Parameter\n📐 d_model={d_model}, nhead={nhead}, layers={layers}, FFN={ffn}"},
    "no_arch_found": {"en": "❌ No matching architecture found", "de": "❌ Keine passende Architektur gefunden"},
    "invalid_values": {"en": "❌ Invalid values", "de": "❌ Ungültige Werte"},
    "btn_advanced": {"en": "▶ Advanced Settings", "de": "▶ Advanced Settings"},
    "btn_advanced_open": {"en": "▼ Advanced Settings", "de": "▼ Advanced Settings"},
    "btn_calculate": {"en": "↻ Calculate", "de": "↻ Berechnen"},
    "diffusion_fixed_arch": {"en": "Diffusion uses fixed architecture:\nd_model=128, d_text=64, channels=64, (1,2,2)", "de": "Diffusion verwendet feste Architektur:\nd_model=128, d_text=64, channels=64, (1,2,2)"},
    "diffusion_shared_params": {"en": "Standard parameters (Grid, Epochs, Batch, LR,\nAugmentation) are taken from the shared\ninputs above.", "de": "Standard-Parameter (Grid, Epochen, Batch, LR,\nAugmentation) werden von den gemeinsamen\nEingaben oben übernommen."},
    "grid_size": {"en": "Grid Size:", "de": "Grid-Größe:"},
    "grid_experimental": {"en": "🧱 96×96×96+ is experimental and not recommended.", "de": "🧱 96×96×96+ ist experimentell und nicht empfohlen zu nutzen."},
    "training_params": {"en": "Training Parameters", "de": "Training Parameter"},
    "epochs": {"en": "Epochs:", "de": "Epochen:"},
    "batch_size": {"en": "Batch Size:", "de": "Batch Size:"},
    "learning_rate": {"en": "Learning Rate:", "de": "Learning Rate:"},
    "aug_diversity": {"en": "Shift/Rotation Diversity:", "de": "Verschiebe-/Rotations-Vielfalt:"},
    "allow_vertical": {"en": "Allow upward movement", "de": "Nach oben bewegen erlauben"},
    "air_weight": {"en": "🏗️ Air Weight (50-100):", "de": "🏗️ Luft-Gewicht (50-100):"},
    "air_weight_hint": {"en": "We recommend keeping the default setting of 75.", "de": "Wir empfehlen die Standard-Einstellung von 75 beizubehalten."},
    "progress": {"en": "Progress:", "de": "Fortschritt:"},
    "btn_train_transformer": {"en": "🎯 Train Transformer", "de": "🎯 Transformer trainieren"},
    "btn_train_diffusion": {"en": "🎯 Train Diffusion", "de": "🎯 Diffusion trainieren"},
    "btn_train_tf_diffusion": {"en": "🤖 Train TF-Diffusion", "de": "🤖 TF-Diffusion trainieren"},
    "btn_stop_save": {"en": "⏹ Stop & Save", "de": "⏹ Stop & Speichern"},
    "btn_kaggle_export": {"en": "📤 Export to Kaggle", "de": "📤 Export to Kaggle"},

    # ─── Train More dialog ───
    "train_more_title": {"en": "Train More: {name}", "de": "Weiter trainieren: {name}"},
    "train_more_header": {"en": "{icon} Train More: {name}", "de": "{icon} Weiter trainieren: {name}"},
    "params_label": {"en": "Parameters:", "de": "Parameter:"},
    "gpu_available": {"en": "GPU: {'✅ CUDA available' if avail else '❌ CPU only'}", "de": "GPU: {'✅ CUDA verfügbar' if avail else '❌ CPU only'}"},
    "btn_start_training": {"en": "🎯 Start Training", "de": "🎯 Training starten"},
    "btn_cancel": {"en": "Cancel", "de": "Abbrechen"},
    "values_must_be_positive": {"en": "Values must be positive", "de": "Werte müssen positiv sein"},

    # ─── Rename dialog ───
    "rename_title": {"en": "Rename Model", "de": "Modell umbenennen"},
    "rename_prompt": {"en": "New name for '{old}':", "de": "Neuer Name für '{old}':"},

    # ─── Delete dialog ───
    "delete_title": {"en": "Delete Model", "de": "Modell löschen"},
    "delete_confirm": {"en": "🚨 Really delete '{name}'?\n\n{path}", "de": "🚨 Wirklich '{name}' löschen?\n\n{path}"},

    # ─── Projects tab ───
    "saved_projects": {"en": "📂 Saved Projects", "de": "📂 Gespeicherte Projekte"},
    "btn_reload": {"en": "↻ Reload", "de": "↻ Neu laden"},
    "btn_load_project": {"en": "▶️ Load", "de": "▶️ Laden"},
    "btn_delete_project": {"en": "🗑 Delete", "de": "🗑 Löschen"},
    "select_project": {"en": "Select a project from the list", "de": "Wähle ein Projekt aus der Liste"},
    "btn_export_project": {"en": "📤 Export as .schem", "de": "📤 Exportieren als .schem"},

    # ─── Settings tab ───
    "general": {"en": "General", "de": "Allgemein"},
    "use_gpu": {"en": "Use GPU:", "de": "GPU verwenden:"},
    "language": {"en": "Language:", "de": "Sprache:"},
    "btn_save_settings": {"en": "💾 Save Settings", "de": "💾 Einstellungen speichern"},

    # ─── About tab ───
    "about_ai_generator": {"en": "An AI-powered Minecraft Structure Generator", "de": "Ein KI-gestützter Minecraft Struktur Generator"},
    "about_features": {"en": "🚀 Features:", "de": "🚀 Features:"},
    "about_tf_diffusion": {"en": "  • 🤖 Transformer Diffusion (Main model)", "de": "  • 🤖 Transformer Diffusion (Hauptmodell)"},
    "about_frozen_encoder": {"en": "     - Frozen pre-trained Text Encoder (Phi-3.5, Gemma 2/3, Flan-T5)", "de": "     - Frozen pre-trained Text Encoder (Phi-3.5, Gemma 2/3, Flan-T5)"},
    "about_cross_attn": {"en": "     - Cross-Attention instead of Average Pooling", "de": "     - Cross-Attention statt Average Pooling"},
    "about_3d_unet": {"en": "     - 3D UNet with discrete Denoising Diffusion", "de": "     - 3D UNet mit diskreter Denoising Diffusion"},
    "about_transformer": {"en": "  • ⚡ Transformer Model (Single-Pass)", "de": "  • ⚡ Transformer Modell (Single-Pass)"},
    "about_diffusion": {"en": "  • 🌀 3D Diffusion Model (discrete Denoising)", "de": "  • 🌀 3D Diffusion Modell (diskrete Denoising)"},
    "about_3d_preview": {"en": "  • 3D Voxel Preview (freely rotatable with mouse, zoomable)", "de": "  • 3D Voxel-Vorschau (frei drehbar mit Maus, zoombar)"},
    "about_text_to_struct": {"en": "  • Text-to-Structure Generation", "de": "  • Text-zu-Struktur Generierung"},
    "about_model_manager": {"en": "  • Model Manager (multiple versions, Default, Train More)", "de": "  • Modell-Manager (mehrere Versionen, Default, Train More)"},
    "about_project_mgmt": {"en": "  • Project Management with Save & Export", "de": "  • Projekt-Management mit Speichern & Export"},
    "about_gpu_training": {"en": "  • GPU-accelerated Training", "de": "  • GPU-beschleunigtes Training"},
    "about_training_data": {"en": "📊 Training data: analyzed + archived structures", "de": "📊 Trainingsdaten: gut analysierte + ausgelagerte Strukturen"},
    "about_grid_sizes": {"en": "🎯 Available Grid Sizes: {sizes}", "de": "🎯 Verfügbare Grid-Größen: {sizes}"},
    "about_created_with": {"en": "Created with PyTorch, CustomTkinter & much ❤️", "de": "Erstellt mit PyTorch, CustomTkinter & viel ❤️"},

    # ─── Generation info ───
    "gen_transformer_done": {"en": "✅ Transformer generation complete\n📐 {x}×{y}×{z} Voxels\n⚙️ Temp={temp}, Top-K={topk}\n📝 Prompt: \"{prompt}...\"", "de": "✅ Transformer Generierung abgeschlossen\n📐 {x}×{y}×{z} Voxels\n⚙️ Temp={temp}, Top-K={topk}\n📝 Prompt: \"{prompt}...\""},
    "gen_diffusion_done": {"en": "✅ Diffusion generation complete\n📐 {x}×{y}×{z} Voxels\n⚙️ Temp={temp}, Steps={steps}\n📝 Prompt: \"{prompt}...\"", "de": "✅ Diffusion Generierung abgeschlossen\n📐 {x}×{y}×{z} Voxels\n⚙️ Temp={temp}, Steps={steps}\n📝 Prompt: \"{prompt}...\""},
    "gen_tf_diffusion_done": {"en": "✅ TF-Diffusion generation complete\n📐 {x}×{y}×{z} Voxels\n⚙️ Temp={temp}, Top-K={topk}, Steps={steps}\n🧠 Encoder: {encoder}\n📝 Prompt: \"{prompt}...\"", "de": "✅ TF-Diffusion Generierung abgeschlossen\n📐 {x}×{y}×{z} Voxels\n⚙️ Temp={temp}, Top-K={topk}, Steps={steps}\n🧠 Encoder: {encoder}\n📝 Prompt: \"{prompt}...\""},
    "gen_no_model": {"en": "⚠️ No suitable model loaded", "de": "⚠️ Kein passendes Modell geladen"},
    "gen_block_types": {"en": "🧱 {count} block types after trim", "de": "🧱 {count} Blocktypen nach Trim"},
    "gen_unsaved_blocks": {"en": "⚠️ {count} blocks not in vocabulary: marked with ! in preview", "de": "⚠️ {count} Blöcke nicht im Vokabular: in der Vorschau mit ! markiert"},
    "gen_trim": {"en": "Trim: {rx}x{ry}x{rz} -> {tx}x{ty}x{tz}", "de": "Trim: {rx}x{ry}x{rz} -> {tx}x{ty}x{tz}"},
    "gen_error": {"en": "❌ Error: {error}", "de": "❌ Fehler: {error}"},
    "gen_loaded": {"en": "✅ Loaded: {name}\n📐 {x}×{y}×{z}\n📝 {prompt}", "de": "✅ Geladen: {name}\n📐 {x}×{y}×{z}\n📝 {prompt}"},

    # ─── Kaggle dialog ───
    "kaggle_title": {"en": "Export to Kaggle", "de": "Export to Kaggle"},
    "kaggle_msg": {"en": "🚀 Export to Kaggle (2× T4)\n\nA complete Kaggle notebook will be created with:\n\n📐 Grid: {gx}×{gy}×{gz}\n📊 Epochs: {epochs}\n📦 Batch Size: {batch}\n⚡ Learning Rate: {lr}\n🔄 Diversity (Aug.): {aug}\n⬆ Vertical movement: {vertical}\n\n📁 Training data ({count} folders) will be packed as ZIP.\n\nContinue?", "de": "🚀 Export to Kaggle (2× T4)\n\nEin vollständiges Kaggle-Notebook wird erstellt mit:\n\n📐 Grid: {gx}×{gy}×{gz}\n📊 Epochen: {epochs}\n📦 Batch Size: {batch}\n⚡ Learning Rate: {lr}\n🔄 Vielfalt (Aug.): {aug}\n⬆ Vertikale Bewegung: {vertical}\n\n📁 Trainingsdaten ({count} Ordner) werden als ZIP eingepackt.\n\nFortfahren?"},
    "kaggle_success_title": {"en": "Export successful", "de": "Export erfolgreich"},
    "kaggle_success_msg": {"en": "Kaggle export created at:\n\n{path}\n\n📁 Contains:\n  • kaggle_notebook.ipynb\n  • training_data.zip\n  • model.py, dataset.py, train.py\n  • README.md with instructions\n\nUpload the files to Kaggle and select GPU T4 x2 as accelerator.", "de": "Kaggle-Export erstellt unter:\n\n{path}\n\n📁 Enthält:\n  • kaggle_notebook.ipynb\n  • training_data.zip\n  • model.py, dataset.py, train.py\n  • README.md mit Anleitung\n\nLade die Dateien auf Kaggle hoch und wähle GPU T4 x2 als Accelerator."},
    "kaggle_failed_title": {"en": "Export failed", "de": "Export fehlgeschlagen"},

    # ─── Misc ───
    "unknown": {"en": "Unknown", "de": "Unbekannt"},
    "no_cache": {"en": "(no cache)", "de": "(kein Cache)"},
    "none": {"en": "(none)", "de": "(keine)"},
    "computing": {"en": "⏳ Computing...", "de": "⏳ Berechne..."},
    "cache_created": {"en": "✅ Cache created ({count} structures)", "de": "✅ Cache erstellt ({count} Strukturen)"},
    "cache_error": {"en": "❌ Error: {msg}", "de": "❌ Fehler: {msg}"},
    "encoder_not_loaded_yet": {"en": "⏳ {name} - not loaded yet", "de": "⏳ {name} - noch nicht geladen"},
    "encoder_loaded_info": {"en": "✅ {name} loaded (dim={dim}, frozen)", "de": "✅ {name} geladen (dim={dim}, frozen)"},
    "batch_progress": {"en": "Batch {bn}/{tb}", "de": "Batch {bn}/{tb}"},
    "epoch_progress": {"en": "Epoch {e}/{epochs}, Loss={loss}", "de": "Epoche {e}/{epochs}, Loss={loss}"},
    "tf_epoch_progress": {"en": "TF-Diffusion Epoch {e}/{epochs}, Loss={loss}", "de": "TF-Diffusion Epoche {e}/{epochs}, Loss={loss}"},
    "transformer_epoch_progress": {"en": "Transformer Epoch {e}/{epochs}, Loss={loss}", "de": "Transformer Epoche {e}/{epochs}, Loss={loss}"},
    "diffusion_epoch_progress": {"en": "Diffusion Epoch {e}/{epochs}, Loss={loss}", "de": "Diffusion Epoche {e}/{epochs}, Loss={loss}"},
    "loss_time": {"en": "Loss: {loss} | Time: {time}s", "de": "Loss: {loss} | Zeit: {time}s"},
    "tf_loss_time": {"en": "TF-Diffusion | Loss: {loss} | Time: {time}s", "de": "TF-Diffusion | Loss: {loss} | Zeit: {time}s"},
    "tf_error": {"en": "❌ Error: {msg}", "de": "❌ Fehler: {msg}"},
    "training_error": {"en": "❌ Error: {msg}", "de": "❌ Fehler: {msg}"},
    "with_cached_hs": {"en": "with Pre-Computed HS", "de": "mit Pre-Computed HS"},
    "with_encoder": {"en": "with Encoder", "de": "mit Encoder"},

    # ─── Advanced settings labels ───
    "adv_d_model": {"en": "d_model (Model Dimension)", "de": "d_model (Modell-Dimension)"},
    "adv_nhead": {"en": "nhead (Attention Heads)", "de": "nhead (Attention Heads)"},
    "adv_layers": {"en": "Layers (Decoder Levels)", "de": "Layers (Decoder-Ebenen)"},
    "adv_ff_ratio": {"en": "FFN Multiplier", "de": "FFN Multiplikator"},

    # ─── Train More dialog labels ───
    "tm_epochs": {"en": "Epochs:", "de": "Epochen:"},
    "tm_batch_size": {"en": "Batch Size:", "de": "Batch Size:"},
    "tm_learning_rate": {"en": "Learning Rate:", "de": "Learning Rate:"},
    "tm_diversity": {"en": "Diversity:", "de": "Vielfalt:"},

    # ─── About tab title ───
    "about_title": {"en": "🏗️ {name} v{version}", "de": "🏗️ {name} v{version}"},

    # ─── Additional status messages ───
    "status_no_suitable_model": {"en": "⚠️ No suitable model loaded", "de": "⚠️ Kein passendes Modell geladen"},
    "status_pyglet_not_installed": {"en": "❌ pyglet not installed", "de": "❌ pyglet nicht installiert"},
    "status_delete_confirm_title": {"en": "Delete Model", "de": "Modell löschen"},
    "status_delete_confirm": {"en": "🚨 Really delete '{name}'?\n\n{path}", "de": "🚨 Wirklich '{name}' löschen?\n\n{path}"},
    "status_rename_title": {"en": "Rename Model", "de": "Modell umbenennen"},
    "status_rename_prompt": {"en": "New name for '{old}':", "de": "Neuer Name für '{old}':"},
    "status_rename_success": {"en": "✏️ {old} → {new}", "de": "✏️ {old} → {new}"},
    "status_rename_failed": {"en": "❌ Rename failed (does {name} exist?)", "de": "❌ Umbenennen fehlgeschlagen (existiert {name}?)"},
}


def get_text(key: str, lang: str = "en", **kwargs) -> str:
    """Get translated text for a key.
    
    Args:
        key: Translation key
        lang: Language code ("en" or "de")
        **kwargs: Format arguments for the string
        
    Returns:
        Translated string, or the key itself if not found.
    """
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    text = entry.get(lang, entry.get("en", key))
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError):
            return text
    return text