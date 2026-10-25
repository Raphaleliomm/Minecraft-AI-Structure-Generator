"""Minecraft Structure Generator.
CustomTkinter-based multi-tab app with 3D preview, model switching,
project management, settings, training tab, and Model Manager.

Main focus: Transformer Diffusion model with frozen pre-trained text encoders
(Phi-3.5, Gemma 2/3, T5) and cross-attention."""
from __future__ import annotations

import gc
import json
import logging
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import customtkinter as ctk
import numpy as np
import torch
from PIL import Image

from app.config import AppConfig
from app.constants import (
    ALLOWED_GRID_SIZES,
    DEFAULT_GRID_SIZE_LABEL,
    GRID_SIZE_MAP,
    GRID_SIZE_OPTIONS,
    grid_size_for_label,
    grid_size_label,
)
from app.diffusion_model import (
    VoxelDiffusionModel,
    TransformerDiffusionModel,
    clamp_num_steps,
    train_diffusion_step,
    train_transformer_diffusion_step,
    train_transformer_diffusion_step_cached,
)
from app.grad_accum import (
    accumulation_enabled,
    apply_optimizer_step,
    iter_micro_batches,
    per_sample_weight_mass,
)
from app.model_manager import ModelRegistry, ModelEntry
from app.paths import EXPORTS_DIR, PROJECTS_DIR, RUNS_DIR, default_data_dirs
from app.transformer_encoder import (
    DEFAULT_ENCODER_HIDDEN_DIM,
    ENCODER_HIDDEN_DIMS,
    FrozenTransformerEncoder,
    MODEL_NAMES,
    encoder_working_set_bytes,
    list_supported_models,
)
from app.hidden_state_cache import (
    compute_hidden_states, load_hidden_states, validate_cache, delete_cache, list_caches,
)
from app.training_memory import (
    GIB,
    GENERATION_CONTEXT_TOKENS,
    describe_generation_plan,
    describe_training_plan,
    format_gib,
    plan_diffusion_training,
    plan_generation,
    plan_transformer_training,
    system_ram_bytes,
    transformer_generation_bytes,
    unet_inference_bytes,
)
from app.voxel_preview import PreviewRenderer, VoxelScene, view_angles
from app.voxel_viewer_3d import open_3d_viewer, HAS_PYGLET
from app.translations import get_text as _get_text

from dataset import (
    PromptTokenizer,
    VoxelTokenizer,
    center_token_grid,
    collate_context_batch,
    discover_prompt_files,
    load_schematic,
    save_schem,
    trim_token_grid,
)
from model import SharedWeightVoxelTransformer
from kaggle_export import create_kaggle_export

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

logger = logging.getLogger(__name__)

# ─── Constants ───
APP_NAME = "Minecraft Structure Generator"
VERSION = "0.1"

# Minecraft-inspired UI skin. The app stays CustomTkinter, but the colors,
# edges and typography are intentionally closer to the in-game menus.
MC_BG = "#151913"
MC_PANEL = "#2b2118"
MC_PANEL_ALT = "#352719"
MC_INSET = "#1c160f"
MC_GRASS = "#4f9135"
MC_GRASS_DARK = "#326d2d"
MC_GRASS_LIGHT = "#75b852"
MC_DIRT = "#6f4a2c"
MC_DIRT_DARK = "#4a311f"
MC_STONE = "#6f6f6f"
MC_STONE_DARK = "#3f3f3f"
MC_TEXT = "#f6f0d8"
MC_MUTED = "#c8bda3"
MC_GOLD = "#f0c247"
MC_RED = "#a84032"
MC_RED_DARK = "#742a23"
MC_FONT = "Consolas"

# ─── Preview rendering ───
# Panels are drawn by a worker thread (app.voxel_preview.PreviewRenderer) so the
# Tk main loop never blocks on a render. While the camera is dragged the frame is
# rendered at a lower resolution and upscaled for display; a full quality frame
# follows right after the interaction stops.
PREVIEW_SIZE = (420, 380)
PREVIEW_DRAG_RESOLUTION = 0.6
PREVIEW_POLL_MS = 20
PREVIEW_REFRESH_MS = 160

# ─── Transformer Diffusion UNet presets ───
# Each preset: (display_name, channels, channel_multipliers, d_model, cross_attn_heads)
TF_DIFFUSION_PRESETS = [
    ("Tiny (0.5M)", 16, (1, 2, 2), 32, 2),
    ("Small (1.5M)", 32, (1, 2, 2), 64, 4),
    ("Medium (4.5M)", 48, (1, 2, 3), 96, 4),
    ("Large (12M)", 64, (1, 2, 3, 4), 128, 6),
    ("XL (35M)", 96, (1, 2, 3, 4), 192, 8),
    ("XXL (80M)", 128, (1, 2, 3, 4, 4), 256, 8),
]

# ─── Text-Encoder: VRAM oder System-RAM? ───
# Passt der Encoder schaetzungsweise in diesen Anteil des freien VRAM, wird er
# ohne Nachfrage auf die GPU geladen.
ENCODER_VRAM_FRACTION = 0.9
# Fuer den RAM-Ausweichpfad: fp32 nur, wenn der freie RAM das mit Reserve
# hergibt (beim Laden entsteht kurzzeitig eine zweite Kopie) und der Encoder
# klein genug ist (fp32 ist auf der CPU schneller als fp16).
ENCODER_CPU_RAM_SHARE = 0.5
ENCODER_CPU_FP32_LIMIT = 1 * GIB

# Grid size options are shared with train.py / generate.py (app/constants.py)


def release_cuda_cache() -> None:
    """Nicht mehr benoetigte CUDA-Bloecke an den Treiber zurueckgeben.

    Der PyTorch-Caching-Allocator behaelt freigegebene Bloecke im eigenen Pool;
    ``torch.cuda.mem_get_info()`` (die Grundlage der VRAM-Planung) sieht diesen
    Speicher als *belegt* an. Nach einem Trainingslauf - oder nach einem
    fehlgeschlagenen Encoder-Aufbau - bleiben so mehrere GB reserviert und die
    naechste Aktion endet mit "Cuda out of memory", obwohl nichts mehr benutzt
    wird. Auf der CPU ist der Aufruf ein No-Op.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class MinecraftStructureApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.config = AppConfig.load()

        # ─── Window Setup ───
        self.title(f"{APP_NAME} v{VERSION}")
        self.geometry("1400x850")
        self.minsize(1100, 700)
        self.configure(fg_color=MC_BG)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ─── Model Registry ───
        self.model_registry = ModelRegistry()
        self._discover_models()

        # ─── State ───
        self.generated_grid: Optional[torch.Tensor] = None
        self.generated_np: Optional[np.ndarray] = None
        self.id_to_block: Optional[List[str]] = None
        self.current_project_path: Optional[Path] = None
        self.current_preview_img: Optional[ctk.CTkImage] = None
        self.current_view: str = "free"
        # Startmodell-Typ aus config.json (dort bereits validiert)
        self.model_type: str = self.config.default_model
        self.generation_dit_mode: bool = False
        self.transformer_model: Optional[SharedWeightVoxelTransformer] = None
        self.dit_encoder: Optional[FrozenTransformerEncoder] = None
        self.diffusion_model: Optional[VoxelDiffusionModel] = None
        self.tf_diffusion_model: Optional[TransformerDiffusionModel] = None
        self.tf_encoder: Optional[FrozenTransformerEncoder] = None
        # Einmal pro Sitzung beantwortete Frage "Encoder in den System-RAM?".
        self._encoder_cpu_choice: Dict[str, bool] = {}
        # Modell-Loader laufen in eigenen Threads: ohne diese Sperre koennten
        # zwei gleichzeitig einen 7-GB-Encoder laden (RAM/Rueckfrage-Chaos).
        self._encoder_load_lock = threading.Lock()
        self.prompt_tokenizer: Optional[PromptTokenizer] = None
        self.voxel_tokenizer: Optional[VoxelTokenizer] = None
        self.current_transformer_name: Optional[str] = None
        self.current_diffusion_name: Optional[str] = None
        self.current_tf_diffusion_name: Optional[str] = None
        # Datenordner werden aus app/paths.py abgeleitet (relativ zum Projekt,
        # unabhängig vom Arbeitsverzeichnis). Der Ordner mit den bewusst
        # ausgelagerten "zu gross"-Strukturen gehört NICHT dazu.
        self.data_dirs = default_data_dirs()
        self.generation_running = False
        self.training_running = False
        self._current_training_type: Optional[str] = None

        # Preview rendering runs on a worker thread; the Tk loop only polls for
        # finished frames (see PREVIEW_* constants above).
        self.preview_renderer = PreviewRenderer()
        self._preview_scene_cache: Optional[VoxelScene] = None
        self._preview_scene_source: Optional[np.ndarray] = None
        self._preview_scene_palette: Optional[List[str]] = None
        self._preview_poll_job: Optional[str] = None
        self._preview_refresh_job: Optional[str] = None

        # ─── Build UI ───
        self._build_ui()
        self._setup_tab_icons()
        self._apply_minecraft_skin()
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # ─── Load default models ───
        self.after(100, self._load_models_async)

    def _on_window_close(self):
        """Stop the preview worker before tearing the window down."""
        try:
            self.preview_renderer.stop()
        finally:
            self.destroy()

    def _tr(self, key: str, **kwargs) -> str:
        """Translate a UI string using the current language."""
        return _get_text(key, self.config.language, **kwargs)

    def _rebuild_ui(self):
        """Rebuild the entire UI (used when language changes)."""
        # Destroy all children and rebuild
        for widget in self.winfo_children():
            widget.destroy()
        self._build_ui()
        self._setup_tab_icons()
        self._apply_minecraft_skin()
        self._refresh_model_combo()
        self._refresh_models_tab()
        self._refresh_projects()

    def _discover_models(self):
        """Scan runs/ and load default names from config."""
        self.model_registry.load_config_defaults(
            self.config.default_transformer_name,
            self.config.default_diffusion_name,
            getattr(self.config, 'default_tf_diffusion_name', None),
        )
        self.model_registry.discover()

    # ═══════════════════════════════════════════════════════════════
    # UI BUILD
    # ═══════════════════════════════════════════════════════════════

    def _mc_font(self, size: int, weight: Optional[str] = None) -> tuple:
        return (MC_FONT, size, weight) if weight else (MC_FONT, size)

    def _safe_configure(self, widget, **kwargs) -> None:
        try:
            widget.configure(**kwargs)
        except Exception:
            pass

    def _is_transparent(self, widget) -> bool:
        try:
            return widget.cget("fg_color") == "transparent"
        except Exception:
            return False

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, height=58, corner_radius=0,
                              fg_color=MC_GRASS_DARK, border_width=3, border_color=MC_STONE_DARK)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        header.grid_columnconfigure(1, weight=1)
        header.grid_propagate(False)
        header._mc_skin_locked = True

        ctk.CTkLabel(header, text="MC", width=44, font=self._mc_font(30, "bold"),
                     text_color=MC_GOLD, fg_color=MC_DIRT, corner_radius=0,
        ).grid(row=0, column=0, sticky="nsw", padx=(8, 10), pady=8)

        title_stack = ctk.CTkFrame(header, fg_color="transparent")
        title_stack.grid(row=0, column=1, sticky="ew")
        ctk.CTkLabel(title_stack, text=APP_NAME, font=self._mc_font(22, "bold"), text_color=MC_TEXT,
                     ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(title_stack, text=f"v{VERSION}  |  Text zu Minecraft .schem",
                     font=self._mc_font(11), text_color=MC_MUTED,
                     ).grid(row=1, column=0, sticky="w")

    def _skin_widget_tree(self, widget) -> None:
        if getattr(widget, "_mc_skin_locked", False):
            return
        cls = widget.__class__.__name__
        if cls in {"CTkFrame", "CTkScrollableFrame"} and not self._is_transparent(widget):
            self._safe_configure(widget, fg_color=MC_PANEL, border_width=2,
                                 border_color=MC_STONE_DARK, corner_radius=0)
        elif cls == "CTkTabview":
            self._safe_configure(widget, fg_color=MC_BG, border_width=2, border_color=MC_STONE_DARK,
                                 corner_radius=0, segmented_button_fg_color=MC_DIRT_DARK,
                                 segmented_button_selected_color=MC_GRASS_DARK,
                                 segmented_button_selected_hover_color=MC_GRASS,
                                 segmented_button_unselected_color=MC_PANEL_ALT,
                                 segmented_button_unselected_hover_color=MC_DIRT, text_color=MC_TEXT)
        elif cls == "CTkLabel":
            self._safe_configure(widget, text_color=MC_TEXT, font=self._mc_font(12))
            if not self._is_transparent(widget):
                self._safe_configure(widget, fg_color=MC_INSET, corner_radius=0, text_color=MC_TEXT)
        elif cls == "CTkButton":
            text = ""
            try:
                text = str(widget.cget("text")).lower()
            except Exception:
                pass
            if "löschen" in text or "delete" in text:
                fg, hover = MC_RED, MC_RED_DARK
            elif "generieren" in text or "laden" in text or "speichern" in text or "save" in text:
                fg, hover = MC_GRASS_DARK, MC_GRASS
            else:
                fg, hover = MC_DIRT, MC_DIRT_DARK
            self._safe_configure(widget, fg_color=fg, hover_color=hover, text_color=MC_TEXT,
                                 border_width=2, border_color=MC_STONE_DARK, corner_radius=0,
                                 font=self._mc_font(12, "bold"))
        elif cls in {"CTkTextbox", "CTkEntry"}:
            self._safe_configure(widget, fg_color=MC_INSET, border_color=MC_STONE_DARK,
                                 border_width=2, corner_radius=0, text_color=MC_TEXT,
                                 font=self._mc_font(12))
        elif cls == "CTkOptionMenu":
            self._safe_configure(widget, fg_color=MC_DIRT, button_color=MC_GRASS_DARK,
                                 button_hover_color=MC_GRASS, dropdown_fg_color=MC_PANEL_ALT,
                                 dropdown_hover_color=MC_DIRT, dropdown_text_color=MC_TEXT,
                                 text_color=MC_TEXT, corner_radius=0, font=self._mc_font(12))
        elif cls == "CTkSegmentedButton":
            self._safe_configure(widget, fg_color=MC_DIRT_DARK, selected_color=MC_GRASS_DARK,
                                 selected_hover_color=MC_GRASS, unselected_color=MC_PANEL_ALT,
                                 unselected_hover_color=MC_DIRT, text_color=MC_TEXT, corner_radius=0,
                                 font=self._mc_font(12, "bold"))
        elif cls == "CTkSlider":
            self._safe_configure(widget, fg_color=MC_DIRT_DARK, progress_color=MC_GRASS,
                                 button_color=MC_GRASS_LIGHT, button_hover_color=MC_GOLD, border_width=1)
        elif cls == "CTkProgressBar":
            self._safe_configure(widget, fg_color=MC_DIRT_DARK, progress_color=MC_GRASS_LIGHT,
                                 border_width=1, corner_radius=0)
        elif cls == "CTkSwitch":
            self._safe_configure(widget, progress_color=MC_GRASS, button_color=MC_GOLD,
                                 button_hover_color=MC_GRASS_LIGHT, fg_color=MC_DIRT_DARK,
                                 text_color=MC_TEXT, font=self._mc_font(12))
        for child in widget.winfo_children():
            self._skin_widget_tree(child)

    def _apply_minecraft_skin(self) -> None:
        self._safe_configure(self, fg_color=MC_BG)
        self._skin_widget_tree(self)

    def _build_ui(self):
        """Build the complete UI with tab view."""
        self._build_header()
        self.tab_view = ctk.CTkTabview(self, corner_radius=0)
        self.tab_view.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.tab_view.grid_rowconfigure(0, weight=1)
        self.tab_view.grid_columnconfigure(0, weight=1)

        # ─── Create Tabs ───
        self.tab_generate = self.tab_view.add(self._tr("tab_generate"))
        self.tab_models = self.tab_view.add(self._tr("tab_models"))
        self.tab_training = self.tab_view.add(self._tr("tab_training"))
        self.tab_projects = self.tab_view.add(self._tr("tab_projects"))
        self.tab_settings = self.tab_view.add(self._tr("tab_settings"))
        self.tab_about = self.tab_view.add(self._tr("tab_about"))

        self._build_generate_tab()
        self._build_models_tab()
        self._build_training_tab()
        self._build_projects_tab()
        self._build_settings_tab()
        self._build_about_tab()

        # ─── Status Bar ───
        self.status_var = ctk.StringVar(value=self._tr("status_ready"))
        self.status_bar = ctk.CTkLabel(self, textvariable=self.status_var,
                                       fg_color=MC_DIRT_DARK, corner_radius=0,
                                       text_color=MC_TEXT, font=self._mc_font(12, "bold"),
                                       anchor="w", padx=12, pady=4)
        self.status_bar._mc_skin_locked = True
        self.status_bar.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 8))

    # ═══════════════════════════════════════════════════════════════
    # GENERATE TAB
    # ═══════════════════════════════════════════════════════════════

    def _build_generate_tab(self):
        tab = self.tab_generate
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=0, minsize=480)
        tab.grid_columnconfigure(1, weight=1)

        self.azimuth = 45.0
        self.elevation = 30.0
        self.zoom_scale = 3.5

        # ─── Left Panel ───
        left = ctk.CTkFrame(tab, corner_radius=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.grid_rowconfigure(2, weight=1)
        left.grid_columnconfigure(0, weight=1)

        # Model selector row
        model_frame = ctk.CTkFrame(left, fg_color="transparent")
        model_frame.grid(row=0, column=0, sticky="ew", pady=(10, 4), padx=10)
        model_frame.grid_columnconfigure(1, weight=1)
        model_frame.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(model_frame, text=self._tr("model_type"), font=("Segoe UI", 14, "bold")).grid(row=0, column=0, padx=(0, 8))
        self.model_type_selector = ctk.CTkSegmentedButton(
            model_frame, values=["TF-Diffusion", "DiT"],
            command=self._on_model_type_change, font=("Segoe UI", 12),
        )
        self.model_type_selector.grid(row=0, column=1, sticky="ew", padx=(0, 12))
        self.model_type_selector.set("TF-Diffusion")

        ctk.CTkLabel(model_frame, text=self._tr("version"), font=("Segoe UI", 14, "bold")).grid(row=0, column=2, padx=(0, 8))
        self.model_version_combo = ctk.CTkOptionMenu(
            model_frame, values=[self._tr("none")], font=("Segoe UI", 12), command=self._on_model_version_change,
        )
        self.model_version_combo.grid(row=0, column=3, sticky="ew")

        # Prompt
        prompt_frame = ctk.CTkFrame(left, fg_color="transparent")
        prompt_frame.grid(row=1, column=0, sticky="nsew", pady=6, padx=10)
        prompt_frame.grid_rowconfigure(1, weight=1)
        prompt_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(prompt_frame, text=self._tr("build_description"), font=("Segoe UI", 13, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.prompt_text = ctk.CTkTextbox(prompt_frame, height=120, font=("Segoe UI", 12), wrap="word")
        self.prompt_text.grid(row=1, column=0, sticky="nsew")
        self.prompt_text.insert("1.0", "small medieval wooden cottage with stone foundation and steep oak roof")

        # Parameters
        params_frame = ctk.CTkFrame(left, fg_color="transparent")
        params_frame.grid(row=2, column=0, sticky="nsew", pady=6, padx=10)
        params_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(params_frame, text=self._tr("parameters"), font=("Segoe UI", 13, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ctk.CTkLabel(params_frame, text=self._tr("temperature")).grid(row=1, column=0, sticky="w", padx=(0, 8))
        self.temp_slider = ctk.CTkSlider(params_frame, from_=0.0, to=1.5, number_of_steps=15)
        self.temp_slider.grid(row=1, column=1, sticky="ew", padx=(0, 8))
        self.temp_slider.set(self.config.temperature)
        self.temp_label = ctk.CTkLabel(params_frame, text=f"{self.config.temperature:.2f}", width=40)
        self.temp_label.grid(row=1, column=2, sticky="w")

        ctk.CTkLabel(params_frame, text=self._tr("top_k")).grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        self.topk_slider = ctk.CTkSlider(params_frame, from_=5, to=100, number_of_steps=19)
        self.topk_slider.grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=(6, 0))
        self.topk_slider.set(self.config.top_k)
        self.topk_label = ctk.CTkLabel(params_frame, text=f"{self.config.top_k}", width=40)
        self.topk_label.grid(row=2, column=2, sticky="w", pady=(6, 0))

        self.diff_steps_label = ctk.CTkLabel(params_frame, text=self._tr("diff_steps"))
        self.diff_steps_slider = ctk.CTkSlider(params_frame, from_=10, to=500)
        self.diff_steps_slider.set(self.config.diffusion_steps)
        self.diff_steps_value = ctk.CTkLabel(params_frame, text=f"{self.config.diffusion_steps}", width=40)
        self.diff_steps_label.grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        self.diff_steps_slider.grid(row=3, column=1, sticky="ew", padx=(0, 8), pady=(6, 0))
        self.diff_steps_value.grid(row=3, column=2, sticky="w", pady=(6, 0))
        self._update_diffusion_visibility()
        self.temp_slider.configure(command=lambda v: self.temp_label.configure(text=f"{v:.2f}"))
        self.topk_slider.configure(command=lambda v: self.topk_label.configure(text=f"{int(v)}"))
        self.diff_steps_slider.configure(command=lambda v: self.diff_steps_value.configure(text=f"{int(v)}"))

        # Action buttons
        btn_frame = ctk.CTkFrame(left, fg_color="transparent")
        btn_frame.grid(row=3, column=0, sticky="ew", pady=10, padx=10)
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)
        btn_frame.grid_columnconfigure(2, weight=1)

        self.generate_btn = ctk.CTkButton(btn_frame, text=self._tr("btn_generate"), font=("Segoe UI", 14, "bold"),
                                          height=40, fg_color="#2563eb", hover_color="#1d4ed8",
                                          command=self._generate_structure)
        self.generate_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.save_btn = ctk.CTkButton(btn_frame, text=self._tr("btn_save"), font=("Segoe UI", 13), height=40,
                                      state="disabled", command=self._save_to_project)
        self.save_btn.grid(row=0, column=1, sticky="ew", padx=4)
        self.export_btn = ctk.CTkButton(btn_frame, text=self._tr("btn_export_schem"), font=("Segoe UI", 13),
                                        height=40, state="disabled", fg_color="#1e293b",
                                        command=self._export_schematic)
        self.export_btn.grid(row=0, column=2, sticky="ew", padx=(4, 0))

        # ─── Right Panel: Preview ───
        right = ctk.CTkFrame(tab, corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.model_status = ctk.CTkLabel(right, text=self._tr("status_loading_models"), font=("Segoe UI", 12),
                                         fg_color=("gray90", "gray20"), corner_radius=6)
        self.model_status.grid(row=0, column=0, sticky="ew", pady=(10, 4), padx=10)

        orbit_frame = ctk.CTkFrame(right, fg_color="transparent")
        orbit_frame.grid(row=1, column=0, sticky="ew", pady=4, padx=10)
        orbit_frame.grid_columnconfigure(0, weight=0)
        orbit_frame.grid_columnconfigure(1, weight=0)
        orbit_frame.grid_columnconfigure(2, weight=0)
        orbit_frame.grid_columnconfigure(3, weight=1)
        orbit_frame.grid_columnconfigure(4, weight=0)

        ctk.CTkLabel(orbit_frame, text=self._tr("rotation"), font=("Segoe UI", 12)).grid(row=0, column=0, padx=(0, 6))
        ctk.CTkButton(orbit_frame, text="NW", font=("Segoe UI", 11), width=50, height=26,
                      command=lambda: self._set_horizontal_orbit(45)).grid(row=0, column=1, padx=2)
        ctk.CTkButton(orbit_frame, text="NO", font=("Segoe UI", 11), width=50, height=26,
                      command=lambda: self._set_horizontal_orbit(315)).grid(row=0, column=2, padx=2)
        ctk.CTkButton(orbit_frame, text="SO", font=("Segoe UI", 11), width=50, height=26,
                      command=lambda: self._set_horizontal_orbit(225)).grid(row=0, column=3, padx=2)
        ctk.CTkLabel(orbit_frame, text="Zoom", font=("Segoe UI", 12)).grid(row=0, column=4, padx=(8, 2))
        self.zoom_slider = ctk.CTkSlider(orbit_frame, from_=1.5, to=6.0, number_of_steps=45, width=80)
        self.zoom_slider.grid(row=0, column=5, padx=(0, 4))
        self.zoom_slider.set(self.zoom_scale)
        self.zoom_slider.configure(command=lambda v: self._update_zoom(v))
        # while the slider is dragged only low resolution frames are drawn
        self.zoom_slider.bind("<ButtonRelease-1>", lambda _event: self._update_preview())

        self.preview_label = ctk.CTkLabel(right, text=self._tr("preview_hint"),
                                          font=("Segoe UI", 14))
        self.preview_label.grid(row=2, column=0, sticky="nsew", padx=10, pady=4)

        self._drag_start_x = 0
        self._drag_start_y = 0
        self._orbit_start_az = 45.0
        self._orbit_start_el = 30.0
        self._dragging = False
        self.preview_label.bind("<Button-1>", self._on_mouse_down)
        self.preview_label.bind("<B1-Motion>", self._on_mouse_drag)
        self.preview_label.bind("<ButtonRelease-1>", self._on_mouse_up)
        right.bind("<Button-1>", self._on_mouse_down, add="+")
        right.bind("<B1-Motion>", self._on_mouse_drag, add="+")
        right.bind("<ButtonRelease-1>", self._on_mouse_up, add="+")
        self.preview_label.bind("<MouseWheel>", self._on_mouse_wheel)
        right.bind("<MouseWheel>", self._on_mouse_wheel, add="+")

        self.info_frame = ctk.CTkFrame(right, fg_color="transparent")
        self.info_frame.grid(row=3, column=0, sticky="ew", pady=8, padx=10)
        self.info_var = ctk.StringVar(value="")
        self.info_label = ctk.CTkLabel(self.info_frame, textvariable=self.info_var,
                                       font=("Segoe UI", 11), justify="left",
                                       fg_color=("gray90", "gray20"), corner_radius=6, padx=10, pady=8)
        self.info_label.grid(row=0, column=0, sticky="ew")

        viewer_btn_frame = ctk.CTkFrame(right, fg_color="transparent")
        viewer_btn_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=(4, 4))
        viewer_btn_frame.grid_columnconfigure(0, weight=1)
        viewer_btn_frame.grid_columnconfigure(1, weight=1)

        self.viewer_btn = ctk.CTkButton(viewer_btn_frame, text=self._tr("btn_3d_viewer"),
                                        font=("Segoe UI", 13, "bold"), height=36,
                                        fg_color="#7c3aed", hover_color="#6d28d9",
                                        state="disabled", command=self._open_3d_viewer)
        self.viewer_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.viewer_status = ctk.CTkLabel(viewer_btn_frame, text="", font=("Segoe UI", 11),
                                          fg_color=("gray90", "gray20"), corner_radius=4)
        self.viewer_status.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.progress_bar = ctk.CTkProgressBar(right, mode="indeterminate")
        self.progress_bar.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 8))
        self.progress_bar.set(0)
        if not HAS_PYGLET:
            self.viewer_status.configure(text=self._tr("pyglet_missing"), text_color="#f87171")
            self.viewer_btn.configure(state="disabled", text=self._tr("btn_3d_viewer_missing"))

        # Refresh model list
        self._refresh_model_combo()

    # ═══════════════════════════════════════════════════════════════
    # MODELS TAB
    # ═══════════════════════════════════════════════════════════════

    def _build_models_tab(self):
        tab = self.tab_models
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        scroll = ctk.CTkScrollableFrame(tab, corner_radius=10)
        scroll.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        scroll.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(scroll, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text=self._tr("model_manager"), font=("Segoe UI", 22, "bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(header, text=self._tr("btn_scan"), font=("Segoe UI", 12),
                      command=self._refresh_models_tab, width=120).grid(row=0, column=1, padx=8)
        ctk.CTkButton(header, text=self._tr("btn_back_editor"), font=("Segoe UI", 12),
                      command=lambda: self.tab_view.set(self._tr("tab_generate")), width=160).grid(row=0, column=2)

        sep1 = ctk.CTkFrame(scroll, height=3, fg_color="#2563eb")
        sep1.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        self._model_table_frames: Dict[str, ctk.CTkFrame] = {}
        self.models_container = ctk.CTkFrame(scroll, fg_color="transparent")
        self.models_container.grid(row=2, column=0, sticky="nsew")
        self.models_container.grid_columnconfigure(0, weight=1)
        self._refresh_models_tab()

    def _refresh_models_tab(self):
        self._discover_models()
        for widget in self.models_container.winfo_children():
            widget.destroy()
        row = 0
        # TF-Diffusion
        ctk.CTkLabel(self.models_container, text=self._tr("tf_diffusion_models"),
                     font=("Segoe UI", 18, "bold"), text_color="#c084fc").grid(row=row, column=0, sticky="w", pady=(0, 8))
        row += 1
        tf_diff_models = self.model_registry.get_by_type("transformer_diffusion")
        if not tf_diff_models:
            ctk.CTkLabel(self.models_container, text=self._tr("no_tf_diffusion"),
                         font=("Segoe UI", 12), text_color=("gray60", "gray50")).grid(row=row, column=0, sticky="w", pady=(0, 12))
            row += 1
        else:
            for entry in tf_diff_models:
                self._build_model_card(self.models_container, row, entry)
                row += 1
        sep = ctk.CTkFrame(self.models_container, height=2, fg_color=("gray70", "gray30"))
        sep.grid(row=row, column=0, sticky="ew", pady=16)
        row += 1
        # Recurrent-depth DiT models only (plain Transformer checkpoints are hidden by the registry).
        ctk.CTkLabel(self.models_container, text="DiT Models",
                     font=("Segoe UI", 18, "bold"), text_color="#60a5fa").grid(row=row, column=0, sticky="w", pady=(0, 8))
        row += 1
        transformers = self.model_registry.get_by_type("transformer")
        if not transformers:
            ctk.CTkLabel(self.models_container, text="(No recurrent-depth DiT models found)",
                         font=("Segoe UI", 12), text_color=("gray60", "gray50")).grid(row=row, column=0, sticky="w", pady=(0, 12))
            row += 1
        else:
            for entry in transformers:
                self._build_model_card(self.models_container, row, entry)
                row += 1
        self._refresh_model_combo()
        self._skin_widget_tree(self.models_container)

    def _build_model_card(self, parent, row: int, entry: ModelEntry):
        is_default = ((entry.model_type == "transformer" and self.model_registry.default_transformer == entry.name) or
                      (entry.model_type == "diffusion" and self.model_registry.default_diffusion == entry.name) or
                      (entry.model_type == "transformer_diffusion" and self.model_registry.default_tf_diffusion == entry.name))
        card = ctk.CTkFrame(parent, corner_radius=8, fg_color=("gray90", "gray20"),
                            border_width=2, border_color="#3b82f6" if is_default else ("gray70", "gray30"))
        card.grid(row=row, column=0, sticky="ew", pady=4)
        card.grid_columnconfigure(0, weight=1)
        hdr = ctk.CTkFrame(card, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 4))
        hdr.grid_columnconfigure(0, weight=1)
        name_text = f"{entry.name}"
        if is_default:
            name_text += self._tr("default_badge")
        ctk.CTkLabel(hdr, text=name_text, font=("Segoe UI", 15, "bold")).grid(row=0, column=0, sticky="w")
        badge_color = "#3b82f6" if entry.model_type == "transformer" else "#8b5cf6"
        if entry.model_type == "transformer_diffusion":
            badge_color = "#c084fc"
        model_label = "DIT" if entry.model_type == "transformer" else entry.model_type.upper()
        ctk.CTkLabel(hdr, text=f" {model_label} ", font=("Segoe UI", 11, "bold"),
                     fg_color=badge_color, text_color="white", corner_radius=4, padx=8).grid(row=0, column=1, padx=4)
        meta = ctk.CTkFrame(card, fg_color="transparent")
        meta.grid(row=1, column=0, sticky="ew", padx=14, pady=2)
        meta.grid_columnconfigure(0, weight=1)
        grid_str = f"{entry.grid_size[0]}×{entry.grid_size[1]}×{entry.grid_size[2]}"
        bl = self._tr("blocks_label")
        if entry.model_type == "transformer":
            text_encoder_info = f"  text={entry.text_encoder_name}" if entry.text_encoder_name else "  text=learned"
            info = (f"d_model={entry.d_model}  max depth={entry.recurrent_depth}{text_encoder_info}  "
                    f"MoE={entry.moe_active}/{entry.moe_experts}  heads={entry.nhead}  "
                    f"FFN={entry.dim_feedforward}  {bl}={entry.block_vocab_size}")
        elif entry.model_type == "transformer_diffusion":
            info = (f"d_model={entry.d_model}  channels={entry.channels}  steps={entry.num_timesteps}  "
                    f"ctx={entry.context_dim}  encoder={entry.encoder_name or '?'}  {bl}={entry.block_vocab_size}")
        else:
            info = (f"d_model={entry.d_model}  channels={entry.channels}  steps={entry.num_timesteps}  "
                    f"{bl}={entry.block_vocab_size}")
        ctk.CTkLabel(meta, text=f"{grid_str}  {info}", font=("Segoe UI", 11),
                     text_color=("gray50", "gray40")).grid(row=0, column=0, sticky="w")
        if entry.epochs_trained > 0:
            ctk.CTkLabel(meta, text=f"{entry.epochs_trained} {self._tr('epochs_label')}  Loss={entry.last_loss:.4f}",
                         font=("Segoe UI", 11), text_color=("gray50", "gray40")).grid(row=1, column=0, sticky="w")
        actions = ctk.CTkFrame(card, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=14, pady=(6, 10))
        actions.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(actions, text=self._tr("btn_load"), font=("Segoe UI", 11), width=80, height=28,
                      fg_color="#2563eb", hover_color="#1d4ed8",
                      command=lambda n=entry.name: self._load_single_model(n)).grid(row=0, column=0, sticky="w", padx=(0, 4))
        ctk.CTkButton(actions, text=self._tr("btn_set_default"), font=("Segoe UI", 11), width=100, height=28,
                      command=lambda n=entry.name: self._set_model_default(n)).grid(row=0, column=1, padx=4)
        ctk.CTkButton(actions, text=self._tr("btn_train_more"), font=("Segoe UI", 11), width=130, height=28,
                      fg_color="#7c3aed", hover_color="#6d28d9",
                      command=lambda n=entry.name: self._train_more_model(n)).grid(row=0, column=2, padx=4)
        self._model_table_frames[entry.name] = card
        ctk.CTkButton(actions, text=self._tr("btn_rename"), font=("Segoe UI", 11), width=110, height=28,
                      command=lambda n=entry.name: self._rename_model_dialog(n)).grid(row=0, column=3, padx=4)
        ctk.CTkButton(actions, text=self._tr("btn_delete"), font=("Segoe UI", 11), width=90, height=28,
                      fg_color="#dc2626", hover_color="#b91c1c",
                      command=lambda n=entry.name: self._delete_model_confirm(n)).grid(row=0, column=4, padx=4)

    # ─── Model Manager Actions ───

    def _load_single_model(self, name: str):
        entry = self.model_registry.get(name)
        if entry is None:
            self.status_var.set(self._tr("status_model_not_found", name=name))
            return
        if entry.model_type == "transformer":
            if entry.recurrent_depth <= 1 and entry.moe_experts <= 0:
                self.status_var.set("Dieser normale Transformer ist nicht in der Generier-Auswahl")
                return
            self.model_type = "transformer"
            self.generation_dit_mode = True
            self.current_transformer_name = name
            self.model_type_selector.set("DiT")
        elif entry.model_type == "diffusion":
            self.status_var.set("Normale Diffusion ist nicht in der Generier-Auswahl")
            return
        elif entry.model_type == "transformer_diffusion":
            self.model_type = "transformer_diffusion"
            self.generation_dit_mode = False
            self.current_tf_diffusion_name = name
            self.model_type_selector.set("TF-Diffusion")
        self.tab_view.set(self._tr("tab_generate"))
        self._refresh_model_combo()
        self._load_models_async()
        self.status_var.set(self._tr("status_model_loading", name=name))

    def _set_model_default(self, name: str):
        if self.model_registry.set_default(name):
            entry = self.model_registry.get(name)
            if entry:
                if entry.model_type == "transformer":
                    self.config.default_transformer_name = name
                elif entry.model_type == "diffusion":
                    self.config.default_diffusion_name = name
                elif entry.model_type == "transformer_diffusion":
                    self.config.default_tf_diffusion_name = name
                self.config.save()
            self._refresh_models_tab()
            self.status_var.set(self._tr("status_model_default_set", name=name, type=entry.model_type.upper()))
        else:
            self.status_var.set(self._tr("status_model_default_failed", name=name))

    def _train_more_model(self, name: str):
        entry = self.model_registry.get(name)
        if entry is None:
            self.status_var.set(self._tr("status_model_not_found", name=name))
            return
        self._show_train_more_dialog(name, entry)

    def _show_train_more_dialog(self, name: str, entry: ModelEntry):
        """Dialog fuer 'Weiter trainieren' eines vorhandenen Checkpoints.

        Spiegelt den Training-Tab (gleiche Parameter, gleiche Worker), aber die
        Architektur kommt aus dem Checkpoint: Ziel ist es, genau dieses Modell
        fortzusetzen. Deshalb gibt es hier keine Modellgroessen-/UNet-Auswahl -
        die wuerde nur ein neues Modell erzeugen - und das Grid ist durch den
        Checkpoint vorgegeben.
        """
        width = 620
        height = min(760, max(480, self.winfo_screenheight() - 140))
        dialog = ctk.CTkToplevel(self)
        dialog.title(self._tr("train_more_title", name=name))
        dialog.geometry(f"{width}x{height}")
        dialog.resizable(False, True)
        dialog.transient(self)
        dialog.grab_set()
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - width) // 2
        y = max(0, self.winfo_y() + (self.winfo_height() - height) // 2)
        dialog.geometry(f"+{x}+{y}")
        dialog.grid_columnconfigure(0, weight=1)
        dialog.grid_rowconfigure(1, weight=1)
        self._safe_configure(dialog, fg_color=MC_BG)

        header = ctk.CTkFrame(dialog, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 10))
        header.grid_columnconfigure(0, weight=1)
        icon = {"transformer": "TF", "diffusion": "DF", "transformer_diffusion": "TF-DF"}.get(entry.model_type, "MD")
        ctk.CTkLabel(header, text=self._tr("train_more_header", icon=icon, name=name),
                     font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")

        body = ctk.CTkScrollableFrame(dialog, corner_radius=10)
        body.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 8))
        body.grid_columnconfigure(0, weight=1)

        # ─── Checkpoint-Info ───
        info_frame = ctk.CTkFrame(body, fg_color=("gray90", "gray15"), corner_radius=6)
        info_frame.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        info_frame.grid_columnconfigure(0, weight=1)
        gs = f"{entry.grid_size[0]}×{entry.grid_size[1]}×{entry.grid_size[2]}"
        info_lines = [f"{gs}  |  {entry.model_type.upper()}  |  {self._tr('blocks_label')}: {entry.block_vocab_size}"]
        if entry.model_type == "transformer":
            info_lines.append(f"d_model={entry.d_model}  heads={entry.nhead}  layers={entry.num_layers}  "
                              f"FFN={entry.dim_feedforward}")
        elif entry.model_type == "transformer_diffusion":
            info_lines.append(f"d_model={entry.d_model}  channels={entry.channels}  "
                              f"steps={entry.num_timesteps}  ctx={entry.context_dim}")
            info_lines.append(self._tr("tm_encoder_used", name=entry.encoder_name or "?"))
        else:
            info_lines.append(f"d_model={entry.d_model}  channels={entry.channels}  "
                              f"steps={entry.num_timesteps}")
        if entry.epochs_trained > 0:
            info_lines.append(f"{entry.epochs_trained} {self._tr('epochs_label')}  |  Loss: {entry.last_loss:.4f}")
        ctk.CTkLabel(info_frame, text="\n".join(info_lines), font=("Segoe UI", 11),
                     justify="left").grid(row=0, column=0, sticky="w", padx=14, pady=10)
        ctk.CTkLabel(info_frame, text=self._tr("tm_continue_info", name=name, path=entry.path),
                     font=("Segoe UI", 10), text_color=("#7c3aed", "#c084fc"),
                     justify="left", wraplength=520).grid(row=1, column=0, sticky="w", padx=14, pady=(0, 10))

        row = 1
        ctk.CTkLabel(body, text=self._tr("tm_cont_arch", type=entry.model_type.upper()),
                     font=("Segoe UI", 12, "bold"), text_color="#c084fc").grid(row=row, column=0, sticky="w")
        row += 1
        if entry.model_type in ("diffusion", "transformer_diffusion"):
            ctk.CTkLabel(body, text=self._tr("tm_steps_fixed", steps=entry.num_timesteps or 50),
                         font=("Segoe UI", 10), text_color=("gray50", "gray40")).grid(
                row=row, column=0, sticky="w", pady=(2, 4))
            row += 1

        # ─── Hidden-State-Cache (nur TF-Diffusion, wie im Training-Tab) ───
        cache_var = ctk.BooleanVar(value=False)
        cache_combo = None
        cache_status = None
        cache_dims: Dict[str, int] = {}

        def _cache_dim(cache_label: str) -> int:
            return cache_dims.get(cache_label, 0)

        def _update_cache_status(*_args) -> None:
            if cache_status is None:
                return
            if not cache_var.get():
                cache_status.configure(text=self._tr("no_cache_used"),
                                       text_color=("gray50", "gray40"))
                return
            if cache_combo is None or cache_combo.get() == self._tr("no_cache"):
                cache_status.configure(text=self._tr("select_cache_first"),
                                       text_color=("#f59e0b", "#fbbf24"))
                return
            dim = _cache_dim(cache_combo.get())
            if entry.context_dim and dim and dim != entry.context_dim:
                cache_status.configure(
                    text=self._tr("tm_cache_dim_mismatch", dim=dim, need=entry.context_dim),
                    text_color="#f87171")
                return
            cache_status.configure(
                text=self._tr("tm_cache_dim", dim=dim or "?", need=entry.context_dim or "?"),
                text_color="#34d399")

        if entry.model_type == "transformer_diffusion":
            cache_frame = ctk.CTkFrame(body, fg_color="transparent")
            cache_frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
            cache_frame.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(cache_frame, text=self._tr("hidden_states"), font=("Segoe UI", 13, "bold"),
                         text_color="#34d399").grid(row=0, column=0, sticky="w", pady=(0, 2))
            ctk.CTkSwitch(cache_frame, text=self._tr("use_cached_hs"), variable=cache_var,
                          onvalue=True, offvalue=False,
                          command=_update_cache_status).grid(row=1, column=0, sticky="w", pady=2)
            cache_sel_frame = ctk.CTkFrame(cache_frame, fg_color="transparent")
            cache_sel_frame.grid(row=2, column=0, sticky="ew", pady=2)
            cache_sel_frame.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(cache_sel_frame, text=self._tr("select_cache"), font=("Segoe UI", 11)).grid(
                row=0, column=0, sticky="w", padx=(0, 6))
            cache_dims = {c["encoder_name"]: int(c.get("hidden_dim") or 0) for c in list_caches()}
            cache_combo = ctk.CTkOptionMenu(
                cache_sel_frame, values=list(cache_dims) or [self._tr("no_cache")],
                font=("Segoe UI", 10), width=220, command=_update_cache_status)
            cache_combo.grid(row=0, column=1, sticky="w")
            cache_status = ctk.CTkLabel(cache_frame, text=self._tr("no_cache_used"), font=("Segoe UI", 10),
                                        justify="left", wraplength=520)
            cache_status.grid(row=3, column=0, sticky="w", pady=(2, 0))
            row += 1

        # ─── Trainingsparameter (wie im Training-Tab) ───
        params = ctk.CTkFrame(body, fg_color="transparent")
        params.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        params.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(params, text=self._tr("training_params"), font=("Segoe UI", 13, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        def add_entry_row(grid_row: int, label: str, var: ctk.StringVar) -> None:
            ctk.CTkLabel(params, text=label, font=("Segoe UI", 12)).grid(
                row=grid_row, column=0, sticky="w", padx=(0, 12), pady=4)
            ctk.CTkEntry(params, width=120, textvariable=var, font=("Segoe UI", 12)).grid(
                row=grid_row, column=1, sticky="w", pady=4)

        epochs_var = ctk.StringVar(value="10")
        add_entry_row(1, self._tr("tm_epochs"), epochs_var)
        batch_var = ctk.StringVar(value="16")
        add_entry_row(2, self._tr("tm_batch_size"), batch_var)
        lr_var = ctk.StringVar(value="1.5e-3")
        add_entry_row(3, self._tr("tm_learning_rate"), lr_var)

        epochs_total = ctk.CTkLabel(params, text="", font=("Segoe UI", 10),
                                    text_color=("gray50", "gray40"))
        epochs_total.grid(row=4, column=0, columnspan=3, sticky="w", pady=(0, 2))

        def _update_epochs_total(*_args) -> None:
            try:
                new_epochs = int(str(epochs_var.get()).strip())
            except (TypeError, ValueError):
                epochs_total.configure(text=self._tr("tm_epochs_new", new="?"))
                return
            done = max(0, int(entry.epochs_trained or 0))
            epochs_total.configure(text=self._tr("tm_epochs_total", done=done, new=new_epochs,
                                                total=done + new_epochs))

        epochs_var.trace_add("write", _update_epochs_total)
        _update_epochs_total()

        ctk.CTkLabel(params, text=self._tr("tm_air_weight"), font=("Segoe UI", 12)).grid(
            row=5, column=0, sticky="w", padx=(0, 12), pady=(8, 0))
        air_var = ctk.DoubleVar(value=75.0)
        air_value = ctk.CTkLabel(params, text="75", width=28)
        ctk.CTkSlider(params, from_=50, to=100, number_of_steps=50, variable=air_var,
                      command=lambda v: air_value.configure(text=str(int(round(float(v)))))).grid(
            row=5, column=1, sticky="ew", pady=(8, 0))
        air_value.grid(row=5, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        ctk.CTkLabel(params, text=self._tr("tm_diversity"), font=("Segoe UI", 12)).grid(
            row=6, column=0, sticky="w", padx=(0, 12), pady=4)
        aug_var = ctk.DoubleVar(value=1)
        aug_value = ctk.CTkLabel(params, text="1", width=28)
        ctk.CTkSlider(params, from_=0, to=5, number_of_steps=5, variable=aug_var,
                      command=lambda v: aug_value.configure(text=str(int(round(float(v)))))).grid(
            row=6, column=1, sticky="ew", pady=4)
        aug_value.grid(row=6, column=2, sticky="w", padx=(8, 0), pady=4)

        aug_vertical_switch = ctk.CTkSwitch(params, text=self._tr("allow_vertical"),
                                            onvalue=True, offvalue=False)
        aug_vertical_switch.grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # Batch-Groesse halten (Gradient Accumulation) - wie im Training-Tab.
        grad_accum_var = ctk.BooleanVar(value=accumulation_enabled())
        grad_accum_switch = ctk.CTkSwitch(params, text=self._tr("grad_accum"),
                                          variable=grad_accum_var,
                                          onvalue=True, offvalue=False)
        grad_accum_switch.grid(row=8, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ctk.CTkLabel(params, text=self._tr("grad_accum_hint"), font=("Segoe UI", 9),
                     text_color=("gray50", "gray40"), justify="left", wraplength=520).grid(
            row=9, column=0, columnspan=3, sticky="w")

        gpu_avail = torch.cuda.is_available() and self.config.gpu_enabled
        gpu_text = self._tr("tm_gpu_on") if gpu_avail else self._tr("tm_gpu_off")
        ctk.CTkLabel(params, text=gpu_text,
                     font=("Segoe UI", 11), text_color="#34d399" if gpu_avail else "#f87171",
                     ).grid(row=10, column=0, columnspan=3, sticky="w", pady=(8, 0))


        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(8, 18))
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)

        def on_start():
            try:
                epochs = int(str(epochs_var.get()).strip())
                batch = int(str(batch_var.get()).strip())
                lr = float(str(lr_var.get()).strip())
                air_weight = float(air_var.get())
                aug = int(round(float(aug_var.get())))
                allow_vertical = bool(aug_vertical_switch.get())
                if epochs <= 0 or batch <= 0 or lr <= 0 or aug < 0:
                    raise ValueError(self._tr("values_must_be_positive"))
            except (ValueError, TypeError):
                self.status_var.set(self._tr("status_invalid_params"))
                return
            if self.training_running:
                self.status_var.set(self._tr("status_training_running"))
                return
            use_cache = False
            cache_name = None
            if entry.model_type == "transformer_diffusion" and bool(cache_var.get()):
                cache_name = cache_combo.get() if cache_combo is not None else self._tr("no_cache")
                if cache_name == self._tr("no_cache"):
                    self.status_var.set(self._tr("status_no_cache"))
                    return
                dim = _cache_dim(cache_name)
                if entry.context_dim and dim and dim != entry.context_dim:
                    self.status_var.set(self._tr("tm_cache_dim_mismatch",
                                                 dim=dim, need=entry.context_dim))
                    return
                use_cache = True
            dialog.destroy()
            self._start_train_more(entry, {
                "epochs": epochs, "batch_size": batch, "lr": lr, "air_weight": air_weight,
                "augmentation_diversity": aug, "allow_vertical_movement": allow_vertical,
                "use_cached": use_cache, "cache_name": cache_name,
                "grad_accum": bool(grad_accum_var.get()),
            })

        ctk.CTkButton(btn_frame, text=self._tr("btn_start_training"), font=("Segoe UI", 13, "bold"),
                      height=36, fg_color="#7c3aed", hover_color="#6d28d9", command=on_start,
                     ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkButton(btn_frame, text=self._tr("btn_cancel"), font=("Segoe UI", 13), height=36,
                      fg_color="gray40", hover_color="gray50", command=dialog.destroy,
                     ).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self._skin_widget_tree(dialog)

    def _start_train_more(self, entry: ModelEntry, params: Dict[str, Any]) -> None:
        """'Weiter trainieren' fuer genau diesen Checkpoint starten.

        Die Dialogwerte werden in den Training-Tab gespiegelt und dann die
        normalen Einstiegspunkte mit ``continue_from`` gestartet: beide Wege
        benutzen dieselben Worker und dieselben Fortschrittsbalken. Die
        Architektur kommt aus dem Checkpoint, deshalb wird hier nichts an
        Modellgroesse/UNet-Preset veraendert.
        """
        if self.training_running:
            self.status_var.set(self._tr("status_training_running"))
            return
        grid_size = tuple(entry.grid_size)
        self._sync_training_widgets(entry, params)
        overrides: Dict[str, Any] = {
            "epochs": params["epochs"],
            "batch_size": params["batch_size"],
            "lr": params["lr"],
            "air_weight": params["air_weight"],
            "augmentation_diversity": params["augmentation_diversity"],
            "allow_vertical_movement": params["allow_vertical_movement"],
            "grad_accum": bool(params.get("grad_accum", True)),
            "grid_size": grid_size,
        }
        if entry.model_type == "transformer_diffusion":
            overrides["use_cached"] = bool(params.get("use_cached"))
            overrides["cache_name"] = params.get("cache_name")
            self._start_tf_diffusion_training(continue_from=entry.name, overrides=overrides)
        else:
            self._start_training(entry.model_type, grid_size=grid_size,
                                 overrides=overrides, continue_from=entry.name)

    def _sync_training_widgets(self, entry: ModelEntry, params: Dict[str, Any]) -> None:
        """Dialogwerte in den Training-Tab spiegeln (Anzeige, nicht Berechnung).

        Der Dialog hat keine eigenen Fortschrittsanzeigen: das Training laeuft
        sichtbar im Training-Tab weiter. Damit Anzeige und Lauf zusammenpassen,
        werden Panel, Grid und Parameter dort auf den Checkpoint gesetzt -
        gerechnet wird mit den Werten aus dem Dialog (siehe _start_train_more).
        """
        panel_label = {"transformer": "DiT (Recurrent)",
                       "transformer_diffusion": "TF-Diffusion"}.get(entry.model_type)
        if panel_label:
            self.train_model_type_selector.set(panel_label)
            self._on_train_type_change(panel_label)
        grid_label = grid_size_label(entry.grid_size)
        if grid_label:
            self.grid_size_var.set(grid_label)

        def _set_entry(widget, value) -> None:
            widget.delete(0, "end")
            widget.insert(0, str(value))

        _set_entry(self.train_epochs_entry, params["epochs"])
        _set_entry(self.train_batch_entry, params["batch_size"])
        _set_entry(self.train_lr_entry, params["lr"])
        self.augmentation_diversity_var.set(params["augmentation_diversity"])
        self._update_aug_label(params["augmentation_diversity"])
        if hasattr(self, "air_weight_var"):
            self.air_weight_var.set(params["air_weight"])
            self._update_air_weight_label(params["air_weight"])
        if params["allow_vertical_movement"]:
            self.setting_aug_vertical.select()
        else:
            self.setting_aug_vertical.deselect()
        if hasattr(self, "setting_grad_accum"):
            if params.get("grad_accum", True):
                self.setting_grad_accum.select()
            else:
                self.setting_grad_accum.deselect()
        if entry.model_type == "transformer_diffusion":
            try:
                self.tf_encoder_combo.set(entry.encoder_name or "Phi-3.5-mini")
            except Exception:  # Encoder nicht in der Liste: Auswahl belassen
                pass
            self.tf_use_cached_hs_var.set(bool(params.get("use_cached")))
            if params.get("cache_name"):
                try:
                    self.tf_cache_selector.set(params["cache_name"])
                except Exception:
                    pass
            if entry.num_timesteps:
                _set_entry(self.tf_diff_steps_entry, entry.num_timesteps)
        elif entry.model_type == "diffusion" and entry.num_timesteps:
            _set_entry(self.diff_train_steps_entry, entry.num_timesteps)
        self.tab_view.set(self._tr("tab_training"))

    def _rename_model_dialog(self, old_name: str):
        dialog = ctk.CTkInputDialog(title=self._tr("rename_title"), text=self._tr("rename_prompt", old=old_name))
        new_name = dialog.get_input()
        if new_name and new_name.strip():
            new_name = new_name.strip()
            if self.model_registry.rename(old_name, new_name):
                entry = self.model_registry.get(new_name)
                if entry:
                    if self.config.default_transformer_name == old_name:
                        self.config.default_transformer_name = new_name
                    if self.config.default_diffusion_name == old_name:
                        self.config.default_diffusion_name = new_name
                    if getattr(self.config, 'default_tf_diffusion_name', None) == old_name:
                        self.config.default_tf_diffusion_name = new_name
                    self.config.save()
                self._refresh_models_tab()
                self._refresh_model_combo()
                self.status_var.set(self._tr("status_model_renamed", old=old_name, new=new_name))
            else:
                self.status_var.set(self._tr("status_model_rename_failed", name=new_name))

    def _delete_model_confirm(self, name: str):
        entry = self.model_registry.get(name)
        if entry is None:
            return
        from tkinter.messagebox import askyesno
        if askyesno(self._tr("delete_title"), self._tr("delete_confirm", name=name, path=entry.path)):
            self.model_registry.delete(name)
            if self.config.default_transformer_name == name:
                self.config.default_transformer_name = None
            if self.config.default_diffusion_name == name:
                self.config.default_diffusion_name = None
            if getattr(self.config, 'default_tf_diffusion_name', None) == name:
                self.config.default_tf_diffusion_name = None
            self.config.save()
            self._refresh_models_tab()
            self._refresh_model_combo()
            self.status_var.set(self._tr("status_model_deleted", name=name))

    # ═══════════════════════════════════════════════════════════════
    # TRAINING TAB
    # ═══════════════════════════════════════════════════════════════

    def _build_training_tab(self):
        tab = self.tab_training
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(tab, corner_radius=10)
        scroll.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        scroll.grid_columnconfigure(0, weight=1)

        row = 0

        # ─── Model Type Selector ───
        ctk.CTkLabel(scroll, text=self._tr("training_title"), font=("Segoe UI", 20, "bold")).grid(
            row=row, column=0, sticky="w", pady=(0, 12))
        row += 1

        type_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        type_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        type_frame.grid_columnconfigure(1, weight=1)
        row += 1

        ctk.CTkLabel(type_frame, text=self._tr("model_type"), font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, padx=(0, 8), sticky="w")
        self.train_model_type_selector = ctk.CTkSegmentedButton(
            type_frame, values=["TF-Diffusion", "DiT (Recurrent)"],
            command=self._on_train_type_change, font=("Segoe UI", 12),
        )
        self.train_model_type_selector.grid(row=0, column=1, sticky="w")

        # ─── Container for model-specific panels ───
        self.train_panels_container = ctk.CTkFrame(scroll, fg_color="transparent")
        self.train_panels_container.grid(row=row, column=0, sticky="ew")
        self.train_panels_container.grid_columnconfigure(0, weight=1)

        # ─── Build supported training panels ───
        self._build_tf_diffusion_train_panel(self.train_panels_container)
        self._build_transformer_train_panel(self.train_panels_container)

        # ─── Show default panel later once all widgets exist ───

        # ─── Shared: Grid-Größe ───
        sep = ctk.CTkFrame(scroll, height=2, fg_color=("gray70", "gray30"))
        sep.grid(row=row+1, column=0, sticky="ew", pady=10)

        grid_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        grid_frame.grid(row=row+2, column=0, sticky="ew", pady=(6, 6))
        grid_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(grid_frame, text=self._tr("grid_size"), font=("Segoe UI", 13, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 12))
        self.grid_size_var = ctk.StringVar(value="16×16×16")
        self.grid_size_combo = ctk.CTkOptionMenu(
            grid_frame, values=list(GRID_SIZE_OPTIONS), variable=self.grid_size_var,
            font=("Segoe UI", 12), width=200,
        )
        self.grid_size_combo.grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(grid_frame, text=self._tr("grid_experimental"),
                     font=("Segoe UI", 10), text_color=("gray50", "gray40")).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))

        # ─── Shared: Training Parameter ───
        sep2 = ctk.CTkFrame(scroll, height=2, fg_color=("gray70", "gray30"))
        sep2.grid(row=row+3, column=0, sticky="ew", pady=10)

        ctk.CTkLabel(scroll, text=self._tr("training_params"), font=("Segoe UI", 13, "bold")).grid(
            row=row+4, column=0, sticky="w", pady=(0, 6))

        param_grid = ctk.CTkFrame(scroll, fg_color="transparent")
        param_grid.grid(row=row+5, column=0, sticky="ew", pady=2)
        param_grid.grid_columnconfigure(1, weight=1)
        param_grid.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(param_grid, text=self._tr("epochs")).grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.train_epochs_entry = ctk.CTkEntry(param_grid, width=80)
        self.train_epochs_entry.grid(row=0, column=1, sticky="w")
        self.train_epochs_entry.insert(0, "10")

        ctk.CTkLabel(param_grid, text=self._tr("batch_size")).grid(row=0, column=2, sticky="w", padx=(16, 8))
        self.train_batch_entry = ctk.CTkEntry(param_grid, width=80)
        self.train_batch_entry.grid(row=0, column=3, sticky="w")
        self.train_batch_entry.insert(0, "16")

        ctk.CTkLabel(param_grid, text=self._tr("learning_rate")).grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(4, 0))
        self.train_lr_entry = ctk.CTkEntry(param_grid, width=80)
        self.train_lr_entry.grid(row=1, column=1, sticky="w", pady=(4, 0))
        self.train_lr_entry.insert(0, "1.5e-3")

        ctk.CTkLabel(param_grid, text=self._tr("aug_diversity")).grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        self.augmentation_diversity_var = ctk.DoubleVar(value=0)
        self.augmentation_diversity_slider = ctk.CTkSlider(
            param_grid, from_=0, to=5, number_of_steps=5,
            variable=self.augmentation_diversity_var, command=self._update_aug_label,
        )
        self.augmentation_diversity_slider.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(8, 0))
        self.augmentation_diversity_label = ctk.CTkLabel(param_grid, text="0", width=28)
        self.augmentation_diversity_label.grid(row=2, column=3, sticky="w", pady=(8, 0))

        self.setting_aug_vertical = ctk.CTkSwitch(param_grid, text=self._tr("allow_vertical"),
                                                  onvalue=True, offvalue=False)
        self.setting_aug_vertical.grid(row=3, column=0, columnspan=4, sticky="w", pady=(4, 0))

        # ── Air Weight ──
        ctk.CTkLabel(param_grid, text=self._tr("air_weight")).grid(
            row=4, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        self.air_weight_var = ctk.DoubleVar(value=75.0)
        self.air_weight_slider = ctk.CTkSlider(param_grid, from_=50, to=100, number_of_steps=50,
                                               variable=self.air_weight_var, command=self._update_air_weight_label)
        self.air_weight_slider.grid(row=4, column=1, columnspan=2, sticky="ew", pady=(8, 0))
        self.air_weight_label = ctk.CTkLabel(param_grid, text="75", width=28)
        self.air_weight_label.grid(row=4, column=3, sticky="w", pady=(8, 0))
        ctk.CTkLabel(param_grid, text=self._tr("air_weight_hint"),
                     font=("Segoe UI", 9), text_color=("gray50", "gray40")).grid(
            row=5, column=0, columnspan=4, sticky="w", pady=(0, 4))

        # ── Gradient Accumulation ──
        # Ohne diesen Schalter verkleinert der VRAM-Guard den Batch dauerhaft;
        # mit ihm bleiben die Gradienten ueber mehrere Mikro-Batches erhalten und
        # der Optimizer-Schritt sieht weiterhin den gewuenschten Batch.
        self.grad_accum_var = ctk.BooleanVar(value=accumulation_enabled())
        self.setting_grad_accum = ctk.CTkSwitch(
            param_grid, text=self._tr("grad_accum"), variable=self.grad_accum_var,
            onvalue=True, offvalue=False)
        self.setting_grad_accum.grid(row=6, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ctk.CTkLabel(param_grid, text=self._tr("grad_accum_hint"),
                     font=("Segoe UI", 9), text_color=("gray50", "gray40"),
                     justify="left", wraplength=520).grid(
            row=7, column=0, columnspan=4, sticky="w", pady=(0, 2))

        # ─── Progress Display ───
        progress_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        progress_frame.grid(row=row+6, column=0, sticky="ew", pady=(10, 2))
        progress_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(progress_frame, text=self._tr("progress"), font=("Segoe UI", 13, "bold")).grid(row=0, column=0, sticky="w")
        self.train_batch_var = ctk.StringVar(value="")
        ctk.CTkLabel(progress_frame, textvariable=self.train_batch_var, font=("Segoe UI", 11),
                     fg_color=("gray90", "gray20"), corner_radius=4, padx=8, pady=2).grid(row=0, column=1, sticky="e")

        self.train_epoch_bar = ctk.CTkProgressBar(scroll, height=18, corner_radius=6,
                                                  fg_color=("gray80", "gray25"),
                                                  progress_color=("#2563eb", "#3b82f6"))
        self.train_epoch_bar.grid(row=row+7, column=0, sticky="ew", pady=2)
        self.train_epoch_label = ctk.CTkLabel(scroll, text="", font=("Segoe UI", 11))
        self.train_epoch_label.grid(row=row+7, column=0, sticky="ew", pady=(2, 0))

        self.train_batch_bar = ctk.CTkProgressBar(scroll, height=10, corner_radius=4,
                                                  fg_color=("gray80", "gray25"),
                                                  progress_color=("#10b981", "#34d399"))
        self.train_batch_bar.grid(row=row+8, column=0, sticky="ew", pady=2)
        self.train_batch_label = ctk.CTkLabel(scroll, text="", font=("Segoe UI", 10))
        self.train_batch_label.grid(row=row+8, column=0, sticky="ew", pady=(2, 4))

        self.train_loss_var = ctk.StringVar(value="")
        self.train_loss_label = ctk.CTkLabel(scroll, textvariable=self.train_loss_var,
                                             font=("Segoe UI", 12), justify="left",
                                             fg_color=("gray90", "gray20"), corner_radius=6, padx=10, pady=6)
        self.train_loss_label.grid(row=row+9, column=0, sticky="ew", pady=2)

        # ─── Action Buttons ───
        btn_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_frame.grid(row=row+10, column=0, sticky="w", pady=10)

        self.train_btn_dit = ctk.CTkButton(
            btn_frame, text=self._tr("btn_train_dit"), font=("Segoe UI", 12, "bold"),
            command=lambda: self._start_training("transformer"))
        self.train_btn_dit.grid(row=0, column=0, padx=(0, 8))

        self.train_btn_tf_diffusion = ctk.CTkButton(
            btn_frame, text=self._tr("btn_train_tf_diffusion"), font=("Segoe UI", 12, "bold"),
            fg_color="#c084fc", hover_color="#a855f7",
            command=lambda: self._start_tf_diffusion_training())
        self.train_btn_tf_diffusion.grid(row=0, column=1, padx=(0, 8))

        self.train_stop_btn = ctk.CTkButton(
            btn_frame, text=self._tr("btn_stop_save"), font=("Segoe UI", 12, "bold"),
            fg_color="#dc2626", hover_color="#b91c1c", state="disabled",
            command=self._stop_training)
        self.train_stop_btn.grid(row=0, column=2, padx=(8, 0))

        self.kaggle_export_btn = ctk.CTkButton(
            btn_frame, text=self._tr("btn_kaggle_export"), font=("Segoe UI", 12, "bold"),
            fg_color="#dc2626", hover_color="#b91c1c",
            command=self._export_kaggle_dialog)
        self.kaggle_export_btn.grid(row=0, column=3, padx=(8, 0))

        # Show default panel and TF-Diffusion button after full construction
        self.after(0, lambda: (
            self.train_btn_dit.grid_remove(),
            self.train_model_type_selector.set("TF-Diffusion"),
            self._on_train_type_change("TF-Diffusion"),
        ))

    # ─── TF-Diffusion Training Panel ───

    def _build_tf_diffusion_train_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color="transparent")
        panel.grid(row=0, column=0, sticky="ew")
        panel.grid_columnconfigure(1, weight=1)

        row = 0
        ctk.CTkLabel(panel, text=self._tr("tf_diffusion_settings"), font=("Segoe UI", 16, "bold"),
                     text_color="#c084fc").grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10))
        row += 1

        # Encoder selection
        ctk.CTkLabel(panel, text=self._tr("text_encoder"), font=("Segoe UI", 13, "bold")).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        self.tf_encoder_combo = ctk.CTkOptionMenu(
            panel, values=list(MODEL_NAMES), font=("Segoe UI", 10),
            command=self._on_tf_encoder_change, width=160)
        self.tf_encoder_combo.grid(row=row, column=1, sticky="w", padx=(0, 8), pady=4)
        self.tf_encoder_combo.set("Phi-3.5-mini")
        self.tf_encoder_status = ctk.CTkLabel(panel, text=self._tr("encoder_not_loaded"), font=("Segoe UI", 10),
                                              fg_color=("gray90", "gray20"), corner_radius=4, padx=6, pady=2)
        self.tf_encoder_status.grid(row=row, column=2, sticky="ew", pady=4)
        row += 1
        ctk.CTkButton(panel, text=self._tr("btn_load_encoder"), font=("Segoe UI", 11, "bold"),
                      fg_color="#2563eb", hover_color="#1d4ed8",
                      command=self._load_tf_encoder, height=28, width=140,
                     ).grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
        row += 1

        # ─── Hidden State Cache Section ───
        ctk.CTkLabel(panel, text=self._tr("hidden_states"), font=("Segoe UI", 13, "bold"),
                     text_color="#34d399").grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 4))
        row += 1
        self.tf_use_cached_hs_var = ctk.BooleanVar(value=False)
        self.tf_use_cached_hs_switch = ctk.CTkSwitch(
            panel, text=self._tr("use_cached_hs"),
            variable=self.tf_use_cached_hs_var, onvalue=True, offvalue=False,
            command=self._on_cached_hs_toggle,
        )
        self.tf_use_cached_hs_switch.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1

        # Cache selector: choose which encoder's cache to use
        cache_sel_frame = ctk.CTkFrame(panel, fg_color="transparent")
        cache_sel_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=2)
        cache_sel_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(cache_sel_frame, text=self._tr("select_cache"), font=("Segoe UI", 11)).grid(
            row=0, column=0, sticky="w", padx=(0, 6))
        self.tf_cache_selector = ctk.CTkOptionMenu(
            cache_sel_frame, values=[self._tr("no_cache")], font=("Segoe UI", 10),
            command=self._on_cache_selection_change, width=160)
        self.tf_cache_selector.grid(row=0, column=1, sticky="w")
        ctk.CTkButton(cache_sel_frame, text="Refresh", font=("Segoe UI", 10),
                      width=30, height=26, fg_color="#334155", hover_color="#475569",
                      command=self._refresh_cache_list).grid(row=0, column=2, padx=(4, 0))
        row += 1

        self.tf_cache_status_label = ctk.CTkLabel(panel, text=self._tr("no_cache_loaded"),
                                                   font=("Segoe UI", 10),
                                                   fg_color=("gray90", "gray20"),
                                                   corner_radius=4, padx=6, pady=2)
        self.tf_cache_status_label.grid(row=row, column=0, columnspan=3, sticky="ew", pady=2)
        row += 1
        cache_btn_frame = ctk.CTkFrame(panel, fg_color="transparent")
        cache_btn_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        ctk.CTkButton(cache_btn_frame, text=self._tr("btn_precompute"), font=("Segoe UI", 10, "bold"),
                      fg_color="#059669", hover_color="#047857",
                      command=self._precompute_hidden_states, height=26, width=120,
                     ).grid(row=0, column=0, padx=(0, 4))
        ctk.CTkButton(cache_btn_frame, text=self._tr("btn_check_cache"), font=("Segoe UI", 10),
                      fg_color="#334155", hover_color="#475569",
                      command=self._check_cache_status, height=26, width=110,
                     ).grid(row=0, column=1, padx=4)
        row += 1

        # UNet Presets
        ctk.CTkLabel(panel, text=self._tr("unet_preset"), font=("Segoe UI", 13, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(10, 4))
        row += 1
        self.tf_preset_var = ctk.StringVar(value="Tiny (0.5M)")
        preset_frame = ctk.CTkFrame(panel, fg_color="transparent")
        preset_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        for i, (label, channels, ch_mult, d_model, ca_heads) in enumerate(TF_DIFFUSION_PRESETS):
            ctk.CTkButton(preset_frame, text=label, font=("Segoe UI", 10), width=120, height=28,
                          command=lambda l=label, c=channels, cm=ch_mult, dm=d_model, h=ca_heads:
                          self._set_tf_diffusion_preset(l, c, cm, dm, h),
                         ).grid(row=i // 3, column=i % 3, padx=2, pady=1)
        row += 1

        # TF-Diffusion Diff steps
        ctk.CTkLabel(panel, text=self._tr("diff_steps"), font=("Segoe UI", 12)).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        self.tf_diff_steps_entry = ctk.CTkEntry(panel, width=80)
        self.tf_diff_steps_entry.grid(row=row, column=1, sticky="w", pady=4)
        self.tf_diff_steps_entry.insert(0, "50")
        row += 1

        # TF-Diffusion progress (Aug. Vielfalt und Vertikal-Bewegung werden
        # von den gemeinsamen Parametern unterhalb übernommen)
        self.tf_progress_bar = ctk.CTkProgressBar(panel, mode="determinate")
        self.tf_progress_bar.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(6, 2))
        self.tf_progress_label = ctk.CTkLabel(panel, text="", font=("Segoe UI", 10))
        self.tf_progress_label.grid(row=row+1, column=0, columnspan=3, sticky="ew")
        self.tf_loss_var = ctk.StringVar(value="")
        self.tf_loss_label = ctk.CTkLabel(panel, textvariable=self.tf_loss_var, font=("Segoe UI", 11),
                                          fg_color=("gray90", "gray20"), corner_radius=4, padx=8, pady=2)
        self.tf_loss_label.grid(row=row+2, column=0, columnspan=3, sticky="ew", pady=2)

        self._tf_unet_config = {
            "channels": 16, "channel_multipliers": (1, 2, 2),
            "d_model": 32, "cross_attn_heads": 2,
        }
        self.tf_panel = panel  # save reference
        # Initialize cache list after panel is built
        self.after(500, self._refresh_cache_list)

    # ─── Transformer Training Panel ───

    def _build_transformer_train_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color="transparent")
        panel.grid(row=0, column=0, sticky="ew")
        panel.grid_columnconfigure(1, weight=1)

        row = 0
        ctk.CTkLabel(panel, text=self._tr("transformer_settings"), font=("Segoe UI", 16, "bold"),
                     text_color="#60a5fa").grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10))
        row += 1

        # Modellgröße presets
        ctk.CTkLabel(panel, text=self._tr("model_size"), font=("Segoe UI", 13, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1
        preset_frame = ctk.CTkFrame(panel, fg_color="transparent")
        preset_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        for i, (label, val) in enumerate([("Tiny (1.7M)", 1.7), ("Small (3.9M)", 3.9),
                                          ("Medium (7.5M)", 7.5), ("Large (20M)", 20),
                                          ("XL (45M)", 45), ("XXL (117M)", 117)]):
            ctk.CTkButton(preset_frame, text=label, font=("Segoe UI", 10), width=110, height=28,
                          command=lambda v=val: self._set_model_size(v)).grid(row=i // 3, column=i % 3, padx=2, pady=1)
        row += 1

        self.size_slider_var = ctk.DoubleVar(value=3.9)
        self.size_slider = ctk.CTkSlider(panel, from_=0.5, to=150.0, number_of_steps=299,
                                         variable=self.size_slider_var, command=self._on_size_slider)
        self.size_slider.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(4, 2))
        row += 1

        self.params_info_var = ctk.StringVar(value=self._tr("target_arch", val=3.9, params=0, d_model=0, nhead=0, layers=0, ffn=0))
        self.params_info_label = ctk.CTkLabel(panel, textvariable=self.params_info_var, font=("Segoe UI", 11),
                                              justify="left", fg_color=("gray90", "gray20"),
                                              corner_radius=6, padx=10, pady=6)
        self.params_info_label.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(4, 2))
        row += 1

        ctk.CTkLabel(panel, text="Prompt input", font=("Segoe UI", 11, "bold")).grid(
            row=row, column=0, sticky="w", pady=(4, 2))
        self.dit_text_encoder_var = ctk.StringVar(value="Learned embeddings")
        self.dit_text_encoder_menu = ctk.CTkOptionMenu(
            panel, values=["Learned embeddings", *MODEL_NAMES],
            variable=self.dit_text_encoder_var, width=200,
            command=lambda _value: self._update_arch_from_advanced(),
        )
        self.dit_text_encoder_menu.grid(row=row, column=1, sticky="w", padx=4, pady=(4, 2))
        row += 1
        ctk.CTkLabel(panel, text="Learned by DiT or frozen pretrained text encoder",
                     font=("Segoe UI", 10), text_color=("gray50", "gray40")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 3))
        row += 1

        ctk.CTkLabel(panel, text=self._tr("attention_window"), font=("Segoe UI", 11)).grid(
            row=row, column=0, sticky="w", pady=(4, 2))
        self.attention_window_var = ctk.StringVar(value="8³")
        self.attention_window_menu = ctk.CTkOptionMenu(
            panel, values=["4³", "8³", "16³", "Global"],
            variable=self.attention_window_var, width=150,
            command=lambda _value: self._on_attention_window_changed(),
        )
        self.attention_window_menu.grid(row=row, column=1, sticky="w", padx=4, pady=(4, 2))
        row += 1

        self.show_advanced = False
        self.advanced_btn = ctk.CTkButton(panel, text=self._tr("btn_advanced"), font=("Segoe UI", 11),
                                          width=180, height=28, fg_color="#334155", hover_color="#475569",
                                          command=self._toggle_advanced)
        self.advanced_btn.grid(row=row, column=0, sticky="w", pady=(4, 2))
        row += 1

        self.advanced_frame = ctk.CTkFrame(panel, fg_color=("gray92", "gray15"), corner_radius=8)
        self.advanced_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(2, 6))
        self.advanced_frame.grid_remove()
        self.advanced_frame.grid_columnconfigure(1, weight=1)
        self.advanced_frame.grid_columnconfigure(3, weight=1)

        self.adv_vars = {}
        self.adv_entries = {}
        labels_adv = [
            (self._tr("adv_d_model"), "d_model", 64, 1024, 1, 192),
            (self._tr("adv_nhead"), "nhead", 2, 16, 1, 6),
            (self._tr("adv_layers"), "layers", 2, 24, 1, 5),
            (self._tr("adv_ff_ratio"), "ff_ratio", 2, 6, 1, 4),
        ]
        for r, (label, key, lo, hi, step, default) in enumerate(labels_adv):
            ctk.CTkLabel(self.advanced_frame, text=label, font=("Segoe UI", 11)).grid(
                row=r, column=0, sticky="w", padx=(8, 4), pady=3)
            var = ctk.StringVar(value=str(default))
            entry = ctk.CTkEntry(self.advanced_frame, width=80, textvariable=var)
            entry.grid(row=r, column=1, sticky="w", padx=4, pady=3)
            self.adv_vars[key] = var
            self.adv_entries[key] = entry
            var.trace_add("write", lambda *_: self._update_arch_from_advanced())
        ctk.CTkLabel(self.advanced_frame, text="Max. recurrent depth").grid(row=4, column=0, sticky="w", padx=8, pady=3)
        self.dit_depth_var = ctk.StringVar(value="4")
        ctk.CTkEntry(self.advanced_frame, width=80, textvariable=self.dit_depth_var).grid(row=4, column=1, sticky="w", padx=4, pady=3)
        self.moe_enabled_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.advanced_frame, text="Mixture of Experts", variable=self.moe_enabled_var).grid(row=5, column=0, sticky="w", padx=8, pady=3)
        self.moe_experts_var = ctk.StringVar(value="4")
        self.moe_active_var = ctk.StringVar(value="2")
        ctk.CTkLabel(self.advanced_frame, text="Experts total / active").grid(row=5, column=1, sticky="w", padx=4, pady=3)
        ctk.CTkEntry(self.advanced_frame, width=48, textvariable=self.moe_experts_var).grid(row=5, column=2, sticky="w", padx=2, pady=3)
        ctk.CTkEntry(self.advanced_frame, width=48, textvariable=self.moe_active_var).grid(row=5, column=3, sticky="w", padx=2, pady=3)
        ctk.CTkButton(self.advanced_frame, text=self._tr("btn_calculate"), font=("Segoe UI", 10), width=100, height=26,
                      command=self._update_arch_from_advanced).grid(row=6, column=0, padx=8, pady=4, sticky="w")

        self.transformer_panel = panel  # save reference

    # ─── Diffusion Training Panel ───

    def _build_diffusion_train_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color="transparent")
        panel.grid(row=0, column=0, sticky="ew")
        panel.grid_columnconfigure(1, weight=1)

        row = 0
        ctk.CTkLabel(panel, text=self._tr("diffusion_settings"), font=("Segoe UI", 16, "bold"),
                     text_color="#a78bfa").grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 10))
        row += 1

        ctk.CTkLabel(panel, text=self._tr("diffusion_fixed_arch"),
                     font=("Segoe UI", 11), text_color=("gray50", "gray40"), justify="left").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 8))
        row += 1

        ctk.CTkLabel(panel, text=self._tr("diff_steps"), font=("Segoe UI", 12)).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        self.diff_train_steps_entry = ctk.CTkEntry(panel, width=80)
        self.diff_train_steps_entry.grid(row=row, column=1, sticky="w", pady=4)
        self.diff_train_steps_entry.insert(0, "50")
        row += 1

        ctk.CTkLabel(panel, text=self._tr("diffusion_shared_params"),
                     font=("Segoe UI", 11), text_color=("gray50", "gray40"), justify="left").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.diffusion_panel = panel  # save reference

    # ─── Show/Hide panels based on model type ───

    def _on_train_type_change(self, value: str):
        # Hide all panels
        for panel_name in ['tf_panel', 'transformer_panel']:
            panel = getattr(self, panel_name, None)
            if panel:
                panel.grid_remove()
        # Hide all training buttons
        self.train_btn_tf_diffusion.grid_remove()
        self.train_btn_dit.grid_remove()
        # Show selected panel and matching button
        if value == "TF-Diffusion":
            if hasattr(self, 'tf_panel'):
                self.tf_panel.grid()
            self.train_btn_tf_diffusion.grid()
        elif value == "DiT (Recurrent)":
            if hasattr(self, 'transformer_panel'):
                self.transformer_panel.grid()
            self.train_btn_dit.grid()

    # ═══════════════════════════════════════════════════════════════
    # TRANSFORMER DIFFUSION - TRAINING CONTROLS
    # ═══════════════════════════════════════════════════════════════

    def _set_tf_diffusion_preset(self, label: str, channels: int, ch_mult: tuple,
                                  d_model: int, ca_heads: int):
        self.tf_preset_var.set(label)
        self._tf_unet_config = {
            "channels": channels,
            "channel_multipliers": ch_mult,
            "d_model": d_model,
            "cross_attn_heads": ca_heads,
        }
        self.status_var.set(self._tr("status_unet_preset", label=label))

    def _on_tf_encoder_change(self, value: str):
        self.tf_encoder_status.configure(text=self._tr("encoder_not_loaded_yet", name=value))

    def _load_tf_encoder(self):
        model_name = self.tf_encoder_combo.get()
        self.tf_encoder_status.configure(text=self._tr("status_encoder_loading", name=model_name))
        self.status_var.set(self._tr("status_encoder_loading", name=model_name))
        threading.Thread(target=self._load_tf_encoder_worker, args=(model_name,), daemon=True).start()

    def _load_tf_encoder_worker(self, model_name: str):
        try:
            # Ein expliziter Klick darf die VRAM/RAM-Frage erneut stellen.
            self._encoder_cpu_choice.pop(model_name, None)
            encoder = self._create_tf_encoder(model_name)
            self.tf_encoder = encoder
            location = "CPU" if encoder.device.type == "cpu" else "GPU"
            self.after(0, lambda: self.tf_encoder_status.configure(
                text=self._tr("encoder_loaded_info", name=model_name,
                              dim=encoder.hidden_dim, device=location)))
            self.after(0, lambda: self.status_var.set(self._tr("status_encoder_loaded", name=model_name)))
        except Exception as exc:
            err_msg = str(exc)
            lowered = err_msg.lower()
            if "out of memory" in lowered:
                # Klare Meldung statt nacktem CUDA-Fehler (Encoder zu gross).
                err_msg = self._tr("encoder_vram_error", name=model_name)
                release_cuda_cache()
            elif isinstance(exc, MemoryError) or "allocate memory" in lowered:
                # Auch der Ausweichpfad in den System-RAM ist gescheitert.
                err_msg = self._tr("encoder_ram_error", name=model_name)
            self.after(0, lambda m=err_msg: self.tf_encoder_status.configure(
                text=self._tr("cache_error", msg=m[:60])))
            self.after(0, lambda m=err_msg: self.status_var.set(self._tr("status_encoder_error", msg=m[:40])))

    # ─── Text-Encoder laden (VRAM, bei Bedarf nach Rueckfrage System-RAM) ───

    def _ask_yes_no_blocking(self, title: str, question: str) -> bool:
        """Ja/Nein-Dialog aus einem Worker-Thread heraus anzeigen.

        Tk laeuft im Main-Thread, das Modell-Laden im Worker - der Dialog wird
        deshalb per ``after`` an den Main-Thread gereicht und hier auf die
        Antwort gewartet.
        """
        from tkinter.messagebox import askyesno

        answer = {"value": False}
        answered = threading.Event()

        def show():
            try:
                answer["value"] = bool(askyesno(title, question))
            finally:
                answered.set()

        self.after(0, show)
        answered.wait(timeout=300)
        return bool(answer["value"])

    def _encoder_vram_fits(self, model_name: str) -> bool:
        """Schaetzung: passt der Encoder (fp16) in den freien VRAM?"""
        estimated = encoder_working_set_bytes(model_name, item_bytes=2)
        if not estimated or not (torch.cuda.is_available() and self.config.gpu_enabled):
            return True
        free_bytes, _total = torch.cuda.mem_get_info()
        return estimated <= free_bytes * ENCODER_VRAM_FRACTION

    def _request_encoder_cpu(self, model_name: str) -> bool:
        """Fragt (einmal pro Sitzung) nach dem Ausweichpfad in den System-RAM."""
        chosen = self._encoder_cpu_choice.get(model_name)
        if chosen is not None:
            return chosen
        estimated = encoder_working_set_bytes(model_name, item_bytes=2)
        ram_bytes = system_ram_bytes()
        free_bytes, _total = torch.cuda.mem_get_info()
        question = self._tr(
            "encoder_cpu_question", name=model_name,
            need=f"{estimated / GIB:.1f}" if estimated else "?",
            vram=format_gib(free_bytes),
            ram=f"{ram_bytes / GIB:.1f}" if ram_bytes else "?")
        if ram_bytes and estimated and estimated > ram_bytes:
            question += self._tr("encoder_cpu_ram_warning")
        wants_cpu = self._ask_yes_no_blocking(self._tr("encoder_cpu_title"), question)
        self._encoder_cpu_choice[model_name] = wants_cpu
        return wants_cpu

    def _encoder_cpu_dtype(self, model_name: str) -> torch.dtype:
        """dtype fuer den Encoder im RAM: fp32, wenn der Speicher sicher reicht.

        Phi-3.5-mini braucht in fp32 ~15 GB (mehr als 16-GB-Rechner haben), in
        fp16 ~7,6 GB; kleine Encoder (Flan-T5-small) laufen in fp32 schneller.
        """
        fp32_bytes = encoder_working_set_bytes(model_name, item_bytes=4)
        ram_bytes = system_ram_bytes()
        ram_enough = ram_bytes is None or fp32_bytes <= ram_bytes * ENCODER_CPU_RAM_SHARE
        if not fp32_bytes or (ram_enough and fp32_bytes <= ENCODER_CPU_FP32_LIMIT):
            return torch.float32
        return torch.float16

    def _build_tf_encoder(self, model_name: str,
                          device: torch.device) -> FrozenTransformerEncoder:
        """Encoder auf dem gewuenschten Geraet bauen (GPU und RAM-Fallback fp16)."""
        dtype = (torch.float16 if device.type == "cuda"
                 else self._encoder_cpu_dtype(model_name))
        return FrozenTransformerEncoder(model_name=model_name, device=device, dtype=dtype)

    def _create_tf_encoder(self, model_name: str) -> FrozenTransformerEncoder:
        """Encoder laden - passt er nicht in den VRAM, nach Rueckfrage in den RAM.

        Phi-3.5-mini braucht in fp16 rund 7,6 GB VRAM; auf einer 6-GB-Karte
        (GTX 1060) scheiterte der Aufbau bisher mit "Cuda out of memory" und die
        Generierung meldete danach "kein Modell". Passt der Encoder nicht in den
        VRAM, fragt die App einmal je Encoder, ob er stattdessen in den
        System-RAM geladen wird (fp16, damit 16-GB-Rechner reichen) - er rechnet
        dann auf der CPU: langsamer, aber benutzbar.

        Serialisiert: zwei Model-Loader-Threads duerfen nicht gleichzeitig einen
        mehrere GB grossen Encoder aufbauen.
        """
        with self._encoder_load_lock:
            return self._create_tf_encoder_locked(model_name)

    def _create_tf_encoder_locked(self, model_name: str) -> FrozenTransformerEncoder:
        gpu = torch.device("cuda" if torch.cuda.is_available() and self.config.gpu_enabled
                           else "cpu")
        if gpu.type == "cpu":
            return self._build_tf_encoder(model_name, gpu)

        if not self._encoder_vram_fits(model_name) and self._request_encoder_cpu(model_name):
            return self._load_tf_encoder_on_cpu(model_name)

        try:
            return self._build_tf_encoder(model_name, gpu)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            # Die Schaetzung war zu optimistisch - jetzt doch fragen.
            release_cuda_cache()
            gc.collect()
            if not self._request_encoder_cpu(model_name):
                hint = self._tr("encoder_vram_error", name=model_name)
                logger.error("%s (%s)", hint, exc)
                raise RuntimeError(f"{hint} ({exc})") from exc
            return self._load_tf_encoder_on_cpu(model_name)

    def _load_tf_encoder_on_cpu(self, model_name: str) -> FrozenTransformerEncoder:
        """Encoder im System-RAM laden (langsamer, aber ohne VRAM-Bedarf)."""
        self.after(0, lambda: self.status_var.set(
            self._tr("status_encoder_cpu", name=model_name)))
        try:
            return self._build_tf_encoder(model_name, torch.device("cpu"))
        except Exception:
            # Beim naechsten Mal wieder fragen (z.B. war der RAM voll).
            self._encoder_cpu_choice.pop(model_name, None)
            raise

    def _tf_diffusion_request(self) -> Dict[str, Any]:
        """TF-Diffusion-Einstellungen aus dem Training-Tab (nur Main-Thread)."""
        return {
            "use_cached": bool(self.tf_use_cached_hs_var.get()),
            "cache_name": self.tf_cache_selector.get(),
            "num_timesteps": self._read_int_entry(self.tf_diff_steps_entry, 50),
            "grid_size": self._get_selected_grid_size(),
            "augmentation_diversity": int(round(float(self.augmentation_diversity_var.get()))),
            "allow_vertical_movement": bool(self.setting_aug_vertical.get()),
        }

    def _start_tf_diffusion_training(self, continue_from: Optional[str] = None,
                                     overrides: Optional[Dict[str, Any]] = None) -> None:
        """TF-Diffusion-Training starten (neues Modell oder 'Weiter trainieren').

        ``continue_from`` nennt den Checkpoint, der fortgesetzt wird; der Worker
        laedt Encoder, Vokabular und Architektur dann daraus (siehe
        _prepare_tf_diffusion_continue).
        """
        if self.training_running:
            return
        params = self._collect_training_params()
        params.update(self._tf_diffusion_request())
        if overrides:
            params.update(overrides)
        use_cached = bool(params["use_cached"])
        cache_name = params.get("cache_name")
        if use_cached:
            if not cache_name or cache_name == self._tr("no_cache"):
                self.status_var.set(self._tr("status_no_cache"))
                return
            # Validate cache exists
            schem_files, txt_files = self._get_training_schem_files()
            result = validate_cache(cache_name, schem_files, txt_files)
            if not result["valid"]:
                self.status_var.set(self._tr("status_cache_invalid", msg=result['message']))
                return
        elif self.tf_encoder is None and continue_from is None:
            # Beim Fortsetzen laedt der Worker den Encoder aus dem Checkpoint
            # (inkl. VRAM/RAM-Fallback); sonst muss er hier geladen sein.
            self.status_var.set(self._tr("status_no_encoder"))
            return
        self.training_running = True
        params["continue_from"] = continue_from
        params["architecture"] = dict(arch)
        self.tf_progress_bar.set(0)
        self.train_btn_tf_diffusion.configure(state="disabled")
        self.train_stop_btn.configure(state="normal")
        if continue_from:
            self.status_var.set(self._tr("status_training_continue", name=continue_from))
        else:
            mode = self._tr("with_cached_hs") if use_cached else self._tr("with_encoder")
            self.status_var.set(self._tr("status_tf_training_started", mode=mode))
        threading.Thread(target=self._tf_diffusion_training_worker, args=(params,), daemon=True).start()

    def _refresh_cache_list(self):
        """Refresh the cache selector dropdown with available caches."""
        caches = list_caches()
        names = [c["encoder_name"] for c in caches] if caches else [self._tr("no_cache")]
        current = self.tf_cache_selector.get()
        self.tf_cache_selector.configure(values=names)
        if current in names:
            try:
                self.tf_cache_selector.set(current)
            except Exception:
                self.tf_cache_selector.set(names[0])
        else:
            self.tf_cache_selector.set(names[0])

    def _on_cache_selection_change(self, value: str):
        """Called when the user selects a different cache."""
        if value == self._tr("no_cache"):
            self.tf_cache_status_label.configure(text=self._tr("no_cache_selected"))
            return
        # Show cache info
        schem_files, txt_files = self._get_training_schem_files()
        result = validate_cache(value, schem_files, txt_files)
        self.tf_cache_status_label.configure(text=result["message"])

    def _tf_diffusion_training_worker(self, params: Dict[str, Any]):
        try:
            epochs = int(params["epochs"])
            batch_size = int(params["batch_size"])
            lr = float(params["lr"])
            num_timesteps = int(params["num_timesteps"])
            grid_size = tuple(params["grid_size"])
            aug_diversity = int(params["augmentation_diversity"])
            allow_vertical = bool(params["allow_vertical_movement"])
            air_weight = float(params["air_weight"])
            device = torch.device("cuda" if torch.cuda.is_available() and self.config.gpu_enabled else "cpu")
            config = dict(self._tf_unet_config)
            use_cached = bool(params["use_cached"])
            # Batch halten statt verkleinern: der Plan teilt ihn dann in
            # Mikro-Batches (siehe app/grad_accum.py).
            allow_accum = bool(params.get("grad_accum", True))
            cache_name = params.get("cache_name")
            continue_from = params.get("continue_from")

            from dataset import MultiSourceSchematicDataset
            from app.hidden_state_cache import CACHE_DIR, cache_key, load_hidden_states_raw

            # ── "Weiter trainieren": Architektur, Vokabular, Encoder und
            # Zielordner kommen aus dem Checkpoint ──
            continued = None
            if continue_from:
                continued = self._prepare_tf_diffusion_continue(
                    continue_from, device, use_cached, cache_name)
                grid_size = continued["grid_size"]
                num_timesteps = continued["num_timesteps"]
                config = {
                    "channels": continued["channels"],
                    "channel_multipliers": continued["channel_multipliers"],
                    "d_model": continued["d_model"],
                    "cross_attn_heads": continued["cross_attn_heads"],
                }
            vocab_kwargs: Dict[str, Any] = {}
            if continued:
                vocab_kwargs = {
                    "prompt_tokenizer": continued["prompt_tokenizer"],
                    "voxel_tokenizer": continued["voxel_tokenizer"],
                }

            if use_cached:
                # Cache was selected in the main thread - do not touch Tk here
                if not cache_name:
                    raise RuntimeError("Kein Hidden-State-Cache ausgewaehlt")
                encoder_name = cache_name
                key = cache_key(encoder_name)
                cache_dir = CACHE_DIR / key

                # Load cache metadata to get encoder info
                cache_data = load_hidden_states_raw(cache_dir)
                cache_meta = cache_data["metadata"]
                hs = cache_data["hidden_states"]  # flat [total_tokens, hidden_dim]
                # hidden_dim comes from the metadata - the flat cache has no
                # per-prompt sequence axis anymore (rows are stored unpadded).
                context_dim = int(cache_meta.get("hidden_dim") or hs.shape[-1])
                if continued and context_dim != continued["context_dim"]:
                    # Der Cache muss zum Encoder des Checkpoints passen, sonst
                    # wuerde das Modell mit fremden Textmerkmalen weitertrainieren.
                    raise RuntimeError(self._tr("tm_cache_dim_mismatch",
                                                dim=context_dim, need=continued["context_dim"]))

                # Build dataset with cache
                dataset = MultiSourceSchematicDataset.with_cache(
                    self.data_dirs, cache_dir, target_size=grid_size, max_voxels=400_000,
                    augmentation_diversity=aug_diversity,
                    allow_vertical_movement=allow_vertical,
                    air_weight_factor=air_weight,
                    progress_callback=self._dataset_progress_callback(),
                    **vocab_kwargs,
                )
                self._report_dataset_ready(dataset)

                # ── VRAM-Plan VOR dem Modellaufbau ──
                # Dieser Pfad ist zuvor mit "Tried to allocate 34.55 GiB"
                # abgebrochen: Grid 32^3 * 17.690 Blockzustaende * Batch 16
                # ergibt einen Logits-Tensor von 34,55 GiB. Der Plan reduziert
                # die Batch-Groesse und liefert die Chunk-Groesse, mit der die
                # Logits stueckweise berechnet werden.
                lengths = cache_data.get("lengths")
                if lengths is not None and int(lengths.numel()) > 0:
                    try:
                        quantile = torch.quantile(lengths.to(torch.float32), 0.95)
                        context_len = max(8, int(quantile))
                    except RuntimeError:
                        context_len = min(64, int(cache_meta.get("max_seq_len", 64) or 64))
                else:
                    context_len = min(64, int(cache_meta.get("max_seq_len", 64) or 64))
                plan = plan_diffusion_training(
                    grid_size, batch_size, len(dataset.voxel_tokenizer.id_to_block),
                    device=device,
                    channels=config["channels"],
                    channel_multipliers=tuple(config["channel_multipliers"]),
                    d_model=config["d_model"],
                    cross_attn_heads=config["cross_attn_heads"],
                    context_dim=context_dim,
                    context_len=context_len,
                    allow_accumulation=allow_accum,
                )
                batch_size, chunk_voxels, micro_batch = self._apply_training_plan(plan, batch_size)

                from torch.utils.data import DataLoader
                # collate_context_batch pads the unpadded cache rows per batch
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                    collate_fn=collate_context_batch)
                total_batches = len(loader)

                model = self._build_tf_diffusion_model(
                    config, grid_size, num_timesteps, context_dim,
                    len(dataset.voxel_tokenizer.id_to_block), continued, device)

                if continued:
                    encoder_config = continued["encoder_config"]
                    out_dir = RUNS_DIR / continue_from
                else:
                    # Build encoder_config from cache metadata
                    encoder_config = {
                        "display_name": cache_meta.get("encoder_name", encoder_name),
                        "hf_id": cache_meta.get("encoder_hf_id", ""),
                        "hidden_dim": context_dim,
                    }

                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    safe_enc_name = encoder_name.replace("/", "_").replace(" ", "_")
                    out_dir = RUNS_DIR / f"tf_diffusion_cached_{safe_enc_name}_{timestamp}"

            else:
                # ── Use live encoder ──
                if continued:
                    # _prepare_tf_diffusion_continue hat den Encoder des
                    # Checkpoints geladen (VRAM oder System-RAM).
                    context_dim = continued["context_dim"]
                else:
                    if self.tf_encoder is None:
                        raise RuntimeError("Encoder not loaded")
                    context_dim = self.tf_encoder.hidden_dim

                dataset = MultiSourceSchematicDataset(
                    self.data_dirs, target_size=grid_size, max_voxels=400_000,
                    augmentation_diversity=aug_diversity,
                    allow_vertical_movement=allow_vertical,
                    air_weight_factor=air_weight,
                    progress_callback=self._dataset_progress_callback(),
                    **vocab_kwargs,
                )
                self._report_dataset_ready(dataset)

                # VRAM-Plan (siehe gecachter Zweig). Die Textlaenge wird hier
                # geschaetzt: der Encoder padet intern auf den laengsten Prompt
                # des Batches; die Cache-Statistik (Median 14, Maximum 512
                # Tokens) zeigt, dass ein Batch von 16 selten ueber 64 Tokens
                # kommt. Der Posten geht nur in die Abschaetzung ein.
                plan = plan_diffusion_training(
                    grid_size, batch_size, len(dataset.voxel_tokenizer.id_to_block),
                    device=device,
                    channels=config["channels"],
                    channel_multipliers=tuple(config["channel_multipliers"]),
                    d_model=config["d_model"],
                    cross_attn_heads=config["cross_attn_heads"],
                    context_dim=context_dim,
                    context_len=min(128, int(getattr(self.tf_encoder, "max_length", 128) or 128)),
                    allow_accumulation=allow_accum,
                )
                batch_size, chunk_voxels, micro_batch = self._apply_training_plan(plan, batch_size)

                from torch.utils.data import DataLoader
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                total_batches = len(loader)

                model = self._build_tf_diffusion_model(
                    config, grid_size, num_timesteps, context_dim,
                    len(dataset.voxel_tokenizer.id_to_block), continued, device)

                if continued:
                    encoder_config = continued["encoder_config"]
                    out_dir = RUNS_DIR / continue_from
                else:
                    encoder_config = self.tf_encoder.get_config()
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    encoder_name = self.tf_encoder.display_name.replace("/", "_").replace(" ", "_")
                    out_dir = RUNS_DIR / f"tf_diffusion_{encoder_name}_{timestamp}"

            # Fortsetzen schreibt in denselben Ordner; die Epochen werden auf
            # die im Checkpoint gespeicherte Zahl addiert.
            start_epoch = int(continued["start_epoch"]) if continued else 0

            out_dir.mkdir(parents=True, exist_ok=True)
            dataset.prompt_tokenizer.save(out_dir / "prompt_vocab.json")
            dataset.voxel_tokenizer.save(out_dir / "block_vocab.json")
            (out_dir / "encoder_config.json").write_text(
                json.dumps(encoder_config, indent=2), encoding="utf-8")

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            total_start = time.time()
            noise_block_prob = getattr(self.config, 'noise_block_prob', 0.20)

            for epoch in range(1, epochs + 1):
                if not self.training_running:
                    break
                model.train()
                total_loss = 0.0
                batch_num = 0
                for batch in loader:
                    if not self.training_running:
                        break
                    batch_num += 1

                    # Gradient-Akkumulation: der DataLoader liefert den ganzen
                    # (effektiven) Batch, gerechnet wird in Mikro-Batches, die in
                    # den VRAM passen. Ohne Akkumulation ist das genau ein
                    # Durchlauf - altes Verhalten, nur mit Gewicht 1.0. Als Anteil
                    # zaehlt die Gewichtsmasse (Luft-Gewicht 75!): nur so sind die
                    # Gradienten exakt die des grossen Batches.
                    accumulating = micro_batch < batch_size
                    masses = (per_sample_weight_mass(batch["voxel_ids"], batch)
                              if accumulating else None)
                    if accumulating:
                        optimizer.zero_grad(set_to_none=True)
                    batch_loss = 0.0
                    for micro, scale in iter_micro_batches(batch, micro_batch,
                                                            sample_mass=masses):
                        if use_cached:
                            # Use cached training step (no encoder needed)
                            step_loss = train_transformer_diffusion_step_cached(
                                model, micro, optimizer, device,
                                noise_block_prob=noise_block_prob,
                                chunk_voxels=chunk_voxels,
                                accumulate=accumulating, grad_scale=scale)
                        else:
                            # Use live encoder
                            step_loss = train_transformer_diffusion_step(
                                model, micro, self.tf_encoder, optimizer, device,
                                noise_block_prob=noise_block_prob,
                                chunk_voxels=chunk_voxels,
                                accumulate=accumulating, grad_scale=scale)
                        batch_loss += step_loss * scale
                    if accumulating:
                        # Genau ein Schritt pro Batch (nicht pro Mikro-Batch).
                        apply_optimizer_step(model, optimizer)
                    total_loss += batch_loss

                    if batch_num % max(1, total_batches // 10) == 0 or batch_num == total_batches:
                        self.after(0, lambda bp=batch_num/total_batches, bn=batch_num, tb=total_batches: (
                            self.tf_progress_bar.set(bp),
                            self.tf_progress_label.configure(text=self._tr("batch_progress", bn=bn, tb=tb)),
                        ))
                avg_loss = total_loss / max(total_batches, 1)
                elapsed = time.time() - total_start
                done_epoch = start_epoch + epoch
                total_epoch = start_epoch + epochs
                self.after(0, lambda e=done_epoch, tot=total_epoch, l=avg_loss,
                                  p=done_epoch/total_epoch, el=elapsed: (
                    self.tf_progress_bar.set(p),
                    self.tf_progress_label.configure(text=self._tr("tf_epoch_progress", e=e, epochs=tot, loss=f"{l:.4f}")),
                    self.tf_loss_var.set(self._tr("loss_time", loss=f"{l:.4f}", time=f"{el:.0f}")),
                    self.train_epoch_bar.set(p),
                    self.train_epoch_label.configure(text=self._tr("tf_epoch_progress", e=e, epochs=tot, loss=f"{l:.4f}")),
                    self.train_loss_var.set(self._tr("tf_loss_time", loss=f"{l:.4f}", time=f"{el:.0f}")),
                ))
                torch.save({"model_state": model.state_dict(), "grid_size": grid_size,
                            "block_vocab_size": model.num_blocks, "num_blocks": model.num_blocks,
                            "d_model": model.d_model, "channels": model.channels,
                            "channel_multipliers": [int(m) for m in model.channel_multipliers],
                            "num_timesteps": model.num_timesteps, "context_dim": model.context_dim,
                            "cross_attn_heads": model.cross_attn_heads,
                            "context_proj_dim": model.effective_context_dim,
                            "encoder_config": encoder_config,
                            "augmentation_diversity": aug_diversity,
                            "allow_vertical_movement": allow_vertical,
                            "epoch": done_epoch, "loss": avg_loss,
                           }, out_dir / "model.pt")
            self.after(0, lambda: self.status_var.set(self._tr("status_tf_diffusion_done")))
            self.after(0, self._discover_models)
            self.after(0, self._refresh_models_tab)
            if continue_from:
                # Der fortgesetzte Checkpoint liegt im selben Ordner: neu laden,
                # damit die Generierung direkt die trainierten Gewichte nutzt.
                self.after(0, self._load_models_async)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.after(0, lambda msg=str(e)[:100]: self.tf_loss_var.set(self._tr("tf_error", msg=msg)))
        finally:
            self.training_running = False
            self.after(0, lambda: self.train_btn_tf_diffusion.configure(state="normal"))
            self.after(0, lambda: self.train_stop_btn.configure(state="disabled"))
            # VRAM des Trainings zurueckgeben: der Caching-Allocator behaelt die
            # Logits-/Gradienten-Bloecke sonst im eigenen Pool, mem_get_info()
            # meldet sie als belegt und die Generierung plant zu klein bzw.
            # scheitert mit "Cuda out of memory". Der verzoegerte Aufruf laeuft,
            # nachdem dieser Frame (und damit die letzten Batch-Tensoren)
            # abgebaut ist.
            release_cuda_cache()
            self.after(200, release_cuda_cache)

    # ═══════════════════════════════════════════════════════════════
    # TF-DIFFUSION: WEITER TRAINIEREN (CHECKPOINT FORTSETZEN)
    # ═══════════════════════════════════════════════════════════════

    def _prepare_tf_diffusion_continue(self, name: str, device, use_cached: bool,
                                       cache_name: Optional[str]) -> Dict[str, Any]:
        """Zustand fuer 'Weiter trainieren' eines TF-Diffusion-Modells sammeln.

        Architektur, Vokabular, Encoder-Konfiguration und Zielordner kommen
        komplett aus dem Checkpoint - nur so passen die Gewichte zum weiteren
        Training. Ohne Cache wird der Encoder hier im Worker geladen
        (VRAM/RAM-Fallback wie beim Laden ueber die GUI), mit Cache wird
        geprueft, dass er zum Checkpoint passt.
        """
        entry = self.model_registry.get(name)
        if entry is None or not entry.checkpoint_path.exists():
            self._discover_models()
            entry = self.model_registry.get(name)
        if entry is None or not entry.checkpoint_path.exists():
            raise RuntimeError(self._tr("tm_missing_checkpoint", name=name))
        pt_path = entry.path / "prompt_vocab.json"
        vt_path = entry.path / "block_vocab.json"
        if not (pt_path.exists() and vt_path.exists()):
            # Ohne das Vokabular des Checkpoints wuerden die Block-IDs nicht
            # mehr passen - dann lieber sauber abbrechen.
            raise RuntimeError(self._tr("tm_missing_vocab", name=name))

        ckpt = torch.load(entry.checkpoint_path, map_location="cpu")
        encoder_config = ckpt.get("encoder_config", {}) or {}
        context_dim = int(ckpt.get("context_dim", DEFAULT_ENCODER_HIDDEN_DIM))
        encoder_hint = str(encoder_config.get("display_name", "") or "")
        if use_cached:
            if not cache_name:
                raise RuntimeError("Kein Hidden-State-Cache ausgewaehlt")
            from app.hidden_state_cache import CACHE_DIR, cache_key, load_hidden_states_raw
            cache_meta = load_hidden_states_raw(CACHE_DIR / cache_key(cache_name))["metadata"]
            cache_dim = int(cache_meta.get("hidden_dim") or 0)
            if cache_dim and cache_dim != context_dim:
                raise RuntimeError(self._tr("tm_cache_dim_mismatch",
                                            dim=cache_dim, need=context_dim))
            cache_encoder = str(cache_meta.get("encoder_name", "") or "")
            if encoder_hint and cache_encoder and cache_encoder != encoder_hint:
                raise RuntimeError(self._tr("tm_cache_encoder_mismatch",
                                            cache=cache_encoder, encoder=encoder_hint))
        elif encoder_hint:
            encoder = self._ensure_encoder_for_continue(encoder_hint)
            if int(encoder.hidden_dim) != context_dim:
                raise RuntimeError(self._tr("tm_encoder_dim_mismatch", name=encoder_hint,
                                            dim=encoder.hidden_dim, need=context_dim))

        loaded_d_model = int(ckpt.get("d_model", 64) or 64)
        return {
            "state_dict": ckpt["model_state"],
            "grid_size": tuple(ckpt.get("grid_size", entry.grid_size)),
            "num_timesteps": int(ckpt.get("num_timesteps", 50) or 50),
            "channels": int(ckpt.get("channels", 32) or 32),
            "channel_multipliers": tuple(ckpt.get("channel_multipliers", (1, 2, 2))),
            "d_model": loaded_d_model,
            "cross_attn_heads": int(ckpt.get("cross_attn_heads", 4) or 4),
            "context_dim": context_dim,
            "context_proj_dim": int(ckpt.get("context_proj_dim", loaded_d_model * 2)),
            "num_blocks": int(ckpt.get("num_blocks", ckpt.get("block_vocab_size", 0))),
            "encoder_config": encoder_config,
            "prompt_tokenizer": PromptTokenizer.load(pt_path),
            "voxel_tokenizer": VoxelTokenizer.load(vt_path),
            "start_epoch": int(ckpt.get("epoch", 0) or 0),
        }

    def _ensure_encoder_for_continue(self, encoder_name: str) -> FrozenTransformerEncoder:
        """Encoder fuer 'Weiter trainieren' laden (VRAM, sonst System-RAM).

        Laeuft im Worker-Thread: der GUI-Loader ist zum Trainingsstart nicht
        zwingend fertig, der Encoder muss aber zum Checkpoint passen. Der
        Ausweichpfad in den RAM bleibt derselbe wie beim manuellen Laden.
        """
        if self.tf_encoder is not None and self.tf_encoder.display_name == encoder_name:
            return self.tf_encoder
        encoder = self._create_tf_encoder(encoder_name)
        self.tf_encoder = encoder
        location = "CPU" if encoder.device.type == "cpu" else "GPU"
        hidden_dim = int(encoder.hidden_dim)

        def _show_in_ui() -> None:
            try:
                self.tf_encoder_combo.set(encoder_name)
            except Exception:  # Encoder nicht in der Liste: Anzeige belassen
                pass
            self.tf_encoder_status.configure(
                text=self._tr("encoder_loaded_info", name=encoder_name,
                              dim=hidden_dim, device=location))

        self.after(0, _show_in_ui)
        return encoder

    def _build_tf_diffusion_model(self, config: Dict[str, Any], grid_size: tuple,
                                  num_timesteps: int, context_dim: int, num_blocks: int,
                                  continued: Optional[Dict[str, Any]], device):
        """TF-Diffusion-Modell bauen - neu oder aus dem Checkpoint fortgesetzt.

        ``continued`` ist der Zustand aus _prepare_tf_diffusion_continue; ist er
        gesetzt, kommen Architektur und Gewichte aus dem Checkpoint (statt aus
        dem UNet-Preset der GUI) und die Gewichte werden direkt geladen.
        """
        model = TransformerDiffusionModel(
            num_blocks=int(continued["num_blocks"]) if continued else int(num_blocks),
            grid_size=tuple(grid_size),
            d_model=int(config["d_model"]),
            channels=int(config["channels"]),
            channel_multipliers=tuple(config["channel_multipliers"]),
            num_timesteps=int(num_timesteps),
            context_dim=int(context_dim),
            cross_attn_heads=int(config["cross_attn_heads"]),
            context_proj_dim=(int(continued["context_proj_dim"]) if continued
                              else int(config["d_model"]) * 2),
        ).to(device)
        if continued:
            # Hier wird aus "neu bauen" ein "weiter trainieren": der Checkpoint
            # wird strikt geladen (Architektur muss exakt passen).
            model.load_state_dict(continued.pop("state_dict"), strict=True)
        return model

    # ═══════════════════════════════════════════════════════════════
    # PROJECTS TAB
    # ═══════════════════════════════════════════════════════════════

    def _build_projects_tab(self):
        tab = self.tab_projects
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=0, minsize=300)
        tab.grid_columnconfigure(1, weight=1)
        left = ctk.CTkFrame(tab, corner_radius=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(left, text=self._tr("saved_projects"), font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w", pady=10, padx=10)
        self.project_listbox = ctk.CTkTextbox(left, font=("Segoe UI", 12), state="disabled")
        self.project_listbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        btn_frame = ctk.CTkFrame(left, fg_color="transparent")
        btn_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10), padx=10)
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)
        btn_frame.grid_columnconfigure(2, weight=1)
        ctk.CTkButton(btn_frame, text=self._tr("btn_reload"), command=self._refresh_projects).grid(row=0, column=0, padx=2)
        ctk.CTkButton(btn_frame, text=self._tr("btn_load_project"), command=self._load_selected_project).grid(row=0, column=1, padx=2)
        ctk.CTkButton(btn_frame, text=self._tr("btn_delete_project"), fg_color="#dc2626", hover_color="#b91c1c",
                      command=self._delete_selected_project).grid(row=0, column=2, padx=2)
        right = ctk.CTkFrame(tab, corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(3, weight=1)
        right.grid_columnconfigure(0, weight=1)
        self.project_info = ctk.CTkLabel(right, text=self._tr("select_project"), font=("Segoe UI", 14), justify="left")
        self.project_info.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.project_preview = ctk.CTkLabel(right, text="")
        self.project_preview.grid(row=1, column=0, padx=10, pady=4)
        ctk.CTkButton(right, text=self._tr("btn_export_project"), font=("Segoe UI", 13),
                      command=self._export_selected_project).grid(row=2, column=0, pady=10, padx=10)

    # ═══════════════════════════════════════════════════════════════
    # SETTINGS TAB (only general settings)
    # ═══════════════════════════════════════════════════════════════

    def _build_settings_tab(self):
        tab = self.tab_settings
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        scroll = ctk.CTkScrollableFrame(tab, corner_radius=10)
        scroll.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        scroll.grid_columnconfigure(0, weight=1)

        row = 0
        ctk.CTkLabel(scroll, text=self._tr("general"), font=("Segoe UI", 18, "bold")).grid(row=row, column=0, sticky="w", pady=(0, 10))
        row += 1

        ctk.CTkLabel(scroll, text=self._tr("use_gpu")).grid(row=row, column=0, sticky="w", pady=2)
        row += 1
        self.setting_gpu = ctk.CTkSwitch(scroll, text="CUDA GPU", onvalue=True, offvalue=False)
        self.setting_gpu.grid(row=row, column=0, sticky="w", pady=2)
        self.setting_gpu.select() if self.config.gpu_enabled else self.setting_gpu.deselect()
        row += 1

        ctk.CTkLabel(scroll, text=self._tr("language")).grid(row=row, column=0, sticky="w", pady=2)
        row += 1
        self.setting_lang = ctk.CTkOptionMenu(scroll, values=["Deutsch", "English"])
        self.setting_lang.grid(row=row, column=0, sticky="w", pady=2)
        self.setting_lang.set("Deutsch" if self.config.language == "de" else "English")
        row += 1

        ctk.CTkButton(scroll, text=self._tr("btn_save_settings"), font=("Segoe UI", 13, "bold"),
                      fg_color="#059669", hover_color="#047857",
                      command=self._save_settings).grid(row=row, column=0, sticky="w", pady=20)

    # ═══════════════════════════════════════════════════════════════
    # ABOUT TAB
    # ═══════════════════════════════════════════════════════════════

    def _build_about_tab(self):
        tab = self.tab_about
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        frame = ctk.CTkFrame(tab, corner_radius=10)
        frame.grid(row=0, column=0, sticky="nsew", padx=40, pady=40)
        frame.grid_columnconfigure(0, weight=1)
        grid_sizes = " · ".join(GRID_SIZE_OPTIONS)
        info = [
            self._tr("about_title", name=APP_NAME, version=VERSION), "",
            self._tr("about_ai_generator"), "",
            self._tr("about_features"),
            self._tr("about_tf_diffusion"),
            self._tr("about_frozen_encoder"),
            self._tr("about_cross_attn"),
            self._tr("about_3d_unet"),
            self._tr("about_transformer"),
            self._tr("about_3d_preview"),
            self._tr("about_text_to_struct"),
            self._tr("about_model_manager"),
            self._tr("about_project_mgmt"),
            self._tr("about_gpu_training"), "",
            self._tr("about_training_data"),
            self._tr("about_grid_sizes", sizes=grid_sizes), "",
            self._tr("about_created_with"),
        ]
        ctk.CTkLabel(frame, text="\n".join(info), font=("Segoe UI", 13), justify="left").grid(row=0, column=0, sticky="w", padx=20, pady=20)

    def _setup_tab_icons(self):
        pass

    # ═══════════════════════════════════════════════════════════════
    # MODEL COMBO & DISCOVERY
    # ═══════════════════════════════════════════════════════════════

    def _refresh_model_combo(self):
        self.model_registry.discover()
        models = self.model_registry.get_by_type(self.model_type)
        if self.model_type_selector.get() == "DiT":
            models = [m for m in models if m.recurrent_depth > 1 or m.moe_experts > 0]
        names = [m.name for m in models] if models else [self._tr("none")]
        current = ""
        if self.model_type == "transformer" and self.current_transformer_name:
            current = self.current_transformer_name
        elif self.model_type == "diffusion" and self.current_diffusion_name:
            current = self.current_diffusion_name
        elif self.model_type == "transformer_diffusion" and self.current_tf_diffusion_name:
            current = self.current_tf_diffusion_name
        if not current or current not in names:
            default = (self.model_registry.default_transformer if self.model_type == "transformer"
                       else self.model_registry.default_diffusion if self.model_type == "diffusion"
                       else self.model_registry.default_tf_diffusion)
            if default and default in names:
                current = default
        if self.model_type == "transformer" and self.model_type_selector.get() == "DiT":
            current = current if current in names else (names[0] if names and names[0] != self._tr("none") else "")
            self.current_transformer_name = current or None
        self.model_version_combo.configure(values=names if names else [self._tr("none")])
        if current:
            try:
                self.model_version_combo.set(current)
            except Exception:
                self.model_version_combo.set(names[0] if names else self._tr("none"))
        else:
            self.model_version_combo.set(names[0] if names else self._tr("none"))

    def _on_model_type_change(self, value: str):
        if value == "TF-Diffusion":
            self.model_type = "transformer_diffusion"
            self.generation_dit_mode = False
        else:
            self.model_type = "transformer"
            self.generation_dit_mode = True
        self._refresh_model_combo()
        self._update_diffusion_visibility()
        self._load_models_async()

    def _on_model_version_change(self, value: str):
        if value == self._tr("none"):
            return
        if self.model_type == "transformer":
            self.current_transformer_name = value
        elif self.model_type == "diffusion":
            self.current_diffusion_name = value
        else:
            self.current_tf_diffusion_name = value
        self._load_models_async()

    # ═══════════════════════════════════════════════════════════════
    # MODEL LOADING
    # ═══════════════════════════════════════════════════════════════

    def _load_models_async(self):
        self.model_status.configure(text=self._tr("status_loading_models"))
        threading.Thread(target=self._load_models, daemon=True).start()

    def _guard_model_load(self, label: str, errors: List[str], loader) -> bool:
        """Run one model loader; a single failure must not break the others."""
        try:
            return bool(loader())
        except Exception as exc:  # noqa: BLE001 - gui darf an einem kaputten Modell nicht sterben
            logger.exception("Modell '%s' konnte nicht geladen werden", label)
            errors.append(f"{label}: {exc}")
            # Ein abgebrochener Aufbau (z.B. ein Encoder, der nicht in den VRAM
            # passt) laesst sonst mehrere GB im Caching-Allocator zurueck.
            release_cuda_cache()
            return False

    def _load_tf_diffusion_entry(self, tf_diff_name: str, device) -> bool:
        """Load a TF-Diffusion checkpoint (raises on failure)."""
        entry = self.model_registry.get(tf_diff_name)
        if entry is None or not entry.checkpoint_path.exists():
            return False
        ckpt = torch.load(entry.checkpoint_path, map_location="cpu")
        if self.prompt_tokenizer is None:
            self.prompt_tokenizer = PromptTokenizer.load(entry.path / "prompt_vocab.json")
        if self.voxel_tokenizer is None:
            self.voxel_tokenizer = VoxelTokenizer.load(entry.path / "block_vocab.json")
        encoder_config = ckpt.get("encoder_config", {})
        encoder_name = encoder_config.get("display_name", "Phi-3.5-mini")
        if self.tf_encoder is None or self.tf_encoder.display_name != encoder_name:
            # Laedt auf die GPU - oder nach Rueckfrage in den System-RAM, wenn
            # der Encoder nicht in den VRAM passt (siehe _create_tf_encoder).
            self.tf_encoder = self._create_tf_encoder(encoder_name)
        # Default context_proj_dim to d_model*2 to match training behavior
        loaded_d_model = ckpt.get("d_model", 64)
        model = TransformerDiffusionModel(
            num_blocks=ckpt.get("num_blocks", ckpt.get("block_vocab_size", 0)),
            grid_size=tuple(ckpt.get("grid_size", (16, 16, 16))),
            d_model=loaded_d_model, channels=ckpt.get("channels", 32),
            channel_multipliers=tuple(ckpt.get("channel_multipliers", (1, 2, 2))),
            num_timesteps=ckpt.get("num_timesteps", 50),
            context_dim=ckpt.get("context_dim", 768),
            cross_attn_heads=ckpt.get("cross_attn_heads", 4),
            context_proj_dim=ckpt.get("context_proj_dim", loaded_d_model * 2),
        ).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()
        self.tf_diffusion_model = model
        self.current_tf_diffusion_name = tf_diff_name
        return True


    def _load_transformer_entry(self, tf_name: str, device) -> bool:
        """Load a SharedWeightVoxelTransformer checkpoint (raises on failure)."""
        entry = self.model_registry.get(tf_name)
        if entry is None or not entry.checkpoint_path.exists():
            return False
        ckpt = torch.load(entry.checkpoint_path, map_location="cpu")
        prompt_tokenizer = PromptTokenizer.load(entry.path / "prompt_vocab.json")
        voxel_tokenizer = VoxelTokenizer.load(entry.path / "block_vocab.json")
        model = SharedWeightVoxelTransformer(
            text_vocab_size=ckpt["text_vocab_size"],
            block_vocab_size=ckpt["block_vocab_size"],
            grid_size=tuple(ckpt["grid_size"]), d_model=ckpt["d_model"],
            nhead=ckpt.get("nhead", 6), num_layers=ckpt["layers"],
            dim_feedforward=ckpt.get("dim_feedforward", 768), dropout=0.0,
            recurrent_depth=ckpt.get("recurrent_depth", 1),
            moe_experts=ckpt.get("moe_experts", 0), moe_active=ckpt.get("moe_active", 1),
            text_input_dim=int(ckpt.get("text_input_dim", 0) or 0),
            text_encoder_name=ckpt.get("text_encoder_name", ""),
            window_size=ckpt.get("window_size", 0),
            tread_ratio=ckpt.get("tread_ratio", 0.25),
            ssa_topk=ckpt.get("ssa_topk", 16),
            voxel_mask_ratio=ckpt.get("voxel_mask_ratio", 0.65),
        ).to(device)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        if unexpected or any(not key.startswith("recurrence_router.") for key in missing):
            raise RuntimeError(
                f"Checkpoint passt nicht zur DiT-Architektur (fehlend={missing}, unerwartet={unexpected})"
            )
        model.eval()
        self.prompt_tokenizer = prompt_tokenizer
        self.voxel_tokenizer = voxel_tokenizer
        self.transformer_model = model
        self.current_transformer_name = tf_name
        return True

    def _load_diffusion_entry(self, df_name: str, device) -> bool:
        """Load a VoxelDiffusionModel checkpoint (raises on failure)."""
        entry = self.model_registry.get(df_name)
        if entry is None or not entry.checkpoint_path.exists():
            return False
        ckpt = torch.load(entry.checkpoint_path, map_location="cpu")
        if self.prompt_tokenizer is None:
            self.prompt_tokenizer = PromptTokenizer.load(entry.path / "prompt_vocab.json")
        if self.voxel_tokenizer is None:
            self.voxel_tokenizer = VoxelTokenizer.load(entry.path / "block_vocab.json")
        ms = ckpt["model_state"]
        actual_num_blocks = ms["block_embed.embed.weight"].shape[0]
        actual_text_vocab = ms["text_cond.embed.weight"].shape[0]
        actual_d_text = ms["text_cond.embed.weight"].shape[1]
        actual_d_model = ms["time_embed.0.weight"].shape[1]
        actual_channels = ms["block_embed.embed.weight"].shape[1]
        actual_ch_mult = []
        encoder_idx = 0
        while f"encoder.{encoder_idx}.res.conv1.weight" in ms:
            out_ch = ms[f"encoder.{encoder_idx}.res.conv1.weight"].shape[0]
            actual_ch_mult.append(out_ch // actual_channels)
            encoder_idx += 1
        if not actual_ch_mult:
            actual_ch_mult = ckpt.get("channel_multipliers", (1, 2, 2))
        model = VoxelDiffusionModel(
            num_blocks=actual_num_blocks, text_vocab_size=actual_text_vocab,
            grid_size=tuple(ckpt.get("grid_size", (16, 16, 16))),
            d_model=actual_d_model, d_text=actual_d_text, channels=actual_channels,
            channel_multipliers=tuple(actual_ch_mult),
            num_timesteps=ckpt.get("num_timesteps", 50),
        ).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()
        self.diffusion_model = model
        self.current_diffusion_name = df_name
        return True

    def _load_continue_checkpoint(self, model_type: str, name: str, device) -> bool:
        """Checkpoint fuer 'Weiter trainieren' im Worker-Thread laden.

        Gilt fuer Transformer und Diffusion; TF-Diffusion sammelt seinen
        Checkpoint-Zustand in _prepare_tf_diffusion_continue ein (dort kommen
        auch Vokabular, Encoder und Zielordner dazu).

        Der Model-Manager uebergibt nur den Namen. Das Laden passiert bewusst
        hier und nicht ueber den asynchronen GUI-Loader: sonst haengt es vom
        Zufall ab, ob der Trainingsstart schon die Gewichte des angeklickten
        Modells sieht (bisher wurde dadurch ein neues Modell trainiert).
        """
        entry = self.model_registry.get(name)
        if entry is None or not entry.checkpoint_path.exists():
            self._discover_models()
            entry = self.model_registry.get(name)
        if entry is None or not entry.checkpoint_path.exists():
            return False
        if model_type == "transformer":
            if self.transformer_model is not None and self.current_transformer_name == name:
                return True
            return self._load_transformer_entry(name, device)
        if model_type == "diffusion":
            if self.diffusion_model is not None and self.current_diffusion_name == name:
                return True
            return self._load_diffusion_entry(name, device)
        return False

    def _load_models(self):
        """Load the selected/default checkpoints (each one isolated).

        A broken checkpoint no longer discards the models that loaded fine and
        every failure is logged with a traceback.
        """
        device = torch.device("cuda" if torch.cuda.is_available() and self.config.gpu_enabled else "cpu")
        # Vor dem Laden aufraeumen: der Trainer/Encoder kann Speicher hinterlassen
        # haben, den `mem_get_info` sonst als belegt meldet.
        release_cuda_cache()
        self.model_registry.discover()
        tf_name = self.current_transformer_name or self.model_registry.default_transformer
        if self.generation_dit_mode:
            dit_models = [m for m in self.model_registry.get_by_type("transformer")
                          if m.recurrent_depth > 1 or m.moe_experts > 0]
            tf_name = self.current_transformer_name if self.current_transformer_name in {m.name for m in dit_models} else (dit_models[0].name if dit_models else None)
        df_name = self.current_diffusion_name or self.model_registry.default_diffusion
        tf_diff_name = self.current_tf_diffusion_name or self.model_registry.default_tf_diffusion
        msg_parts: List[str] = []
        errors: List[str] = []

        if tf_diff_name and self._guard_model_load(
            f"TF-Diffusion ({tf_diff_name})", errors,
            lambda: self._load_tf_diffusion_entry(tf_diff_name, device),
        ):
            msg_parts.append(f"TF-Diffusion ({tf_diff_name})")
        if tf_name and self._guard_model_load(
            f"Transformer ({tf_name})", errors,
            lambda: self._load_transformer_entry(tf_name, device),
        ):
            msg_parts.append(f"Transformer ({tf_name})")
        if df_name and self._guard_model_load(
            f"Diffusion ({df_name})", errors,
            lambda: self._load_diffusion_entry(df_name, device),
        ):
            msg_parts.append(f"Diffusion ({df_name})")

        status = " | ".join(msg_parts) if msg_parts else self._tr("status_no_model")
        if errors:
            status = f"{status} | {self._tr('status_model_errors')}: {', '.join(errors)[:120]}"
        self.after(0, lambda s=status: self.model_status.configure(text=s))

    # ═══════════════════════════════════════════════════════════════
    # GENERATION
    # ═══════════════════════════════════════════════════════════════

    def _update_diffusion_visibility(self):
        is_diff = self.model_type in ("diffusion", "transformer_diffusion")
        self.diff_steps_label.configure(text_color=("white", "white") if is_diff else ("gray70", "gray70"))
        self.diff_steps_slider.configure(state="normal" if is_diff else "disabled")
        self.diff_steps_value.configure(text_color=("white", "white") if is_diff else ("gray70", "gray70"))

    def _generate_structure(self):
        if self.generation_running:
            return
        prompt = self.prompt_text.get("1.0", "end-1c").strip()
        if not prompt:
            self.status_var.set(self._tr("status_enter_prompt"))
            return
        # UI-Werte hier im Main-Thread lesen (Tk ist nicht thread-safe) und an
        # den Worker übergeben.
        settings = {
            "temperature": float(self.temp_slider.get()),
            "top_k": int(self.topk_slider.get()),
            "diffusion_steps": int(self.diff_steps_slider.get()),
        }
        self.generation_running = True
        self.generate_btn.configure(state="disabled", text=self._tr("btn_generating"))
        self.progress_bar.start()
        self.status_var.set(self._tr("status_generating"))
        threading.Thread(target=self._generate_worker, args=(prompt, settings), daemon=True).start()

    def _plan_generation(self, device) -> Tuple[Optional[int], str]:
        """Logits-Chunk fuer die Generierung bestimmen (app/training_memory.py).

        Die Generierung rechnete die Logits bisher am Stueck: bei Grid 32^3 und
        17.690 Blockzustaenden sind das 2,16 GiB pro Kopie und mit Temperatur-,
        Top-K-, Softmax- und Permute-Kopie kurzzeitig ~8,6 GiB - auf einer
        6-GB-Karte endet das nach einem Trainingslauf in "Cuda out of memory".
        Rueckgabe: (chunk_voxels oder None, Meldung fuer die Statuszeile).
        """
        language = getattr(self.config, "language", "en") or "en"
        if self.model_type == "transformer" and self.transformer_model is not None:
            model = self.transformer_model
            grid_size = tuple(int(v) for v in model.grid_size)
            layers = model.recurrent_depth if model.decoder is None else len(model.decoder.layers)
            first = model.decoder_layer if model.decoder is None else model.decoder.layers[0]
            attn, rest = transformer_generation_bytes(
                grid_size, 1, model.block_vocab_size, model.d_model,
                first.self_attn.num_heads, layers, first.linear1.out_features,
                window_size=model.window_size,
                recurrent_depth=model.recurrent_depth if model.decoder is None else 1)
            plan = plan_generation(grid_size, model.block_vocab_size, device=device,
                                   activation_bytes=attn + rest)
        elif self.model_type == "diffusion" and self.diffusion_model is not None:
            model = self.diffusion_model
            grid_size = tuple(int(v) for v in model.grid_size)
            activation = unet_inference_bytes(
                1, model.channels, tuple(model.channel_multipliers), grid_size)
            plan = plan_generation(grid_size, model.num_blocks, device=device,
                                   activation_bytes=activation)
        elif (self.model_type == "transformer_diffusion"
              and self.tf_diffusion_model is not None):
            model = self.tf_diffusion_model
            grid_size = tuple(int(v) for v in model.grid_size)
            activation = unet_inference_bytes(
                1, model.channels, tuple(model.channel_multipliers), grid_size,
                model.cross_attn_heads, GENERATION_CONTEXT_TOKENS)
            plan = plan_generation(grid_size, model.num_blocks, device=device,
                                   activation_bytes=activation)
        else:
            return None, ""

        message = describe_generation_plan(plan, language)
        if not plan.fits:
            self.after(0, lambda m=message: self.status_var.set(m))
            raise RuntimeError(self._tr(
                "gen_error_vram",
                need=format_gib(plan.estimated_bytes),
                free=format_gib(plan.budget_bytes)))
        return (plan.chunk_voxels if plan.chunked else None), message

    def _generate_worker(self, prompt: str, settings: Dict[str, Any]):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() and self.config.gpu_enabled else "cpu")
            temp = float(settings["temperature"])
            topk = int(settings["top_k"])
            diff_steps = int(settings["diffusion_steps"])
            # Bloecke aus Training/Modellaufbau freigeben, BEVOR der freie
            # Speicher gemessen wird - sonst plant der Chunk zu klein.
            release_cuda_cache()
            chunk_voxels, plan_message = self._plan_generation(device)
            if plan_message:
                self.after(0, lambda m=plan_message: self.status_var.set(m))
            info = ""
            if self.model_type == "transformer" and self.transformer_model is not None:
                if self.transformer_model.text_encoder_name:
                    if (self.dit_encoder is None or
                            self.dit_encoder.display_name != self.transformer_model.text_encoder_name):
                        self.dit_encoder = (
                            self.tf_encoder if self.tf_encoder is not None and
                            self.tf_encoder.display_name == self.transformer_model.text_encoder_name
                            else self._create_tf_encoder(self.transformer_model.text_encoder_name))
                    with torch.no_grad():
                        encoded = self.dit_encoder([prompt])
                        model_dtype = next(self.transformer_model.parameters()).dtype
                        prompt_embeddings = encoded["last_hidden_state"].to(device=device, dtype=model_dtype)
                        prompt_mask = encoded["attention_mask"].to(device=device)
                    grid = self.transformer_model.generate(
                        temperature=temp, top_k=topk, chunk_voxels=chunk_voxels,
                        prompt_embeddings=prompt_embeddings,
                        prompt_attention_mask=prompt_mask,
                    )[0].cpu()
                else:
                    prompt_ids = self.prompt_tokenizer.encode(prompt).unsqueeze(0).to(device)
                    grid = self.transformer_model.generate(
                        prompt_ids, temperature=temp, top_k=topk,
                        chunk_voxels=chunk_voxels)[0].cpu()
                self.id_to_block = self.voxel_tokenizer.id_to_block
                gs = grid.shape
                info = self._tr("gen_transformer_done", x=gs[0], y=gs[1], z=gs[2], temp=f"{temp:.2f}", topk=topk, prompt=prompt[:60])
            elif self.model_type == "diffusion" and self.diffusion_model is not None:
                num_steps = clamp_num_steps(diff_steps, self.diffusion_model.num_timesteps)
                prompt_ids = self.prompt_tokenizer.encode(prompt).unsqueeze(0).to(device)
                grid = self.diffusion_model.sample(
                    prompt_ids, num_steps=num_steps, temperature=temp,
                    chunk_voxels=chunk_voxels)[0].cpu()
                self.id_to_block = self.voxel_tokenizer.id_to_block
                gs = grid.shape
                info = self._tr("gen_diffusion_done", x=gs[0], y=gs[1], z=gs[2], temp=f"{temp:.2f}", steps=num_steps, prompt=prompt[:60])
            elif self.model_type == "transformer_diffusion" and self.tf_diffusion_model is not None and self.tf_encoder is not None:
                num_steps = clamp_num_steps(diff_steps, self.tf_diffusion_model.num_timesteps)
                with torch.no_grad():
                    encoded = self.tf_encoder([prompt])
                    # Cast to model's dtype to avoid Half/Float mismatch
                    model_dtype = next(self.tf_diffusion_model.parameters()).dtype
                    context = encoded["last_hidden_state"].to(device=device, dtype=model_dtype)
                    context_mask = encoded["attention_mask"].to(device=device)
                grid = self.tf_diffusion_model.sample(
                    context, context_mask, num_steps=num_steps,
                    temperature=temp, top_k=topk,
                    chunk_voxels=chunk_voxels)[0].cpu()
                self.id_to_block = self.voxel_tokenizer.id_to_block
                gs = grid.shape
                info = self._tr("gen_tf_diffusion_done", x=gs[0], y=gs[1], z=gs[2], temp=f"{temp:.2f}", topk=topk, steps=num_steps, encoder=self.tf_encoder.display_name, prompt=prompt[:60])
            else:
                self.after(0, self._generation_failed, self._tr("gen_no_model"))
                return
            raw_shape = tuple(int(v) for v in grid.shape)
            trimmed_grid = trim_token_grid(grid, air_id=0)
            preview_grid = center_token_grid(trimmed_grid, raw_shape, air_id=0)
            self.generated_grid = trimmed_grid
            self.generated_np = preview_grid.numpy()
            trimmed_shape = tuple(int(v) for v in trimmed_grid.shape)
            unique_blocks = int(torch.unique(trimmed_grid).numel())
            info += "\n" + self._tr("gen_block_types", count=unique_blocks)
            unsaved_mask = (trimmed_grid < 0) | (trimmed_grid >= len(self.id_to_block))
            unsaved_count = int(unsaved_mask.sum().item())
            if unsaved_count:
                info += "\n" + self._tr("gen_unsaved_blocks", count=unsaved_count)
            if trimmed_shape != raw_shape:
                info += "\n" + self._tr("gen_trim", rx=raw_shape[0], ry=raw_shape[1], rz=raw_shape[2], tx=trimmed_shape[0], ty=trimmed_shape[1], tz=trimmed_shape[2])
            self.after(0, self._generation_done, info)
        except Exception as e:
            self.after(0, self._generation_failed, str(e))

    def _generation_done(self, info: str):
        self.progress_bar.stop()
        self.progress_bar.set(1)
        self.generate_btn.configure(state="normal", text=self._tr("btn_generate"))
        self.save_btn.configure(state="normal")
        self.export_btn.configure(state="normal")
        self.viewer_btn.configure(state="normal" if HAS_PYGLET else "disabled")
        self.info_var.set(info)
        self.status_var.set(self._tr("status_done"))
        self.generation_running = False
        self._update_preview()
        self.current_project_path = None
        # Chunk-Puffer des Samplings zurueckgeben, damit der naechste Lauf (und
        # die Planung) wieder den vollen freien Speicher sieht.
        release_cuda_cache()

    def _generation_failed(self, error: str):
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.generate_btn.configure(state="normal", text=self._tr("btn_generate"))
        if "out of memory" in str(error).lower():
            error = f"{error}{self._tr('gen_oom_hint')}"
        self.info_var.set(self._tr("gen_error", error=error))
        self.status_var.set(self._tr("gen_error", error=error))
        self.generation_running = False
        release_cuda_cache()

    def _open_3d_viewer(self):
        if self.generated_np is None or self.id_to_block is None:
            return
        if not HAS_PYGLET:
            self.status_var.set(self._tr("status_pyglet_missing"))
            return
        open_3d_viewer(self.generated_np, self.id_to_block, f"Minecraft 3D - {self.model_type.upper()} Generierung")

    # ═══════════════════════════════════════════════════════════════
    # TF PREVIEW (reuse the same generated grid)
    # ═══════════════════════════════════════════════════════════════

    def _update_tf_preview(self):
        """Update preview — delegates to _update_preview since TF-Diffusion
        uses the same preview panel as all other model types."""
        self._update_preview()

    # ═══════════════════════════════════════════════════════════════
    # 3D PREVIEW
    # ═══════════════════════════════════════════════════════════════

    def _preview_scene(self) -> Optional[VoxelScene]:
        """Build (and cache) the render scene for the current grid.

        Identity checks are used on purpose: the cache is dropped as soon as a
        new generated grid or block palette is assigned, and holding a reference
        to the old array keeps its id from being reused.
        """
        if self.generated_np is None or self.id_to_block is None:
            return None
        if (self._preview_scene_cache is None
                or self._preview_scene_source is not self.generated_np
                or self._preview_scene_palette is not self.id_to_block):
            self._preview_scene_source = self.generated_np
            self._preview_scene_palette = self.id_to_block
            self._preview_scene_cache = VoxelScene.build(self.generated_np, self.id_to_block)
        return self._preview_scene_cache

    def _preview_angles(self) -> tuple:
        """Camera angles for the currently selected view mode."""
        return view_angles(self.current_view, self.azimuth, self.elevation)

    def _update_preview(self, interactive: bool = False):
        """Request a preview frame (never blocks; the worker thread draws it).

        interactive=True renders at a reduced resolution, which is what the drag
        and zoom handlers use: the frame appears immediately and a full quality
        frame follows once the interaction stops.
        """
        scene = self._preview_scene()
        if scene is None:
            return
        factor = PREVIEW_DRAG_RESOLUTION if interactive else 1.0
        width, height = PREVIEW_SIZE
        azimuth, elevation = self._preview_angles()
        self.preview_renderer.request(
            scene,
            size=(max(1, int(width * factor)), max(1, int(height * factor))),
            azimuth=azimuth,
            elevation=elevation,
            scale=self.zoom_scale * factor,
        )
        self._schedule_preview_poll()

    def _schedule_preview_poll(self):
        if self._preview_poll_job is None:
            self._preview_poll_job = self.after(PREVIEW_POLL_MS, self._poll_preview)

    def _poll_preview(self):
        """Collect finished frames and swap them into the preview label."""
        self._preview_poll_job = None
        image = self.preview_renderer.take_latest()
        if image is not None:
            img_tk = ctk.CTkImage(light_image=image, dark_image=image, size=PREVIEW_SIZE)
            self.current_preview_img = img_tk
            self.preview_label.configure(image=img_tk, text="")
        if self.preview_renderer.is_busy():
            self._schedule_preview_poll()

    def _request_preview_refresh(self):
        """Render a full quality frame shortly after the last interaction."""
        if self._preview_refresh_job is not None:
            self.after_cancel(self._preview_refresh_job)
        self._preview_refresh_job = self.after(
            PREVIEW_REFRESH_MS, self._preview_refresh_now)

    def _preview_refresh_now(self):
        self._preview_refresh_job = None
        if not self._dragging:
            self._update_preview()

    def _on_mouse_down(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._orbit_start_az = self.azimuth
        self._orbit_start_el = self.elevation
        self._dragging = True

    def _on_mouse_drag(self, event):
        if not self._dragging:
            return
        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y
        self.azimuth = (self._orbit_start_az - dx * 0.5) % 360
        # Vertical dragging was computed but discarded before; keep the camera
        # above the horizon and let the user tilt the view.
        self.elevation = max(5.0, min(89.0, self._orbit_start_el + dy * 0.4))
        self.current_view = "free"
        self._update_preview(interactive=True)

    def _on_mouse_up(self, event):
        self._dragging = False
        self._update_preview()

    def _on_mouse_wheel(self, event):
        delta = event.delta / 120
        self.zoom_scale = max(1.5, min(6.0, self.zoom_scale + delta * 0.3))
        self.zoom_slider.set(self.zoom_scale)
        self._update_preview(interactive=True)
        self._request_preview_refresh()

    def _update_zoom(self, value: float):
        self.zoom_scale = value
        if self.generated_np is not None and self.id_to_block is not None:
            self._update_preview(interactive=True)
            self._request_preview_refresh()

    def _set_horizontal_orbit(self, az: float):
        self.azimuth = az
        self.elevation = 30.0
        self.current_view = "free"
        self._update_preview()

    # ═══════════════════════════════════════════════════════════════
    # PROJECT MANAGEMENT
    # ═══════════════════════════════════════════════════════════════

    def _save_to_project(self):
        if self.generated_grid is None or self.id_to_block is None:
            return
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt = self.prompt_text.get("1.0", "end-1c").strip()[:50]
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:40]
        proj_dir = PROJECTS_DIR / f"{safe_name}_{timestamp}"
        proj_dir.mkdir(parents=True, exist_ok=True)
        schem_path = proj_dir / "structure.schem"
        save_schem(schem_path, self.generated_grid, self.id_to_block)
        meta = {"prompt": prompt, "model": self.model_type, "temperature": self.temp_slider.get(),
                "top_k": int(self.topk_slider.get()), "grid_size": list(self.generated_grid.shape),
                "timestamp": timestamp, "block_count": int(torch.unique(self.generated_grid).numel())}
        (proj_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (proj_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        self.current_project_path = proj_dir
        self.status_var.set(self._tr("status_project_saved", name=proj_dir.name))
        self._refresh_projects()

    def _export_schematic(self):
        if self.generated_grid is None or self.id_to_block is None:
            return
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        prompt = self.prompt_text.get("1.0", "end-1c").strip()[:40]
        safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:30]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = EXPORTS_DIR / f"{safe}_{timestamp}.schem"
        save_schem(path, self.generated_grid, self.id_to_block)
        self.status_var.set(self._tr("status_exported", path=path))

    def _refresh_projects(self):
        self.project_listbox.configure(state="normal")
        self.project_listbox.delete("1.0", "end")
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        projects = sorted(PROJECTS_DIR.iterdir()) if PROJECTS_DIR.exists() else []
        for i, proj in enumerate(projects):
            if proj.is_dir():
                meta_path = proj / "metadata.json"
                pt = "?"
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        pt = meta.get("prompt", "?")
                    except Exception:
                        pass
                self.project_listbox.insert("end", f"{i+1}. {proj.name}\n")
                self.project_listbox.insert("end", f"   {pt}\n\n")
        self.project_listbox.configure(state="disabled")

    def _load_selected_project(self):
        selection = self._get_selected_project()
        if selection is None:
            self.status_var.set(self._tr("status_no_project"))
            return
        try:
            schem_files = list(selection.glob("*.schem"))
            if not schem_files:
                self.status_var.set(self._tr("status_no_schem"))
                return
            schematic = load_schematic(schem_files[0])
            meta = {}
            meta_path = selection / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            blocks = schematic.blocks
            from dataset import AIR
            unique = sorted(set(blocks))
            if AIR in unique:
                unique.remove(AIR)
                unique.insert(0, AIR)
            id2block = unique
            flat = [id2block.index(b) if b in id2block else 0 for b in blocks]
            gx, gy, gz = schematic.size
            tensor_grid = torch.tensor(flat, dtype=torch.long).reshape(gy, gz, gx).permute(2, 0, 1)
            self.generated_grid = tensor_grid
            self.generated_np = tensor_grid.numpy()
            self.id_to_block = id2block
            prompt = meta.get("prompt", prompt_text_from_txt(selection))
            self.info_var.set(self._tr("gen_loaded", name=selection.name, x=schematic.size[0], y=schematic.size[1], z=schematic.size[2], prompt=prompt))
            self._update_preview()
            self._update_tf_preview()
            self.save_btn.configure(state="normal")
            self.export_btn.configure(state="normal")
            self.status_var.set(self._tr("status_project_loaded", name=selection.name))
        except Exception as e:
            self.status_var.set(self._tr("status_load_error", err=e))

    def _delete_selected_project(self):
        selection = self._get_selected_project()
        if selection is None:
            return
        shutil.rmtree(selection)
        self.status_var.set(self._tr("status_project_deleted", name=selection.name))
        self._refresh_projects()

    def _export_selected_project(self):
        selection = self._get_selected_project()
        if selection is None:
            return
        schem_files = list(selection.glob("*.schem"))
        if schem_files:
            EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
            dest = EXPORTS_DIR / f"{selection.name}.schem"
            shutil.copy2(schem_files[0], dest)
            self.status_var.set(self._tr("status_project_exported", dest=dest))

    def _get_selected_project(self) -> Optional[Path]:
        text = self.project_listbox.get("1.0", "end-1c").strip()
        if not text:
            return None
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        projects = sorted(PROJECTS_DIR.iterdir()) if PROJECTS_DIR.exists() else []
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit():
                name_part = line.split(". ", 1)[-1]
                for proj in projects:
                    if proj.name == name_part:
                        return proj
        return None

    # ═══════════════════════════════════════════════════════════════
    # SETTINGS
    # ═══════════════════════════════════════════════════════════════

    def _save_settings(self):
        old_lang = self.config.language
        self.config.gpu_enabled = bool(self.setting_gpu.get())
        self.config.language = "de" if self.setting_lang.get() == "Deutsch" else "en"
        self.config.save()
        self.status_var.set(self._tr("status_settings_saved"))
        # Rebuild UI if language changed
        if old_lang != self.config.language:
            self.after(100, self._rebuild_ui)
        self._load_models_async()

    def _stop_training(self):
        if not self.training_running:
            return
        self.status_var.set(self._tr("status_training_stopping"))
        self.training_running = False

    # ═══════════════════════════════════════════════════════════════
    # HIDDEN STATE CACHE
    # ═══════════════════════════════════════════════════════════════

    def _get_training_schem_files(self) -> Tuple[List[Path], List[Path]]:
        """All (schem, txt) pairs of the configured data dirs that have prompts.

        Uses dataset.discover_prompt_files so the hidden state cache sees exactly
        the same structures as the training dataset (recursive, nested data sets).
        """
        return discover_prompt_files(self.data_dirs)

    def _on_cached_hs_toggle(self):
        """Called when the cached hidden states switch is toggled."""
        if self.tf_use_cached_hs_var.get():
            # Check if cache exists from the selected cache
            cache_sel = self.tf_cache_selector.get()
            if cache_sel == self._tr("no_cache"):
                self.tf_cache_status_label.configure(text=self._tr("select_cache_first"))
                return
            schem_files, txt_files = self._get_training_schem_files()
            result = validate_cache(cache_sel, schem_files, txt_files)
            self.tf_cache_status_label.configure(text=result["message"])
            if not result["valid"]:
                self.status_var.set(self._tr("status_cache_warning", msg=result['message']))
        else:
            self.tf_cache_status_label.configure(text=self._tr("no_cache_used"))

    def _precompute_hidden_states(self):
        """Pre-compute hidden states in a background thread."""
        if self.tf_encoder is None:
            self.status_var.set(self._tr("status_no_encoder_loaded"))
            return
        encoder_name = self.tf_encoder_combo.get()
        schem_files, txt_files = self._get_training_schem_files()
        if not schem_files:
            self.status_var.set(self._tr("status_no_training_data"))
            return

        self.status_var.set(self._tr("status_computing_hs", count=len(schem_files)))
        self.tf_cache_status_label.configure(text=self._tr("computing"))
        threading.Thread(
            target=self._precompute_hidden_states_worker,
            args=(encoder_name, schem_files, txt_files),
            daemon=True,
        ).start()

    def _precompute_hidden_states_worker(self, encoder_name: str, schem_files: List[Path], txt_files: List[Path]):
        """Worker thread for pre-computing hidden states."""
        try:
            def status_cb(msg: str):
                self.after(0, lambda m=msg: self.status_var.set(m))
                self.after(0, lambda m=msg: self.tf_cache_status_label.configure(text=m))

            cache_dir = compute_hidden_states(
                self.tf_encoder, encoder_name, schem_files, txt_files,
                status_callback=status_cb,
            )
            self.after(0, lambda: self.tf_cache_status_label.configure(
                text=self._tr("cache_created", count=len(schem_files))))
            self.after(0, lambda: self.status_var.set(
                self._tr("status_hs_saved", name=cache_dir.name)))
        except Exception as e:
            self.after(0, lambda msg=str(e): self.tf_cache_status_label.configure(
                text=self._tr("cache_error", msg=msg[:60])))
            self.after(0, lambda msg=str(e): self.status_var.set(self._tr("cache_error", msg=msg[:60])))

    def _check_cache_status(self):
        """Check the cache status and display it."""
        encoder_name = self.tf_encoder_combo.get()
        schem_files, txt_files = self._get_training_schem_files()
        result = validate_cache(encoder_name, schem_files, txt_files)
        self.tf_cache_status_label.configure(text=result["message"])
        if result["valid"]:
            self.status_var.set(result["message"])
        else:
            self.status_var.set(self._tr("status_cache_warning", msg=result['message']))

    # ═══════════════════════════════════════════════════════════════
    # MODEL SIZE / ARCHITECTURE SELECTION
    # ═══════════════════════════════════════════════════════════════

    def _get_selected_grid_size(self) -> tuple[int, int, int]:
        return grid_size_for_label(self.grid_size_var.get())

    def _get_suggested_arch(self) -> Optional[dict]:
        grid_size = self._get_selected_grid_size()
        window_size = self._get_attention_window_size()
        # Use actual vocab sizes from loaded tokenizers if available, otherwise use defaults
        text_vocab = len(self.prompt_tokenizer.token_to_id) if self.prompt_tokenizer else 129
        block_vocab = len(self.voxel_tokenizer.id_to_block) if self.voxel_tokenizer else 253
        if self.show_advanced:
            try:
                d_model = int(self.adv_vars["d_model"].get())
                nhead = int(self.adv_vars["nhead"].get())
                layers = int(self.adv_vars["layers"].get())
                ff_ratio = int(self.adv_vars["ff_ratio"].get())
                dim_ff = d_model * ff_ratio
                from model import estimate_transformer_params
                params = estimate_transformer_params(text_vocab, block_vocab, grid_size, d_model, nhead, layers, dim_ff)
                return {"d_model": d_model, "nhead": nhead, "num_layers": layers,
                        "dim_feedforward": dim_ff, "params": params, "params_m": params / 1_000_000,
                        "window_size": window_size}
            except (ValueError, ZeroDivisionError):
                return None
        else:
            target_m = self.size_slider_var.get()
            from model import suggest_architecture
            arch = suggest_architecture(target_m, text_vocab, block_vocab, grid_size)
            arch["window_size"] = window_size
            return arch

    def _get_attention_window_size(self) -> int:
        value = str(self.attention_window_var.get())
        return 0 if value == "Global" else int(value.split("³", 1)[0])

    def _on_attention_window_changed(self):
        if self.show_advanced:
            self._update_arch_from_advanced()
        else:
            self._on_size_slider(self.size_slider_var.get())

    def _on_size_slider(self, value: float):
        if self.show_advanced:
            return
        arch = self._get_suggested_arch()
        if arch is None:
            self.params_info_var.set(self._tr("no_arch_found"))
            return
        self.params_info_var.set(self._tr("target_arch", val=f"{value:.1f}", params=f"{arch['params_m']:.2f}", d_model=arch['d_model'], nhead=arch['nhead'], layers=arch['num_layers'], ffn=arch['dim_feedforward']))
        arch["window_size"] = self._get_attention_window_size()
        self._selected_arch = arch

    def _update_aug_label(self, value: float):
        if hasattr(self, 'augmentation_diversity_label'):
            self.augmentation_diversity_label.configure(text=str(int(round(float(value)))))

    def _update_air_weight_label(self, value: float):
        if hasattr(self, 'air_weight_label'):
            self.air_weight_label.configure(text=str(int(round(float(value)))))

    def _set_model_size(self, value_m: float):
        self.size_slider_var.set(value_m)
        self._on_size_slider(value_m)

    def _toggle_advanced(self):
        self.show_advanced = not self.show_advanced
        if self.show_advanced:
            self.advanced_frame.grid()
            self.advanced_btn.configure(text=self._tr("btn_advanced_open"))
            self._update_arch_from_advanced()
        else:
            self.advanced_frame.grid_remove()
            self.advanced_btn.configure(text=self._tr("btn_advanced"))
            self._on_size_slider(self.size_slider_var.get())

    def _update_arch_from_advanced(self):
        if not self.show_advanced:
            return
        arch = self._get_suggested_arch()
        if arch is None:
            self.params_info_var.set(self._tr("invalid_values"))
            return
        self.params_info_var.set(self._tr("manual_arch", params=f"{arch['params_m']:.2f}", d_model=arch['d_model'], nhead=arch['nhead'], layers=arch['num_layers'], ffn=arch['dim_feedforward']))
        self._selected_arch = arch

    # ═══════════════════════════════════════════════════════════════
    # TRAINING
    # ═══════════════════════════════════════════════════════════════

    def _read_int_entry(self, widget, default: int) -> int:
        """Read an integer from an entry widget (main thread only)."""
        try:
            text = str(widget.get()).strip()
        except (TypeError, ValueError, AttributeError):
            return default
        try:
            return int(text)
        except ValueError:
            return default

    def _collect_training_params(self) -> Dict[str, Any]:
        """Read the shared training parameters - main thread only.

        Tk widgets/variables must never be touched from a worker thread (Tcl is
        not thread safe), so all values are snapshotted here and passed down.
        """
        try:
            epochs = int(str(self.train_epochs_entry.get()).strip())
        except (TypeError, ValueError):
            epochs = 10
        try:
            batch_size = int(str(self.train_batch_entry.get()).strip())
        except (TypeError, ValueError):
            batch_size = 16
        try:
            lr = float(str(self.train_lr_entry.get()).strip())
        except (TypeError, ValueError):
            lr = 1.5e-3
        air_weight = 75.0
        if hasattr(self, "air_weight_var"):
            try:
                air_weight = float(self.air_weight_var.get())
            except (TypeError, ValueError):
                air_weight = 75.0
        grad_accum = accumulation_enabled()
        if hasattr(self, "setting_grad_accum"):
            try:
                grad_accum = bool(self.setting_grad_accum.get())
            except (TypeError, ValueError):
                grad_accum = accumulation_enabled()
        return {
            "epochs": max(1, epochs),
            "batch_size": max(1, batch_size),
            "lr": max(1e-6, lr),
            "air_weight": air_weight,
            "grad_accum": grad_accum,
        }

    def _dataset_progress_callback(self):
        """Rückmeldung aus dem Dataset-Aufbau in die Statuszeile (Thread -> Tk).

        Der Aufbau dekodiert jede Struktur einmal; bei mehreren tausend
        Strukturen dauert das Minuten. Ohne Rückmeldung sieht das aus wie ein
        haengendes Training.
        """
        state = {"last": 0.0}

        def report(done: int, total: int) -> None:
            now = time.time()
            if done < total and now - state["last"] < 0.4:
                return
            state["last"] = now
            self.after(0, lambda d=done, t=total: self.status_var.set(
                self._tr("status_dataset_progress", done=d, total=t)))

        return report

    def _report_dataset_ready(self, dataset) -> None:
        try:
            count = len(dataset)
            skipped = len(getattr(dataset, "skipped_examples", []) or [])
        except TypeError:
            return
        self.after(0, lambda c=count, s=skipped: self.status_var.set(
            self._tr("status_dataset_ready", count=c, skipped=s)))

    def _apply_training_plan(self, plan, requested_batch: int) -> tuple[int, int | None, int]:
        """VRAM-Plan anwenden: Batch, Logits-Chunk und Mikro-Batch zurueckgeben.

        Der Plan (app/training_memory.py) rechnet aus, ob Logits, Aktivierungen
        und Optimizer in den freien VRAM passen. Passt der Wunsch-Batch nicht,
        wird er verkleinert statt mit "Cuda out of memory" abzubrechen; passt
        selbst Batch 1 nicht, gibt es eine klare Fehlermeldung samt Vorschlag.

        Mit Gradient-Akkumulation (``plan.accumulated``) bleibt die *effektive*
        Batch-Groesse erhalten - die Schleifen teilen den Batch dann in
        ``plan.accum_steps`` Mikro-Batches. Rueckgabe daher:
        ``(batch_size, chunk_voxels, micro_batch_size)`` - ``chunk_voxels``
        ``None`` bedeutet ungechunkt, ``batch_size`` ist die Groesse des
        DataLoader-Batches (= Beispiele pro Optimizer-Schritt).
        """
        language = getattr(self.config, "language", "en") or "en"
        message = describe_training_plan(plan, language)
        if not plan.fits:
            self.after(0, lambda m=message: self.status_var.set(m))
            raise RuntimeError(self._tr(
                "training_error_vram",
                need=format_gib(plan.estimated_bytes),
                free=format_gib(plan.budget_bytes)))
        if plan.accumulated:
            message += " - " + self._tr(
                "grad_accum_status", batch=plan.effective_batch_size,
                steps=plan.accum_steps, micro=plan.micro_batch_size)
        self.after(0, lambda m=message: self.status_var.set(m))
        chunk = plan.chunk_voxels if plan.chunked else None
        return plan.effective_batch_size, chunk, plan.micro_batch_size

    def _start_training(self, model_type: str, grid_size: Optional[Tuple[int, int, int]] = None,
                        overrides: Optional[Dict[str, Any]] = None,
                        continue_from: Optional[str] = None):
        """Training fuer 'transformer' oder 'diffusion' starten.

        ``grid_size``/``overrides``/``continue_from`` kommen aus dem
        Model-Manager ("Weiter trainieren"): dort bestimmt der Checkpoint Grid
        und Zielwerte, die Worker bleiben dieselben.
        """
        if self.training_running:
            return
        if model_type != "transformer":
            # Classic Diffusion has been removed from the application UI.
            self.status_var.set(self._tr("status_unknown_train_type", type=model_type))
            return
        arch = getattr(self, '_selected_arch', None)
        if arch is None:
            self._on_size_slider(self.size_slider_var.get())
            arch = self._selected_arch
        arch = dict(arch or {})
        arch["window_size"] = self._get_attention_window_size()
        if self.train_model_type_selector.get() == "DiT (Recurrent)":
            try:
                arch["recurrent_depth"] = int(self.dit_depth_var.get())
                arch["moe_experts"] = max(0, int(self.moe_experts_var.get())) if self.moe_enabled_var.get() else 0
                arch["moe_active"] = max(1, int(self.moe_active_var.get()))
            except (TypeError, ValueError):
                self.status_var.set("Ungültige DiT/MoE-Konfiguration")
                return
            if arch["recurrent_depth"] < 2:
                self.status_var.set("DiT braucht mindestens 2 Recurrent-Schritte")
                return
            if arch["recurrent_depth"] > 64 or arch["moe_experts"] > 32:
                self.status_var.set("Maximal 64 Recurrent-Schritte und 32 Experten sind erlaubt")
                return
            if arch["moe_experts"] and arch["moe_active"] > arch["moe_experts"]:
                self.status_var.set("Aktive Experten dürfen nicht größer als die Gesamtzahl sein")
                return
            from model import estimate_active_transformer_params, estimate_transformer_params
            text_vocab = len(self.prompt_tokenizer.token_to_id) if self.prompt_tokenizer else 129
            block_vocab = len(self.voxel_tokenizer.id_to_block) if self.voxel_tokenizer else 253
            selected_encoder = self.dit_text_encoder_var.get()
            text_input_dim = (ENCODER_HIDDEN_DIMS.get(selected_encoder, DEFAULT_ENCODER_HIDDEN_DIM)
                              if selected_encoder != "Learned embeddings" else 0)
            total_params = estimate_transformer_params(text_vocab, block_vocab, grid_size or self._get_selected_grid_size(),
                arch["d_model"], arch["nhead"], arch["num_layers"], arch["dim_feedforward"],
                arch["recurrent_depth"], arch["moe_experts"], text_input_dim)
            active_params = estimate_active_transformer_params(text_vocab, block_vocab, grid_size or self._get_selected_grid_size(),
                arch["d_model"], arch["nhead"], arch["num_layers"], arch["dim_feedforward"],
                arch["recurrent_depth"], arch["moe_experts"], arch["moe_active"], text_input_dim)
            suffix = f" | MoE {arch['moe_active']}/{arch['moe_experts']} Experten" if arch["moe_experts"] else ""
            self.params_info_var.set(f"DiT: {active_params/1e6:.2f}M aktiv / {total_params/1e6:.2f}M gesamt{suffix}")
        self._selected_arch = arch
        grid_size = tuple(grid_size) if grid_size else self._get_selected_grid_size()
        augmentation_diversity = int(round(float(self.augmentation_diversity_var.get())))
        allow_vertical_movement = bool(self.setting_aug_vertical.get())
        params = self._collect_training_params()
        params["diffusion_steps"] = self._read_int_entry(
            getattr(self, "diff_train_steps_entry", None), 50)
        if overrides:
            params.update(overrides)
            augmentation_diversity = int(params["augmentation_diversity"])
            allow_vertical_movement = bool(params["allow_vertical_movement"])
        params["continue_from"] = continue_from
        params["dit_enabled"] = self.train_model_type_selector.get() == "DiT (Recurrent)"
        params["window_size"] = arch["window_size"]
        gs_label = f"{grid_size[0]}x{grid_size[1]}x{grid_size[2]}"
        self.training_running = True
        self.train_epoch_bar.set(0)
        self.train_batch_bar.set(0)
        self.status_var.set(self._tr("status_training_started", type=model_type, grid=gs_label))
        self.train_btn_dit.configure(state="disabled")
        self.train_btn_tf_diffusion.configure(state="disabled")
        self.train_stop_btn.configure(state="normal")
        self._current_training_type = model_type
        if params.get("dit_enabled"):
            params["dit_text_encoder_name"] = (
                None if self.dit_text_encoder_var.get() == "Learned embeddings"
                else self.dit_text_encoder_var.get())
        threading.Thread(target=self._training_worker,
                         args=(model_type, grid_size, augmentation_diversity,
                               allow_vertical_movement, params),
                         daemon=True).start()

    def _training_worker(self, model_type: str, grid_size: tuple[int, int, int],
                         augmentation_diversity: int, allow_vertical_movement: bool,
                         params: Dict[str, Any]):
        try:
            epochs = int(params["epochs"])
            batch_size = int(params["batch_size"])
            lr = float(params["lr"])
            air_weight = float(params["air_weight"])
            device = torch.device("cuda" if torch.cuda.is_available() and self.config.gpu_enabled else "cpu")
            arch = params.get("architecture", getattr(self, '_selected_arch', None))
            continue_from = params.get("continue_from")
            # Batch halten statt verkleinern (Gradient-Akkumulation, siehe
            # app/grad_accum.py) - der Schalter im Training-Tab entscheidet.
            allow_accum = bool(params.get("grad_accum", True))
            continuing_name = None
            if continue_from:
                # "Weiter trainieren" aus dem Model-Manager: genau diesen
                # Checkpoint laden und dessen Grid uebernehmen (das Grid im
                # Training-Tab ist dabei nur Anzeige).
                if not self._load_continue_checkpoint(model_type, continue_from, device):
                    raise RuntimeError(self._tr("tm_missing_checkpoint", name=continue_from))
                loaded = (self.transformer_model if model_type == "transformer"
                          else self.diffusion_model)
                continuing_name = continue_from
                grid_size = tuple(loaded.grid_size)
                self.after(0, lambda n=continue_from: self.status_var.set(
                    self._tr("status_training_continue", name=n)))
            elif (model_type == "transformer" and self.transformer_model is not None
                  and self.current_transformer_name
                  and tuple(self.transformer_model.grid_size) == grid_size):
                continuing_name = self.current_transformer_name
            elif (model_type == "diffusion" and self.diffusion_model is not None
                  and self.current_diffusion_name
                  and tuple(self.diffusion_model.grid_size) == grid_size):
                continuing_name = self.current_diffusion_name
            continuing_model = continuing_name is not None

            dit_encoder = None
            dit_encoder_name = params.get("dit_text_encoder_name") if model_type == "transformer" else None
            if continuing_model and model_type == "transformer":
                dit_encoder_name = self.transformer_model.text_encoder_name or None
            if dit_encoder_name:
                dit_encoder = (self.tf_encoder if self.tf_encoder is not None and
                               self.tf_encoder.display_name == dit_encoder_name
                               else self._create_tf_encoder(dit_encoder_name))
                self.dit_encoder = dit_encoder

            orig_pt = None
            orig_vt = None
            if continuing_model:
                pt_path = RUNS_DIR / continuing_name / "prompt_vocab.json"
                vt_path = RUNS_DIR / continuing_name / "block_vocab.json"
                if vt_path.exists():
                    orig_vt = VoxelTokenizer.load(vt_path)
                if pt_path.exists():
                    orig_pt = PromptTokenizer.load(pt_path)
                if continue_from and (orig_pt is None or orig_vt is None):
                    # Ohne das Vokabular des Checkpoints wuerden die Block-IDs
                    # nicht mehr passen - dann lieber sauber abbrechen.
                    raise RuntimeError(self._tr("tm_missing_vocab", name=continuing_name))

            # Bereits trainierte Epochen (aus dem Model-Manager, ohne erneutes
            # Lesen des Checkpoints): beim Fortsetzen kommen die neuen hinzu,
            # die Anzeige zeigt danach die Gesamtzahl.
            start_epoch = 0
            if continuing_model:
                entry = self.model_registry.get(continuing_name)
                if entry is not None:
                    start_epoch = max(0, int(entry.epochs_trained or 0))

            from dataset import MultiSourceSchematicDataset
            dataset = MultiSourceSchematicDataset(
                self.data_dirs, target_size=grid_size, max_voxels=400_000,
                augmentation_diversity=augmentation_diversity,
                allow_vertical_movement=allow_vertical_movement,
                prompt_tokenizer=orig_pt, voxel_tokenizer=orig_vt,
                air_weight_factor=air_weight,
                progress_callback=self._dataset_progress_callback(),
            )
            self._report_dataset_ready(dataset)

            # ── VRAM-Plan ──
            # Lokale Fenster-Attention und chunkweise Logits werden vor dem
            # Training gemeinsam mit Batch und Speicherbudget geplant.
            num_blocks = len(dataset.voxel_tokenizer.id_to_block)
            if model_type == "transformer":
                if continuing_model:
                    # Fortsetzen: die echte Architektur des Checkpoints, nicht
                    # die (nur angezeigte) Zielgroesse aus dem Training-Tab.
                    model = self.transformer_model
                    plan_d_model = model.d_model
                    layer = model.decoder_layer if model.decoder is None else model.decoder.layers[0]
                    plan_nhead = layer.self_attn.num_heads
                    plan_layers = model.recurrent_depth if model.decoder is None else len(model.decoder.layers)
                    plan_ff = layer.linear1.out_features
                    plan_window_size = model.window_size
                    plan_recurrent_depth = model.recurrent_depth if model.decoder is None else 1
                elif arch:
                    plan_d_model = arch["d_model"]
                    plan_nhead = arch["nhead"]
                    plan_layers = arch.get("recurrent_depth", arch["num_layers"]) if params.get("dit_enabled") else arch["num_layers"]
                    plan_ff = arch["dim_feedforward"]
                    plan_window_size = int(arch.get("window_size", 8))
                    plan_recurrent_depth = plan_layers if params.get("dit_enabled") else 1
                else:
                    plan_d_model, plan_nhead, plan_layers, plan_ff = 192, 6, 5, 768
                    plan_window_size, plan_recurrent_depth = 8, 1
                plan = plan_transformer_training(
                    grid_size, batch_size, num_blocks, device=device,
                    d_model=plan_d_model, nhead=plan_nhead,
                    num_layers=plan_layers, dim_feedforward=plan_ff,
                    window_size=plan_window_size,
                    recurrent_depth=plan_recurrent_depth,
                    allow_accumulation=allow_accum,
                )
            else:
                plan = plan_diffusion_training(
                    grid_size, batch_size, num_blocks, device=device,
                    channels=64, channel_multipliers=(1, 2, 2), d_model=128,
                    cross_attn_heads=2, context_dim=64, context_len=64,
                    allow_accumulation=allow_accum,
                )
            batch_size, chunk_voxels, micro_batch = self._apply_training_plan(plan, batch_size)

            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            total_batches = len(loader)

            if model_type == "transformer":
                from model import SharedWeightVoxelTransformer
                if continuing_model:
                    # Fortsetzen: derselbe Ordner, dieselben Gewichte/dasselbe Vokabular
                    out_dir = RUNS_DIR / continuing_name
                else:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    out_dir = RUNS_DIR / f"transformer_{timestamp}"
                out_dir.mkdir(parents=True, exist_ok=True)
                if continuing_model:
                    model = self.transformer_model
                else:
                    if arch:
                        d_model, nhead, num_layers, dim_ff = arch["d_model"], arch["nhead"], arch["num_layers"], arch["dim_feedforward"]
                    else:
                        d_model, nhead, num_layers, dim_ff = 192, 6, 5, 768
                    arch_cfg = arch or {}
                    is_dit = bool(params.get("dit_enabled", False))
                    recurrent_depth = max(1, int(arch_cfg.get("recurrent_depth", 4))) if is_dit else 1
                    moe_experts = int(arch_cfg.get("moe_experts", 0)) if is_dit else 0
                    moe_active = int(arch_cfg.get("moe_active", 1)) if is_dit else 1
                    model = SharedWeightVoxelTransformer(
                        text_vocab_size=len(dataset.prompt_tokenizer.token_to_id),
                        block_vocab_size=len(dataset.voxel_tokenizer.id_to_block),
                        grid_size=grid_size, d_model=d_model, nhead=nhead,
                        num_layers=num_layers, dim_feedforward=dim_ff, dropout=0.1,
                        recurrent_depth=recurrent_depth, moe_experts=moe_experts, moe_active=moe_active,
                        window_size=int(arch_cfg.get("window_size", params.get("window_size", 8))),
                        text_input_dim=int(dit_encoder.hidden_dim) if dit_encoder else 0,
                        text_encoder_name=dit_encoder_name,
                    ).to(device)
                    print(f"DiT parameters total/active: {model.num_parameters:,} / {model.active_parameters:,}")
                    dataset.prompt_tokenizer.save(out_dir / "prompt_vocab.json")
                    dataset.voxel_tokenizer.save(out_dir / "block_vocab.json")

                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
                total_start = time.time()
                for epoch in range(1, epochs + 1):
                    if not self.training_running:
                        break
                    model.train()
                    total_loss = 0.0
                    batch_num = 0
                    for batch in loader:
                        if not self.training_running:
                            break
                        batch_num += 1
                        # Gradient-Akkumulation: der Batch wird in Mikro-Batches
                        # geteilt, zero_grad/step passieren einmal pro Batch. Als
                        # Anteil zaehlt die Gewichtsmasse (nicht die Beispielzahl),
                        # damit die Gradienten exakt die des grossen Batches sind.
                        masses = (per_sample_weight_mass(batch["voxel_ids"], batch)
                                  if micro_batch < batch_size else None)
                        optimizer.zero_grad(set_to_none=True)
                        batch_loss = 0.0
                        for micro, share in iter_micro_batches(batch, micro_batch,
                                                               sample_mass=masses):
                            prompt_ids = micro["prompt_ids"].to(device)
                            target = micro["voxel_ids"].to(device).reshape(prompt_ids.shape[0], -1)
                            target = model.safe_clamp_target(target)
                            if dit_encoder is not None:
                                with torch.no_grad():
                                    encoded = dit_encoder(list(micro["prompt_text"]))
                                    prompt_embeddings = encoded["last_hidden_state"].to(
                                        device=device, dtype=next(model.parameters()).dtype)
                                    prompt_mask = encoded["attention_mask"].to(device)
                                model_inputs = {"prompt_embeddings": prompt_embeddings,
                                                "prompt_attention_mask": prompt_mask}
                            else:
                                model_inputs = {"prompt_ids": prompt_ids}
                            sample_weight = micro["sample_weight"].to(device).view(-1, 1).expand_as(target)
                            per_block_w = micro["per_block_weight"].to(device).view(-1, 1).expand_as(target)
                            per_air_w = micro["per_air_weight"].to(device).view(-1, 1).expand_as(target)
                            weight_per_token = torch.where(target == 0, per_air_w, per_block_w) * sample_weight
                            loss = model.backward_weighted_loss(
                                target, weight_per_token, chunk_voxels=chunk_voxels,
                                grad_scale=share, **model_inputs,
                            )
                            batch_loss += loss * share
                        apply_optimizer_step(model, optimizer)
                        total_loss += batch_loss
                        if batch_num % max(1, total_batches // 10) == 0 or batch_num == total_batches:
                            self.after(0, lambda bp=batch_num/total_batches, bn=batch_num, tb=total_batches: (
                                self.train_batch_bar.set(bp), self.train_batch_label.configure(text=self._tr("batch_progress", bn=bn, tb=tb))))
                    avg_loss = total_loss / max(total_batches, 1)
                    done_epoch = start_epoch + epoch
                    total_epoch = start_epoch + epochs
                    self.after(0, lambda e=done_epoch, tot=total_epoch, l=avg_loss,
                                      p=done_epoch/total_epoch: (
                        self.train_epoch_bar.set(p),
                        self.train_epoch_label.configure(text=self._tr("transformer_epoch_progress", e=e, epochs=tot, loss=f"{l:.4f}"))))
                    torch.save({"model_state": model.state_dict(), "grid_size": grid_size,
                                "text_vocab_size": len(dataset.prompt_tokenizer.token_to_id),
                                "block_vocab_size": len(dataset.voxel_tokenizer.id_to_block),
                                "d_model": model.d_model, "nhead": (model.decoder_layer if model.decoder is None else model.decoder.layers[0]).self_attn.num_heads,
                                "layers": len(model.decoder.layers) if model.decoder is not None else num_layers,
                                "dim_feedforward": model.decoder.layers[0].linear1.out_features if model.decoder is not None else dim_ff,
                                "recurrent_depth": model.recurrent_depth,
                                "window_size": model.window_size,
                                "tread_ratio": model.tread_ratio,
                                "ssa_topk": model.ssa_topk,
                                "voxel_mask_ratio": model.voxel_mask_ratio,
                                "moe_experts": model.moe_experts, "moe_active": model.moe_active,
                                "text_encoder_name": model.text_encoder_name,
                                "text_input_dim": model.text_input_dim,
                                "augmentation_diversity": augmentation_diversity,
                                "allow_vertical_movement": allow_vertical_movement,
                                "epoch": done_epoch, "loss": avg_loss}, out_dir / "model.pt")
                self.after(0, lambda: self.status_var.set(self._tr("status_transformer_done")))

            elif model_type == "diffusion":
                if continuing_model:
                    # Fortsetzen: derselbe Ordner, dieselben Gewichte/dasselbe Vokabular
                    out_dir = RUNS_DIR / continuing_name
                else:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    out_dir = RUNS_DIR / f"diffusion_{timestamp}"
                out_dir.mkdir(parents=True, exist_ok=True)
                if continuing_model:
                    model = self.diffusion_model
                else:
                    # Diffusion steps for the new model (snapshotted in the main thread)
                    diff_num_timesteps = int(params.get("diffusion_steps", 50))
                    model = VoxelDiffusionModel(
                        num_blocks=len(dataset.voxel_tokenizer.id_to_block),
                        text_vocab_size=len(dataset.prompt_tokenizer.token_to_id),
                        grid_size=grid_size, d_model=128, d_text=64, channels=64,
                        channel_multipliers=(1, 2, 2), num_timesteps=diff_num_timesteps,
                    ).to(device)
                    dataset.prompt_tokenizer.save(out_dir / "prompt_vocab.json")
                    dataset.voxel_tokenizer.save(out_dir / "block_vocab.json")

                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
                total_start = time.time()
                noise_block_prob = getattr(self.config, 'noise_block_prob', 0.20)
                for epoch in range(1, epochs + 1):
                    if not self.training_running:
                        break
                    model.train()
                    total_loss = 0.0
                    batch_num = 0
                    for batch in loader:
                        if not self.training_running:
                            break
                        batch_num += 1
                        # Gradient-Akkumulation wie im TF-Diffusion-Worker: der
                        # Batch wird in Mikro-Batches geteilt, der Optimizer
                        # schreitet einmal pro Batch. Der Anteil kommt aus der
                        # Gewichtsmasse (siehe app/grad_accum.py).
                        accumulating = micro_batch < batch_size
                        masses = (per_sample_weight_mass(batch["voxel_ids"], batch)
                                  if accumulating else None)
                        if accumulating:
                            optimizer.zero_grad(set_to_none=True)
                        batch_loss = 0.0
                        for micro, share in iter_micro_batches(batch, micro_batch,
                                                               sample_mass=masses):
                            batch_loss += train_diffusion_step(
                                model, micro, optimizer,
                                device, noise_block_prob=noise_block_prob,
                                chunk_voxels=chunk_voxels,
                                accumulate=accumulating, grad_scale=share) * share
                        if accumulating:
                            apply_optimizer_step(model, optimizer)
                        total_loss += batch_loss
                        if batch_num % max(1, total_batches // 10) == 0 or batch_num == total_batches:
                            self.after(0, lambda bp=batch_num/total_batches, bn=batch_num, tb=total_batches: (
                                self.train_batch_bar.set(bp), self.train_batch_label.configure(text=self._tr("batch_progress", bn=bn, tb=tb))))
                    avg_loss = total_loss / max(total_batches, 1)
                    done_epoch = start_epoch + epoch
                    total_epoch = start_epoch + epochs
                    self.after(0, lambda e=done_epoch, tot=total_epoch, l=avg_loss,
                                      p=done_epoch/total_epoch: (
                        self.train_epoch_bar.set(p),
                        self.train_epoch_label.configure(text=self._tr("diffusion_epoch_progress", e=e, epochs=tot, loss=f"{l:.4f}"))))
                    torch.save({"model_state": model.state_dict(), "grid_size": model.grid_size,
                                "text_vocab_size": len(dataset.prompt_tokenizer.token_to_id),
                                "block_vocab_size": model.num_blocks, "num_blocks": model.num_blocks,
                                "d_model": model.d_model, "d_text": model.d_text, "channels": model.channels,
                                "channel_multipliers": [int(m) for m in model.channel_multipliers],
                                "num_timesteps": model.num_timesteps,
                                "augmentation_diversity": augmentation_diversity,
                                "allow_vertical_movement": allow_vertical_movement,
                                "epoch": done_epoch, "loss": avg_loss}, out_dir / "model.pt")
                self.after(0, lambda: self.status_var.set(self._tr("status_diffusion_done")))

            self.after(0, self._discover_models)
            self.after(0, self._refresh_models_tab)
            self.after(0, self._refresh_model_combo)
            self.after(0, self._load_models_async)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.after(0, lambda msg=str(e)[:100]: self.train_loss_var.set(self._tr("training_error", msg=msg)))
        finally:
            self.training_running = False
            self.after(0, lambda: self.train_btn_dit.configure(state="normal"))
            self.after(0, lambda: self.train_btn_tf_diffusion.configure(state="normal"))
            self.after(0, lambda: self.train_stop_btn.configure(state="disabled"))
            # Siehe _tf_diffusion_training_worker: ohne das bleibt der VRAM-Peak
            # des Trainings im Caching-Allocator reserviert und die Generierung
            # meldet danach "Cuda out of memory".
            release_cuda_cache()
            self.after(200, release_cuda_cache)

    def _export_kaggle_dialog(self):
        from tkinter.messagebox import askyesno, showinfo
        epochs = self.train_epochs_entry.get().strip()
        batch_size = self.train_batch_entry.get().strip()
        lr = self.train_lr_entry.get().strip()
        aug_diversity = int(round(float(self.augmentation_diversity_var.get())))
        allow_vertical = bool(self.setting_aug_vertical.get())
        grid_size = self._get_selected_grid_size()
        gx, gy, gz = grid_size
        vertical_str = "Ja" if allow_vertical else "Nein" if self.config.language == "de" else "Yes" if allow_vertical else "No"
        msg = self._tr("kaggle_msg", gx=gx, gy=gy, gz=gz, epochs=epochs, batch=batch_size,
                       lr=lr, aug=aug_diversity, vertical=vertical_str, count=len(self.data_dirs))
        if not askyesno(self._tr("kaggle_title"), msg):
            return
        self.status_var.set(self._tr("status_kaggle_creating"))
        self.kaggle_export_btn.configure(state="disabled")
        # Snapshot every UI value here (main thread) and hand it to the worker.
        params = self._collect_training_params()
        params["grid_size"] = grid_size
        params["augmentation_diversity"] = aug_diversity
        params["allow_vertical_movement"] = allow_vertical
        params["data_dirs"] = list(self.data_dirs)
        params["model_type"] = {
            "Transformer": "transformer",
            "DiT (Recurrent)": "transformer",
            "Diffusion": "diffusion",
            "TF-Diffusion": "transformer_diffusion",
        }.get(self.train_model_type_selector.get(), "transformer")
        params["dit_enabled"] = self.train_model_type_selector.get() == "DiT (Recurrent)"
        params["architecture"] = dict(getattr(self, "_selected_arch", None) or {})
        params["architecture"]["window_size"] = self._get_attention_window_size()
        if self.train_model_type_selector.get() == "DiT (Recurrent)":
            selected_encoder = self.dit_text_encoder_var.get()
            params["dit_text_encoder_name"] = (
                None if selected_encoder == "Learned embeddings" else selected_encoder)
            params["architecture"].update({
                "recurrent_depth": max(1, int(self.dit_depth_var.get())),
                "moe_experts": max(0, int(self.moe_experts_var.get())) if self.moe_enabled_var.get() else 0,
                "moe_active": max(1, int(self.moe_active_var.get())),
            })
        else:
            for key in ("recurrent_depth", "moe_experts", "moe_active"):
                params["architecture"].pop(key, None)
        params["tf_unet_config"] = getattr(self, "_tf_unet_config", None)
        if params["model_type"] == "transformer" and params["dit_enabled"]:
            dit_encoder_name = params.get("dit_text_encoder_name")
            params["encoder"] = ((dit_encoder_name,
                                  ENCODER_HIDDEN_DIMS.get(dit_encoder_name, DEFAULT_ENCODER_HIDDEN_DIM))
                                 if dit_encoder_name else (None, None))
        else:
            params["encoder"] = self._describe_selected_encoder()
        threading.Thread(target=self._kaggle_export_worker, args=(params,), daemon=True).start()

    def _describe_selected_encoder(self) -> Optional[Tuple[str, int]]:
        """(display_name, hidden_dim) of the encoder used for a TF-Diffusion export."""
        if self.tf_encoder is not None:
            return (self.tf_encoder.display_name, int(self.tf_encoder.hidden_dim))
        try:
            name = self.tf_encoder_combo.get()
        except (TypeError, ValueError, AttributeError):
            return None
        if not name:
            return None
        return (name, ENCODER_HIDDEN_DIMS.get(name, DEFAULT_ENCODER_HIDDEN_DIM))

    def _kaggle_export_worker(self, params: Dict[str, Any]):
        from tkinter.messagebox import showinfo, showerror
        try:
            epochs = int(params["epochs"])
            batch_size = int(params["batch_size"])
            lr = float(params["lr"])
            aug_diversity = int(params["augmentation_diversity"])
            allow_vertical = bool(params["allow_vertical_movement"])
            grid_size = tuple(params["grid_size"])
            air_weight = float(params["air_weight"])
            model_type = str(params["model_type"])
            arch = params.get("architecture")
            tf_unet_config = params.get("tf_unet_config")
            enc_name, ctx_dim = (params.get("encoder") or (None, None))
            from kaggle_export import create_kaggle_export
            export_path = create_kaggle_export(output_dir=str(EXPORTS_DIR), epochs=epochs, batch_size=batch_size,
                                               learning_rate=lr, aug_diversity=aug_diversity,
                                               allow_vertical=allow_vertical, grid_size=grid_size,
                                               model_type=model_type, data_dirs=params.get("data_dirs"),
                                               architecture=arch, tf_unet_config=tf_unet_config,
                                               air_weight=air_weight,
                                               encoder_name=enc_name, context_dim=ctx_dim)
            self.after(0, lambda p=export_path: self.status_var.set(self._tr("status_kaggle_done", name=p.name)))
            self.after(0, lambda p=export_path: showinfo(
                self._tr("kaggle_success_title"),
                self._tr("kaggle_success_msg", path=p)))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.after(0, lambda err=str(e): self.status_var.set(self._tr("status_kaggle_failed", err=err[:80])))
            self.after(0, lambda err=str(e): showerror(self._tr("kaggle_failed_title"), str(err)))
        finally:
            self.after(0, lambda: self.kaggle_export_btn.configure(state="normal"))


def prompt_text_from_txt(proj_dir: Path) -> str:
    txt_path = proj_dir / "prompt.txt"
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()
    return "Unbekannt"


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    app = MinecraftStructureApp()
    app.mainloop()


if __name__ == "__main__":
    main()
