"""Minecraft Structure Generator - Complete Shippable Application.
CustomTkinter-based multi-tab app with 3D preview, model switching,
project management, settings, training tab, and Model Manager.

Main focus: Transformer Diffusion model with frozen pre-trained text encoders
(Phi-3.5, Gemma 2/3, T5) and cross-attention."""
from __future__ import annotations

import json
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
from app.diffusion_model import (
    VoxelDiffusionModel,
    TransformerDiffusionModel,
    train_diffusion_step,
    train_transformer_diffusion_step,
    train_transformer_diffusion_step_cached,
)
from app.model_manager import ModelRegistry, ModelEntry
from app.transformer_encoder import FrozenTransformerEncoder, list_supported_models, MODEL_NAMES
from app.hidden_state_cache import (
    compute_hidden_states, load_hidden_states, validate_cache, delete_cache, list_caches,
)
from app.voxel_preview import render_preview
from app.voxel_viewer_3d import open_3d_viewer, HAS_PYGLET
from app.translations import get_text as _get_text

from dataset import PromptTokenizer, VoxelTokenizer, center_token_grid, load_schematic, save_schem, trim_token_grid
from model import SharedWeightVoxelTransformer
from kaggle_export import create_kaggle_export

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

# ─── Constants ───
APP_NAME = "Minecraft Structure Generator"
VERSION = "2.1.0"
PROJECTS_DIR = Path("projects")
EXPORTS_DIR = Path("exports")

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

# ─── Transformer Diffusion UNet presets ───
# Each preset: (display_name, channels, channel_multipliers, d_model, cross_attn_heads)
TF_DIFFUSION_PRESETS = [
    ("🐣 Tiny (0.5M)", 16, (1, 2, 2), 32, 2),
    ("🔹 Small (1.5M)", 32, (1, 2, 2), 64, 4),
    ("🔶 Medium (4.5M)", 48, (1, 2, 3), 96, 4),
    ("🔴 Large (12M)", 64, (1, 2, 3, 4), 128, 6),
    ("💎 XL (35M)", 96, (1, 2, 3, 4), 192, 8),
    ("🚀 XXL (80M)", 128, (1, 2, 3, 4, 4), 256, 8),
]

# Grid size options shared across all model types
GRID_SIZE_OPTIONS = [
    "16×16×16",
    "32×32×32",
    "48×48×48",
    "64×64×64",
    "96×96×96（experimentell）",
    "128×128×128（experimentell）",
    "256×256×256（experimentell）",
]

GRID_SIZE_MAP = {}
for opt in GRID_SIZE_OPTIONS:
    # Split on × and take first 3 numeric parts
    parts = opt.replace("（", "×").split("×")[:3]
    GRID_SIZE_MAP[opt] = tuple(int(p) for p in parts)


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
        self.model_type: str = "transformer_diffusion"
        self.transformer_model: Optional[SharedWeightVoxelTransformer] = None
        self.diffusion_model: Optional[VoxelDiffusionModel] = None
        self.tf_diffusion_model: Optional[TransformerDiffusionModel] = None
        self.tf_encoder: Optional[FrozenTransformerEncoder] = None
        self.prompt_tokenizer: Optional[PromptTokenizer] = None
        self.voxel_tokenizer: Optional[VoxelTokenizer] = None
        self.current_transformer_name: Optional[str] = None
        self.current_diffusion_name: Optional[str] = None
        self.current_tf_diffusion_name: Optional[str] = None
        self.data_dirs = [
            "Trainingsdaten good thoroughly analyzed",
            "Trainingsdaten zu gross vorerst ausgelagert",
        ]
        self.generation_running = False
        self.training_running = False
        self._current_training_type: Optional[str] = None

        # ─── Build UI ───
        self._build_ui()
        self._setup_tab_icons()
        self._apply_minecraft_skin()

        # ─── Load default models ───
        self.after(100, self._load_models_async)

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

        ctk.CTkLabel(header, text="▣", width=44, font=self._mc_font(30, "bold"),
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
            model_frame, values=["Transformer", "Diffusion", "TF-Diffusion"],
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
        ctk.CTkLabel(orbit_frame, text="🔍", font=("Segoe UI", 12)).grid(row=0, column=4, padx=(8, 2))
        self.zoom_slider = ctk.CTkSlider(orbit_frame, from_=1.5, to=6.0, number_of_steps=45, width=80)
        self.zoom_slider.grid(row=0, column=5, padx=(0, 4))
        self.zoom_slider.set(self.zoom_scale)
        self.zoom_slider.configure(command=lambda v: self._update_zoom(v))

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
        # Transformers
        ctk.CTkLabel(self.models_container, text=self._tr("transformer_models"),
                     font=("Segoe UI", 18, "bold"), text_color="#60a5fa").grid(row=row, column=0, sticky="w", pady=(0, 8))
        row += 1
        transformers = self.model_registry.get_by_type("transformer")
        if not transformers:
            ctk.CTkLabel(self.models_container, text=self._tr("no_transformer"),
                         font=("Segoe UI", 12), text_color=("gray60", "gray50")).grid(row=row, column=0, sticky="w", pady=(0, 12))
            row += 1
        else:
            for entry in transformers:
                self._build_model_card(self.models_container, row, entry)
                row += 1
        sep2 = ctk.CTkFrame(self.models_container, height=2, fg_color=("gray70", "gray30"))
        sep2.grid(row=row, column=0, sticky="ew", pady=16)
        row += 1
        # Diffusions
        ctk.CTkLabel(self.models_container, text=self._tr("diffusion_models"),
                     font=("Segoe UI", 18, "bold"), text_color="#a78bfa").grid(row=row, column=0, sticky="w", pady=(0, 8))
        row += 1
        diffusions = self.model_registry.get_by_type("diffusion")
        if not diffusions:
            ctk.CTkLabel(self.models_container, text=self._tr("no_diffusion"),
                         font=("Segoe UI", 12), text_color=("gray60", "gray50")).grid(row=row, column=0, sticky="w", pady=(0, 12))
            row += 1
        else:
            for entry in diffusions:
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
        name_text = f"📦 {entry.name}"
        if is_default:
            name_text += self._tr("default_badge")
        ctk.CTkLabel(hdr, text=name_text, font=("Segoe UI", 15, "bold")).grid(row=0, column=0, sticky="w")
        badge_color = "#3b82f6" if entry.model_type == "transformer" else "#8b5cf6"
        if entry.model_type == "transformer_diffusion":
            badge_color = "#c084fc"
        ctk.CTkLabel(hdr, text=f" {entry.model_type.upper()} ", font=("Segoe UI", 11, "bold"),
                     fg_color=badge_color, text_color="white", corner_radius=4, padx=8).grid(row=0, column=1, padx=4)
        meta = ctk.CTkFrame(card, fg_color="transparent")
        meta.grid(row=1, column=0, sticky="ew", padx=14, pady=2)
        meta.grid_columnconfigure(0, weight=1)
        grid_str = f"{entry.grid_size[0]}×{entry.grid_size[1]}×{entry.grid_size[2]}"
        bl = self._tr("blocks_label")
        if entry.model_type == "transformer":
            info = (f"d_model={entry.d_model}  heads={entry.nhead}  layers={entry.num_layers}  "
                    f"FFN={entry.dim_feedforward}  {bl}={entry.block_vocab_size}")
        elif entry.model_type == "transformer_diffusion":
            info = (f"d_model={entry.d_model}  channels={entry.channels}  steps={entry.num_timesteps}  "
                    f"encoder={entry.encoder_name or '?'}  {bl}={entry.block_vocab_size}")
        else:
            info = (f"d_model={entry.d_model}  channels={entry.channels}  steps={entry.num_timesteps}  "
                    f"{bl}={entry.block_vocab_size}")
        ctk.CTkLabel(meta, text=f"📐 {grid_str}  🧱 {info}", font=("Segoe UI", 11),
                     text_color=("gray50", "gray40")).grid(row=0, column=0, sticky="w")
        if entry.epochs_trained > 0:
            ctk.CTkLabel(meta, text=f"📊 {entry.epochs_trained} {self._tr('epochs_label')}  Loss={entry.last_loss:.4f}",
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
            self.model_type = "transformer"
            self.current_transformer_name = name
            self.model_type_selector.set("Transformer")
        elif entry.model_type == "diffusion":
            self.model_type = "diffusion"
            self.current_diffusion_name = name
            self.model_type_selector.set("Diffusion")
        elif entry.model_type == "transformer_diffusion":
            self.model_type = "transformer_diffusion"
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
        dialog = ctk.CTkToplevel(self)
        dialog.title(self._tr("train_more_title", name=name))
        dialog.geometry("480x500")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 480) // 2
        y = self.winfo_y() + (self.winfo_height() - 500) // 2
        dialog.geometry(f"+{x}+{y}")
        dialog.grid_columnconfigure(0, weight=1)
        self._safe_configure(dialog, fg_color=MC_BG)

        header = ctk.CTkFrame(dialog, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 12))
        header.grid_columnconfigure(0, weight=1)
        icon = {"transformer": "⚡", "diffusion": "🌀", "transformer_diffusion": "🤖"}.get(entry.model_type, "📦")
        ctk.CTkLabel(header, text=self._tr("train_more_header", icon=icon, name=name), font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")

        info_frame = ctk.CTkFrame(dialog, fg_color=("gray90", "gray15"), corner_radius=6)
        info_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 16))
        info_frame.grid_columnconfigure(0, weight=1)
        gs = f"{entry.grid_size[0]}×{entry.grid_size[1]}×{entry.grid_size[2]}"
        info_text = f"📐 {gs}  |  🏷️ {entry.model_type.upper()}"
        if entry.epochs_trained > 0:
            info_text += f"\n📊 {entry.epochs_trained} {self._tr('epochs_label')}  |  Loss: {entry.last_loss:.4f}"
        ctk.CTkLabel(info_frame, text=info_text, font=("Segoe UI", 11), justify="left").grid(row=0, column=0, sticky="w", padx=14, pady=10)

        params = ctk.CTkFrame(dialog, fg_color="transparent")
        params.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 12))
        params.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(params, text=self._tr("params_label"), font=("Segoe UI", 13, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ctk.CTkLabel(params, text=self._tr("tm_epochs"), font=("Segoe UI", 12)).grid(row=1, column=0, sticky="w", padx=(0, 12), pady=4)
        epochs_var = ctk.StringVar(value="10")
        ctk.CTkEntry(params, width=120, textvariable=epochs_var, font=("Segoe UI", 12)).grid(row=1, column=1, sticky="w", pady=4)
        ctk.CTkLabel(params, text=self._tr("tm_batch_size"), font=("Segoe UI", 12)).grid(row=2, column=0, sticky="w", padx=(0, 12), pady=4)
        batch_var = ctk.StringVar(value="4")
        ctk.CTkEntry(params, width=120, textvariable=batch_var, font=("Segoe UI", 12)).grid(row=2, column=1, sticky="w", pady=4)
        ctk.CTkLabel(params, text=self._tr("tm_learning_rate"), font=("Segoe UI", 12)).grid(row=3, column=0, sticky="w", padx=(0, 12), pady=4)
        lr_var = ctk.StringVar(value="1.5e-3")
        ctk.CTkEntry(params, width=120, textvariable=lr_var, font=("Segoe UI", 12)).grid(row=3, column=1, sticky="w", pady=4)

        ctk.CTkLabel(params, text=self._tr("tm_diversity"), font=("Segoe UI", 12)).grid(row=4, column=0, sticky="w", padx=(0, 12), pady=4)
        aug_var = ctk.DoubleVar(value=1)
        aug_value = ctk.CTkLabel(params, text="1", width=28)
        ctk.CTkSlider(params, from_=0, to=5, number_of_steps=5, variable=aug_var,
                      command=lambda v: aug_value.configure(text=str(int(round(float(v)))))).grid(row=4, column=1, sticky="ew", pady=4)
        aug_value.grid(row=4, column=2, sticky="w", padx=(8, 0), pady=4)
        aug_vertical_switch = ctk.CTkSwitch(params, text=self._tr("allow_vertical"), onvalue=True, offvalue=False)
        aug_vertical_switch.grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))

        gpu_avail = torch.cuda.is_available() and self.config.gpu_enabled
        gpu_text = f"GPU: {'✅ CUDA verfügbar' if gpu_avail else '❌ CPU only'}" if self.config.language == "de" else f"GPU: {'✅ CUDA available' if gpu_avail else '❌ CPU only'}"
        ctk.CTkLabel(params, text=gpu_text,
                     font=("Segoe UI", 11), text_color="#34d399" if gpu_avail else "#f87171",
                     ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(8, 0))

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(8, 20))
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)

        def on_start():
            try:
                epochs = int(epochs_var.get().strip())
                batch = int(batch_var.get().strip())
                lr = float(lr_var.get().strip())
                aug = int(round(float(aug_var.get())))
                allow_vertical = bool(aug_vertical_switch.get())
                if epochs <= 0 or batch <= 0 or lr <= 0 or aug < 0:
                    raise ValueError(self._tr("values_must_be_positive"))
            except (ValueError, TypeError):
                self.status_var.set(self._tr("status_invalid_params"))
                return
            dialog.destroy()
            self._load_single_model(name)
            # Apply parameters to training tab
            self.train_epochs_entry.delete(0, "end")
            self.train_epochs_entry.insert(0, str(epochs))
            self.train_batch_entry.delete(0, "end")
            self.train_batch_entry.insert(0, str(batch))
            self.train_lr_entry.delete(0, "end")
            self.train_lr_entry.insert(0, str(lr))
            self.augmentation_diversity_var.set(aug)
            self._update_aug_label(aug)
            if allow_vertical:
                self.setting_aug_vertical.select()
            else:
                self.setting_aug_vertical.deselect()
            self.tab_view.set(self._tr("tab_training"))
            self.after(300, lambda: self._start_training(entry.model_type))

        ctk.CTkButton(btn_frame, text=self._tr("btn_start_training"), font=("Segoe UI", 13, "bold"),
                      height=36, fg_color="#7c3aed", hover_color="#6d28d9", command=on_start,
                     ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkButton(btn_frame, text=self._tr("btn_cancel"), font=("Segoe UI", 13), height=36,
                      fg_color="gray40", hover_color="gray50", command=dialog.destroy,
                     ).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self._skin_widget_tree(dialog)

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
            type_frame, values=["Transformer", "Diffusion", "TF-Diffusion"],
            command=self._on_train_type_change, font=("Segoe UI", 12),
        )
        self.train_model_type_selector.grid(row=0, column=1, sticky="w")

        # ─── Container for model-specific panels ───
        self.train_panels_container = ctk.CTkFrame(scroll, fg_color="transparent")
        self.train_panels_container.grid(row=row, column=0, sticky="ew")
        self.train_panels_container.grid_columnconfigure(0, weight=1)

        # ─── Build all three panels (only one shown at a time) ───
        self._build_tf_diffusion_train_panel(self.train_panels_container)
        self._build_transformer_train_panel(self.train_panels_container)
        self._build_diffusion_train_panel(self.train_panels_container)

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
        self.train_batch_entry.insert(0, "4")

        ctk.CTkLabel(param_grid, text=self._tr("learning_rate")).grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(4, 0))
        self.train_lr_entry = ctk.CTkEntry(param_grid, width=80)
        self.train_lr_entry.grid(row=1, column=1, sticky="w", pady=(4, 0))
        self.train_lr_entry.insert(0, "1.5e-3")

        ctk.CTkLabel(param_grid, text=self._tr("aug_diversity")).grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        self.augmentation_diversity_var = ctk.DoubleVar(value=1)
        self.augmentation_diversity_slider = ctk.CTkSlider(
            param_grid, from_=0, to=5, number_of_steps=5,
            variable=self.augmentation_diversity_var, command=self._update_aug_label,
        )
        self.augmentation_diversity_slider.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(8, 0))
        self.augmentation_diversity_label = ctk.CTkLabel(param_grid, text="1", width=28)
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

        self.train_btn_transformer = ctk.CTkButton(
            btn_frame, text=self._tr("btn_train_transformer"), font=("Segoe UI", 12, "bold"),
            command=lambda: self._start_training("transformer"))
        self.train_btn_transformer.grid(row=0, column=0, padx=(0, 8))

        self.train_btn_diffusion = ctk.CTkButton(
            btn_frame, text=self._tr("btn_train_diffusion"), font=("Segoe UI", 12, "bold"),
            fg_color="#7c3aed", hover_color="#6d28d9",
            command=lambda: self._start_training("diffusion"))
        self.train_btn_diffusion.grid(row=0, column=1, padx=(0, 8))

        self.train_btn_tf_diffusion = ctk.CTkButton(
            btn_frame, text=self._tr("btn_train_tf_diffusion"), font=("Segoe UI", 12, "bold"),
            fg_color="#c084fc", hover_color="#a855f7",
            command=lambda: self._start_tf_diffusion_training())
        self.train_btn_tf_diffusion.grid(row=0, column=2, padx=(0, 8))

        self.train_stop_btn = ctk.CTkButton(
            btn_frame, text=self._tr("btn_stop_save"), font=("Segoe UI", 12, "bold"),
            fg_color="#dc2626", hover_color="#b91c1c", state="disabled",
            command=self._stop_training)
        self.train_stop_btn.grid(row=0, column=3, padx=(8, 0))

        self.kaggle_export_btn = ctk.CTkButton(
            btn_frame, text=self._tr("btn_kaggle_export"), font=("Segoe UI", 12, "bold"),
            fg_color="#dc2626", hover_color="#b91c1c",
            command=self._export_kaggle_dialog)
        self.kaggle_export_btn.grid(row=0, column=4, padx=(8, 0))

        # Show default panel and TF-Diffusion button after full construction
        self.after(0, lambda: (
            self.train_btn_diffusion.grid_remove(),
            self.train_btn_transformer.grid_remove(),
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
        ctk.CTkButton(cache_sel_frame, text="🔄", font=("Segoe UI", 10),
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
        self.tf_preset_var = ctk.StringVar(value="🐣 Tiny (0.5M)")
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
        for i, (label, val) in enumerate([("🐣 Tiny (1.7M)", 1.7), ("🔹 Small (3.9M)", 3.9),
                                          ("🔶 Medium (7.5M)", 7.5), ("🔴 Large (20M)", 20),
                                          ("💎 XL (45M)", 45), ("🚀 XXL (117M)", 117)]):
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
        ctk.CTkButton(self.advanced_frame, text=self._tr("btn_calculate"), font=("Segoe UI", 10), width=100, height=26,
                      command=self._update_arch_from_advanced).grid(row=4, column=0, padx=8, pady=4, sticky="w")

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
        for panel_name in ['tf_panel', 'transformer_panel', 'diffusion_panel']:
            panel = getattr(self, panel_name, None)
            if panel:
                panel.grid_remove()
        # Hide all training buttons
        self.train_btn_tf_diffusion.grid_remove()
        self.train_btn_transformer.grid_remove()
        self.train_btn_diffusion.grid_remove()
        # Show selected panel and matching button
        if value == "TF-Diffusion":
            if hasattr(self, 'tf_panel'):
                self.tf_panel.grid()
            self.train_btn_tf_diffusion.grid()
        elif value == "Transformer":
            if hasattr(self, 'transformer_panel'):
                self.transformer_panel.grid()
            self.train_btn_transformer.grid()
        elif value == "Diffusion":
            if hasattr(self, 'diffusion_panel'):
                self.diffusion_panel.grid()
            self.train_btn_diffusion.grid()

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
            device = torch.device("cuda" if torch.cuda.is_available() and self.config.gpu_enabled else "cpu")
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            encoder = FrozenTransformerEncoder(model_name=model_name, device=device, dtype=dtype)
            self.tf_encoder = encoder
            self.after(0, lambda: self.tf_encoder_status.configure(
                text=self._tr("encoder_loaded_info", name=model_name, dim=encoder.hidden_dim)))
            self.after(0, lambda: self.status_var.set(self._tr("status_encoder_loaded", name=model_name)))
        except Exception as exc:
            err_msg = str(exc)
            self.after(0, lambda m=err_msg: self.tf_encoder_status.configure(
                text=self._tr("cache_error", msg=m[:60])))
            self.after(0, lambda m=err_msg: self.status_var.set(self._tr("status_encoder_error", msg=m[:40])))

    def _start_tf_diffusion_training(self):
        if self.training_running:
            return
        use_cached = self.tf_use_cached_hs_var.get()
        if not use_cached and self.tf_encoder is None:
            self.status_var.set(self._tr("status_no_encoder"))
            return
        if use_cached:
            cache_sel = self.tf_cache_selector.get()
            if cache_sel == self._tr("no_cache"):
                self.status_var.set(self._tr("status_no_cache"))
                return
            # Validate cache exists
            encoder_name = cache_sel
            schem_files, txt_files = self._get_training_schem_files()
            result = validate_cache(encoder_name, schem_files, txt_files)
            if not result["valid"]:
                self.status_var.set(self._tr("status_cache_invalid", msg=result['message']))
                return
        self.training_running = True
        self.tf_progress_bar.set(0)
        self.train_btn_tf_diffusion.configure(state="disabled")
        self.train_stop_btn.configure(state="normal")
        mode = self._tr("with_cached_hs") if use_cached else self._tr("with_encoder")
        self.status_var.set(self._tr("status_tf_training_started", mode=mode))
        threading.Thread(target=self._tf_diffusion_training_worker, daemon=True).start()

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

    def _tf_diffusion_training_worker(self):
        try:
            epochs = int(self.train_epochs_entry.get())
            batch_size = int(self.train_batch_entry.get())
            lr = float(self.train_lr_entry.get())
            num_timesteps = int(self.tf_diff_steps_entry.get())
            grid_size = self._get_selected_grid_size()
            aug_diversity = int(round(float(self.augmentation_diversity_var.get())))
            allow_vertical = bool(self.setting_aug_vertical.get())
            air_weight = float(self.air_weight_var.get()) if hasattr(self, 'air_weight_var') else 75.0
            device = torch.device("cuda" if torch.cuda.is_available() and self.config.gpu_enabled else "cpu")
            config = self._tf_unet_config
            use_cached = self.tf_use_cached_hs_var.get()

            from dataset import MultiSourceSchematicDataset
            from app.hidden_state_cache import CACHE_DIR, cache_key, load_hidden_states_raw

            if use_cached:
                # ── Use pre-computed hidden states ──
                encoder_name = self.tf_cache_selector.get()
                key = cache_key(encoder_name)
                cache_dir = CACHE_DIR / key

                # Load cache metadata to get encoder info
                cache_data = load_hidden_states_raw(cache_dir)
                cache_meta = cache_data["metadata"]
                hs = cache_data["hidden_states"]  # [N, seq_len, hidden_dim]
                context_dim = hs.shape[2]

                # Build dataset with cache
                dataset = MultiSourceSchematicDataset.with_cache(
                    self.data_dirs, cache_dir, target_size=grid_size, max_voxels=400_000,
                    augmentation_diversity=aug_diversity,
                    allow_vertical_movement=allow_vertical,
                    air_weight_factor=air_weight,
                )
                from torch.utils.data import DataLoader
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                total_batches = len(loader)

                model = TransformerDiffusionModel(
                    num_blocks=len(dataset.voxel_tokenizer.id_to_block),
                    grid_size=grid_size,
                    d_model=config["d_model"],
                    channels=config["channels"],
                    channel_multipliers=tuple(config["channel_multipliers"]),
                    num_timesteps=num_timesteps,
                    context_dim=context_dim,
                    cross_attn_heads=config["cross_attn_heads"],
                    context_proj_dim=config["d_model"] * 2,
                ).to(device)

                # Build encoder_config from cache metadata
                encoder_config = {
                    "display_name": cache_meta.get("encoder_name", encoder_name),
                    "hf_id": cache_meta.get("encoder_hf_id", ""),
                    "hidden_dim": context_dim,
                }

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_enc_name = encoder_name.replace("/", "_").replace(" ", "_")
                out_dir = Path("runs") / f"tf_diffusion_cached_{safe_enc_name}_{timestamp}"

            else:
                # ── Use live encoder ──
                if self.tf_encoder is None:
                    raise RuntimeError("Encoder not loaded")
                context_dim = self.tf_encoder.hidden_dim

                dataset = MultiSourceSchematicDataset(
                    self.data_dirs, target_size=grid_size, max_voxels=400_000,
                    augmentation_diversity=aug_diversity,
                    allow_vertical_movement=allow_vertical,
                    air_weight_factor=air_weight,
                )
                from torch.utils.data import DataLoader
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                total_batches = len(loader)

                model = TransformerDiffusionModel(
                    num_blocks=len(dataset.voxel_tokenizer.id_to_block),
                    grid_size=grid_size,
                    d_model=config["d_model"],
                    channels=config["channels"],
                    channel_multipliers=tuple(config["channel_multipliers"]),
                    num_timesteps=num_timesteps,
                    context_dim=context_dim,
                    cross_attn_heads=config["cross_attn_heads"],
                    context_proj_dim=config["d_model"] * 2,
                ).to(device)

                encoder_config = self.tf_encoder.get_config()
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                encoder_name = self.tf_encoder.display_name.replace("/", "_").replace(" ", "_")
                out_dir = Path("runs") / f"tf_diffusion_{encoder_name}_{timestamp}"

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

                    if use_cached:
                        # Use cached training step (no encoder needed)
                        loss = train_transformer_diffusion_step_cached(model, batch, optimizer, device, noise_block_prob=noise_block_prob)
                    else:
                        # Use live encoder
                        loss = train_transformer_diffusion_step(model, batch, self.tf_encoder, optimizer, device, noise_block_prob=noise_block_prob)

                    total_loss += loss
                    if batch_num % max(1, total_batches // 10) == 0 or batch_num == total_batches:
                        self.after(0, lambda bp=batch_num/total_batches, bn=batch_num, tb=total_batches: (
                            self.tf_progress_bar.set(bp),
                            self.tf_progress_label.configure(text=self._tr("batch_progress", bn=bn, tb=tb)),
                        ))
                avg_loss = total_loss / max(total_batches, 1)
                elapsed = time.time() - total_start
                self.after(0, lambda e=epoch, l=avg_loss, p=epoch/epochs, el=elapsed: (
                    self.tf_progress_bar.set(p),
                    self.tf_progress_label.configure(text=self._tr("tf_epoch_progress", e=e, epochs=epochs, loss=f"{l:.4f}")),
                    self.tf_loss_var.set(self._tr("loss_time", loss=f"{l:.4f}", time=f"{el:.0f}")),
                    self.train_epoch_bar.set(p),
                    self.train_epoch_label.configure(text=self._tr("tf_epoch_progress", e=e, epochs=epochs, loss=f"{l:.4f}")),
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
                            "epoch": epoch, "loss": avg_loss,
                           }, out_dir / "model.pt")
            self.after(0, lambda: self.status_var.set(self._tr("status_tf_diffusion_done")))
            self.after(0, self._discover_models)
            self.after(0, self._refresh_models_tab)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.after(0, lambda msg=str(e)[:100]: self.tf_loss_var.set(self._tr("tf_error", msg=msg)))
        finally:
            self.training_running = False
            self.after(0, lambda: self.train_btn_tf_diffusion.configure(state="normal"))
            self.after(0, lambda: self.train_stop_btn.configure(state="disabled"))

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
            self._tr("about_diffusion"),
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
        elif value == "Diffusion":
            self.model_type = "diffusion"
        else:
            self.model_type = "transformer"
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

    def _load_models(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() and self.config.gpu_enabled else "cpu")
            tf_name = self.current_transformer_name or self.model_registry.default_transformer
            df_name = self.current_diffusion_name or self.model_registry.default_diffusion
            tf_diff_name = self.current_tf_diffusion_name or self.model_registry.default_tf_diffusion
            self.model_registry.discover()
            loaded_any = False
            msg_parts = []

            # ── Load TF Diffusion ──
            if tf_diff_name:
                entry = self.model_registry.get(tf_diff_name)
                if entry and entry.checkpoint_path.exists():
                    ckpt = torch.load(entry.checkpoint_path, map_location="cpu")
                    if self.prompt_tokenizer is None:
                        self.prompt_tokenizer = PromptTokenizer.load(entry.path / "prompt_vocab.json")
                    if self.voxel_tokenizer is None:
                        self.voxel_tokenizer = VoxelTokenizer.load(entry.path / "block_vocab.json")
                    encoder_config = ckpt.get("encoder_config", {})
                    encoder_name = encoder_config.get("display_name", "Phi-3.5-mini")
                    enc_device = torch.device("cuda" if torch.cuda.is_available() and self.config.gpu_enabled else "cpu")
                    enc_dtype = torch.float16 if enc_device.type == "cuda" else torch.float32
                    if self.tf_encoder is None or self.tf_encoder.display_name != encoder_name:
                        self.tf_encoder = FrozenTransformerEncoder(model_name=encoder_name, device=enc_device, dtype=enc_dtype)
                    # Default context_proj_dim to d_model*2 to match training behavior
                    _loaded_d_model = ckpt.get("d_model", 64)
                    model = TransformerDiffusionModel(
                        num_blocks=ckpt.get("num_blocks", ckpt.get("block_vocab_size", 0)),
                        grid_size=tuple(ckpt.get("grid_size", (16, 16, 16))),
                        d_model=_loaded_d_model, channels=ckpt.get("channels", 32),
                        channel_multipliers=tuple(ckpt.get("channel_multipliers", (1, 2, 2))),
                        num_timesteps=ckpt.get("num_timesteps", 50),
                        context_dim=ckpt.get("context_dim", 768),
                        cross_attn_heads=ckpt.get("cross_attn_heads", 4),
                        context_proj_dim=ckpt.get("context_proj_dim", _loaded_d_model * 2),
                    ).to(device)
                    model.load_state_dict(ckpt["model_state"], strict=True)
                    model.eval()
                    self.tf_diffusion_model = model
                    self.current_tf_diffusion_name = tf_diff_name
                    loaded_any = True
                    msg_parts.append(f"✅ TF-Diffusion ({tf_diff_name})")

            # ── Load Transformer ──
            if tf_name:
                entry = self.model_registry.get(tf_name)
                if entry and entry.checkpoint_path.exists():
                    ckpt = torch.load(entry.checkpoint_path, map_location="cpu")
                    pt = PromptTokenizer.load(entry.path / "prompt_vocab.json")
                    vt = VoxelTokenizer.load(entry.path / "block_vocab.json")
                    self.prompt_tokenizer = pt
                    self.voxel_tokenizer = vt
                    model = SharedWeightVoxelTransformer(
                        text_vocab_size=ckpt["text_vocab_size"],
                        block_vocab_size=ckpt["block_vocab_size"],
                        grid_size=tuple(ckpt["grid_size"]), d_model=ckpt["d_model"],
                        nhead=ckpt.get("nhead", 6), num_layers=ckpt["layers"],
                        dim_feedforward=ckpt.get("dim_feedforward", 768), dropout=0.0,
                    ).to(device)
                    model.load_state_dict(ckpt["model_state"])
                    model.eval()
                    self.transformer_model = model
                    self.current_transformer_name = tf_name
                    loaded_any = True
                    msg_parts.append(f"✅ Transformer ({tf_name})")

            # ── Load Diffusion ──
            if df_name:
                entry = self.model_registry.get(df_name)
                if entry and entry.checkpoint_path.exists():
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
                        mult = out_ch // actual_channels
                        actual_ch_mult.append(mult)
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
                    loaded_any = True
                    msg_parts.append(f"✅ Diffusion ({df_name})")

            status = " | ".join(msg_parts) if loaded_any else self._tr("status_no_model")
        except Exception as e:
            status = self._tr("gen_error", error=str(e)[:60])
            self.transformer_model = None
            self.diffusion_model = None
            self.tf_diffusion_model = None
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
        self.generation_running = True
        self.generate_btn.configure(state="disabled", text=self._tr("btn_generating"))
        self.progress_bar.start()
        self.status_var.set(self._tr("status_generating"))
        threading.Thread(target=self._generate_worker, args=(prompt,), daemon=True).start()

    def _generate_worker(self, prompt: str):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() and self.config.gpu_enabled else "cpu")
            temp = self.temp_slider.get()
            topk = int(self.topk_slider.get())
            info = ""
            if self.model_type == "transformer" and self.transformer_model is not None:
                prompt_ids = self.prompt_tokenizer.encode(prompt).unsqueeze(0).to(device)
                grid = self.transformer_model.generate(prompt_ids, temperature=temp, top_k=topk)[0].cpu()
                self.id_to_block = self.voxel_tokenizer.id_to_block
                gs = grid.shape
                info = self._tr("gen_transformer_done", x=gs[0], y=gs[1], z=gs[2], temp=f"{temp:.2f}", topk=topk, prompt=prompt[:60])
            elif self.model_type == "diffusion" and self.diffusion_model is not None:
                num_steps = int(self.diff_steps_slider.get())
                prompt_ids = self.prompt_tokenizer.encode(prompt).unsqueeze(0).to(device)
                grid = self.diffusion_model.sample(prompt_ids, num_steps=num_steps, temperature=temp)[0].cpu()
                self.id_to_block = self.voxel_tokenizer.id_to_block
                gs = grid.shape
                info = self._tr("gen_diffusion_done", x=gs[0], y=gs[1], z=gs[2], temp=f"{temp:.2f}", steps=num_steps, prompt=prompt[:60])
            elif self.model_type == "transformer_diffusion" and self.tf_diffusion_model is not None and self.tf_encoder is not None:
                num_steps = int(self.diff_steps_slider.get())
                with torch.no_grad():
                    encoded = self.tf_encoder([prompt])
                    # Cast to model's dtype to avoid Half/Float mismatch
                    model_dtype = next(self.tf_diffusion_model.parameters()).dtype
                    context = encoded["last_hidden_state"].to(device=device, dtype=model_dtype)
                    context_mask = encoded["attention_mask"].to(device=device)
                grid = self.tf_diffusion_model.sample(context, context_mask, num_steps=num_steps,
                                                      temperature=temp, top_k=topk)[0].cpu()
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

    def _generation_failed(self, error: str):
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.generate_btn.configure(state="normal", text=self._tr("btn_generate"))
        self.info_var.set(self._tr("gen_error", error=error))
        self.status_var.set(self._tr("gen_error", error=error))
        self.generation_running = False

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

    def _update_preview(self):
        if self.generated_np is None or self.id_to_block is None:
            return
        img = render_preview(self.generated_np, self.id_to_block, view=self.current_view,
                             size=(420, 380), azimuth=self.azimuth, elevation=self.elevation, scale=self.zoom_scale)
        img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.current_preview_img = img_tk
        self.preview_label.configure(image=img_tk, text="")

    def _on_mouse_down(self, event):
        self._drag_start_x = event.x
        self._orbit_start_az = self.azimuth
        self._dragging = True

    def _on_mouse_drag(self, event):
        if not self._dragging:
            return
        dx = event.x - self._drag_start_x
        self.azimuth = (self._orbit_start_az - dx * 0.5) % 360
        self.elevation = 30.0
        self.current_view = "free"
        self._update_preview()

    def _on_mouse_up(self, event):
        self._dragging = False

    def _on_mouse_wheel(self, event):
        delta = event.delta / 120
        self.zoom_scale = max(1.5, min(6.0, self.zoom_scale + delta * 0.3))
        self.zoom_slider.set(self.zoom_scale)
        self._update_preview()

    def _update_zoom(self, value: float):
        self.zoom_scale = value
        if self.generated_np is not None and self.id_to_block is not None:
            self._update_preview()

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
        save_schem(path, self.generated_grid, self.id_to_block, swap_directions=True)
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
                self.project_listbox.insert("end", f"   📝 {pt}\n\n")
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
        """Get all .schem and .txt files from training data directories."""
        schem_files = []
        txt_files = []
        for data_dir_str in self.data_dirs:
            data_dir = Path(data_dir_str)
            if not data_dir.exists():
                continue
            for f in sorted(data_dir.iterdir()):
                if f.suffix.lower() == ".schem":
                    txt_path = f.with_suffix(".txt")
                    if txt_path.exists():
                        schem_files.append(f)
                        txt_files.append(txt_path)
        return schem_files, txt_files

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
        gs = self.grid_size_var.get()
        return GRID_SIZE_MAP.get(gs, (16, 16, 16))

    def _get_suggested_arch(self) -> Optional[dict]:
        grid_size = self._get_selected_grid_size()
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
                        "dim_feedforward": dim_ff, "params": params, "params_m": params / 1_000_000}
            except (ValueError, ZeroDivisionError):
                return None
        else:
            target_m = self.size_slider_var.get()
            from model import suggest_architecture
            return suggest_architecture(target_m, text_vocab, block_vocab, grid_size)

    def _on_size_slider(self, value: float):
        if self.show_advanced:
            return
        arch = self._get_suggested_arch()
        if arch is None:
            self.params_info_var.set(self._tr("no_arch_found"))
            return
        self.params_info_var.set(self._tr("target_arch", val=f"{value:.1f}", params=f"{arch['params_m']:.2f}", d_model=arch['d_model'], nhead=arch['nhead'], layers=arch['num_layers'], ffn=arch['dim_feedforward']))
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

    def _start_training(self, model_type: str):
        if self.training_running:
            return
        arch = getattr(self, '_selected_arch', None)
        if arch is None:
            self._on_size_slider(self.size_slider_var.get())
            arch = self._selected_arch
        grid_size = self._get_selected_grid_size()
        augmentation_diversity = int(round(float(self.augmentation_diversity_var.get())))
        allow_vertical_movement = bool(self.setting_aug_vertical.get())
        gs_label = f"{grid_size[0]}x{grid_size[1]}x{grid_size[2]}"
        self.training_running = True
        self.train_epoch_bar.set(0)
        self.train_batch_bar.set(0)
        self.status_var.set(self._tr("status_training_started", type=model_type, grid=gs_label))
        self.train_btn_transformer.configure(state="disabled")
        self.train_btn_diffusion.configure(state="disabled")
        self.train_btn_tf_diffusion.configure(state="disabled")
        self.train_stop_btn.configure(state="normal")
        self._current_training_type = model_type
        threading.Thread(target=self._training_worker,
                         args=(model_type, grid_size, augmentation_diversity, allow_vertical_movement),
                         daemon=True).start()

    def _training_worker(self, model_type: str, grid_size: tuple[int, int, int],
                         augmentation_diversity: int, allow_vertical_movement: bool):
        try:
            epochs = int(self.train_epochs_entry.get())
            batch_size = int(self.train_batch_entry.get())
            lr = float(self.train_lr_entry.get())
            device = torch.device("cuda" if torch.cuda.is_available() and self.config.gpu_enabled else "cpu")
            arch = getattr(self, '_selected_arch', None)
            continuing_model = False
            continuing_name = None
            if model_type == "transformer" and self.transformer_model is not None and self.current_transformer_name:
                continuing_model = tuple(self.transformer_model.grid_size) == grid_size
                continuing_name = self.current_transformer_name
            elif model_type == "diffusion" and self.diffusion_model is not None and self.current_diffusion_name:
                continuing_model = tuple(self.diffusion_model.grid_size) == grid_size
                continuing_name = self.current_diffusion_name

            orig_pt = None
            orig_vt = None
            if continuing_model and continuing_name:
                from dataset import PromptTokenizer, VoxelTokenizer
                pt_path = Path("runs") / continuing_name / "prompt_vocab.json"
                vt_path = Path("runs") / continuing_name / "block_vocab.json"
                if vt_path.exists():
                    orig_vt = VoxelTokenizer.load(vt_path)
                if pt_path.exists():
                    orig_pt = PromptTokenizer.load(pt_path)

            air_weight = float(self.air_weight_var.get()) if hasattr(self, 'air_weight_var') else 75.0

            from dataset import MultiSourceSchematicDataset
            dataset = MultiSourceSchematicDataset(
                self.data_dirs, target_size=grid_size, max_voxels=400_000,
                augmentation_diversity=augmentation_diversity,
                allow_vertical_movement=allow_vertical_movement,
                prompt_tokenizer=orig_pt, voxel_tokenizer=orig_vt,
                air_weight_factor=air_weight,
            )
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            total_batches = len(loader)

            if model_type == "transformer":
                from model import SharedWeightVoxelTransformer
                continuing = (self.transformer_model is not None and self.current_transformer_name
                              and tuple(self.transformer_model.grid_size) == grid_size)
                if continuing:
                    out_dir = Path("runs") / self.current_transformer_name
                else:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    out_dir = Path("runs") / f"transformer_{timestamp}"
                out_dir.mkdir(parents=True, exist_ok=True)
                if continuing:
                    model = self.transformer_model
                else:
                    if arch:
                        d_model, nhead, num_layers, dim_ff = arch["d_model"], arch["nhead"], arch["num_layers"], arch["dim_feedforward"]
                    else:
                        d_model, nhead, num_layers, dim_ff = 192, 6, 5, 768
                    model = SharedWeightVoxelTransformer(
                        text_vocab_size=len(dataset.prompt_tokenizer.token_to_id),
                        block_vocab_size=len(dataset.voxel_tokenizer.id_to_block),
                        grid_size=grid_size, d_model=d_model, nhead=nhead,
                        num_layers=num_layers, dim_feedforward=dim_ff, dropout=0.1,
                    ).to(device)
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
                        prompt_ids = batch["prompt_ids"].to(device)
                        target = batch["voxel_ids"].to(device).reshape(prompt_ids.shape[0], -1)
                        target = model.safe_clamp_target(target)
                        logits = model(prompt_ids)
                        target_flat = target.reshape(-1)
                        sample_weight = batch["sample_weight"].to(device).view(-1, 1).expand_as(target).reshape(-1)
                        per_block_w = batch["per_block_weight"].to(device).view(-1, 1).expand_as(target).reshape(-1)
                        per_air_w = batch["per_air_weight"].to(device).view(-1, 1).expand_as(target).reshape(-1)
                        weight_per_token = torch.where(target_flat == 0, per_air_w, per_block_w) * sample_weight
                        logp = torch.log_softmax(logits.reshape(-1, logits.shape[-1]), dim=-1)
                        nll = torch.nn.functional.nll_loss(logp, target_flat, reduction='none')
                        loss = (nll * weight_per_token).sum() / weight_per_token.sum().clamp_min(1.0)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        total_loss += float(loss.detach())
                        if batch_num % max(1, total_batches // 10) == 0 or batch_num == total_batches:
                            self.after(0, lambda bp=batch_num/total_batches, bn=batch_num, tb=total_batches: (
                                self.train_batch_bar.set(bp), self.train_batch_label.configure(text=self._tr("batch_progress", bn=bn, tb=tb))))
                    avg_loss = total_loss / max(total_batches, 1)
                    self.after(0, lambda e=epoch, l=avg_loss, p=epoch/epochs: (
                        self.train_epoch_bar.set(p), self.train_epoch_label.configure(text=self._tr("transformer_epoch_progress", e=e, epochs=epochs, loss=f"{l:.4f}"))))
                    torch.save({"model_state": model.state_dict(), "grid_size": grid_size,
                                "text_vocab_size": len(dataset.prompt_tokenizer.token_to_id),
                                "block_vocab_size": len(dataset.voxel_tokenizer.id_to_block),
                                "d_model": model.d_model, "nhead": model.decoder.layers[0].self_attn.num_heads,
                                "layers": len(model.decoder.layers),
                                "dim_feedforward": model.decoder.layers[0].linear1.out_features,
                                "augmentation_diversity": augmentation_diversity,
                                "allow_vertical_movement": allow_vertical_movement,
                                "epoch": epoch, "loss": avg_loss}, out_dir / "model.pt")
                self.after(0, lambda: self.status_var.set(self._tr("status_transformer_done")))

            elif model_type == "diffusion":
                continuing_diff = (self.diffusion_model is not None and self.current_diffusion_name
                                   and tuple(self.diffusion_model.grid_size) == grid_size)
                if continuing_diff:
                    out_dir = Path("runs") / self.current_diffusion_name
                else:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    out_dir = Path("runs") / f"diffusion_{timestamp}"
                out_dir.mkdir(parents=True, exist_ok=True)
                if continuing_diff:
                    model = self.diffusion_model
                else:
                    # Read diffusion steps from the GUI entry (default 50)
                    diff_num_timesteps = 50
                    if hasattr(self, 'diff_train_steps_entry'):
                        try:
                            diff_num_timesteps = int(self.diff_train_steps_entry.get())
                        except (ValueError, TypeError):
                            diff_num_timesteps = 50
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
                        loss = train_diffusion_step(model, batch, optimizer, device, noise_block_prob=noise_block_prob)
                        total_loss += loss
                        if batch_num % max(1, total_batches // 10) == 0 or batch_num == total_batches:
                            self.after(0, lambda bp=batch_num/total_batches, bn=batch_num, tb=total_batches: (
                                self.train_batch_bar.set(bp), self.train_batch_label.configure(text=self._tr("batch_progress", bn=bn, tb=tb))))
                    avg_loss = total_loss / max(total_batches, 1)
                    self.after(0, lambda e=epoch, l=avg_loss, p=epoch/epochs: (
                        self.train_epoch_bar.set(p), self.train_epoch_label.configure(text=self._tr("diffusion_epoch_progress", e=e, epochs=epochs, loss=f"{l:.4f}"))))
                    torch.save({"model_state": model.state_dict(), "grid_size": model.grid_size,
                                "text_vocab_size": len(dataset.prompt_tokenizer.token_to_id),
                                "block_vocab_size": model.num_blocks, "num_blocks": model.num_blocks,
                                "d_model": model.d_model, "d_text": model.d_text, "channels": model.channels,
                                "channel_multipliers": [int(m) for m in model.channel_multipliers],
                                "num_timesteps": model.num_timesteps,
                                "augmentation_diversity": augmentation_diversity,
                                "allow_vertical_movement": allow_vertical_movement,
                                "epoch": epoch, "loss": avg_loss}, out_dir / "model.pt")
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
            self.after(0, lambda: self.train_btn_transformer.configure(state="normal"))
            self.after(0, lambda: self.train_btn_diffusion.configure(state="normal"))
            self.after(0, lambda: self.train_btn_tf_diffusion.configure(state="normal"))
            self.after(0, lambda: self.train_stop_btn.configure(state="disabled"))

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
        threading.Thread(target=self._kaggle_export_worker, daemon=True).start()

    def _kaggle_export_worker(self):
        from tkinter.messagebox import showinfo, showerror
        try:
            epochs = int(self.train_epochs_entry.get().strip())
            batch_size = int(self.train_batch_entry.get().strip())
            lr = float(self.train_lr_entry.get().strip())
            aug_diversity = int(round(float(self.augmentation_diversity_var.get())))
            allow_vertical = bool(self.setting_aug_vertical.get())
            grid_size = self._get_selected_grid_size()
            air_weight = float(self.air_weight_var.get()) if hasattr(self, 'air_weight_var') else 75.0
            # Use the selected model type from the training tab
            train_type = self.train_model_type_selector.get()
            model_type = {"Transformer": "transformer", "Diffusion": "diffusion",
                          "TF-Diffusion": "transformer_diffusion"}.get(train_type, "transformer")
            # Get the selected architecture (for transformer) or UNet config (for TF-Diffusion)
            arch = getattr(self, '_selected_arch', None)
            if arch is None and model_type == "transformer":
                self._on_size_slider(self.size_slider_var.get())
                arch = getattr(self, '_selected_arch', None)
            tf_unet_config = getattr(self, '_tf_unet_config', None)
            # Get encoder info for TF-Diffusion Kaggle export
            enc_name = None
            ctx_dim = None
            if model_type == "transformer_diffusion":
                if self.tf_encoder is not None:
                    enc_name = self.tf_encoder.display_name
                    ctx_dim = self.tf_encoder.hidden_dim
                else:
                    enc_name = self.tf_encoder_combo.get()
                    # Look up hidden_dim from the encoder registry
                    from app.transformer_encoder import MODEL_TO_ID
                    # Known hidden dims for supported encoders
                    _ENCODER_HIDDEN_DIMS = {
                        "Phi-3.5-mini": 3072,
                        "Gemma-2-2B": 2304,
                        "Gemma-2-9B": 3584,
                        "Gemma-2-27B": 4608,
                        "Gemma-3-1B": 768,
                        "Gemma-3-4B": 2560,
                        "Gemma-3-12B": 3840,
                        "Gemma-3-27B": 5120,
                        "Flan-T5-small": 512,
                        "Flan-T5-base": 768,
                        "Flan-T5-large": 1024,
                        "Flan-T5-XL": 2048,
                        "Flan-T5-XXL": 4096,
                    }
                    ctx_dim = _ENCODER_HIDDEN_DIMS.get(enc_name, 3072)  # Default to Phi-3.5-mini dim
            from kaggle_export import create_kaggle_export
            export_path = create_kaggle_export(output_dir="exports", epochs=epochs, batch_size=batch_size,
                                               learning_rate=lr, aug_diversity=aug_diversity,
                                               allow_vertical=allow_vertical, grid_size=grid_size,
                                               model_type=model_type, data_dirs=self.data_dirs,
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
    app = MinecraftStructureApp()
    app.mainloop()


if __name__ == "__main__":
    main()