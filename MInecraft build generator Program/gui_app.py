from __future__ import annotations

import threading
from pathlib import Path
from tkinter import END, DoubleVar, IntVar, StringVar, Text, Tk, filedialog, messagebox
from tkinter import ttk

import torch

from dataset import PromptTokenizer, VoxelTokenizer, save_schem, trim_token_grid
from model import SharedWeightVoxelTransformer


# Erlaubte Grid-Größen
ALLOWED_GRID_SIZES = {
    "16×16×16": (16, 16, 16),
    "32×32×32": (32, 32, 32),
    "48×48×48": (48, 48, 48),
}

GRID_SIZE_RUN_DIRS = {
    (16, 16, 16): "runs/voxel_transformer_16",
    (32, 32, 32): "runs/voxel_transformer_32",
    (48, 48, 48): "runs/voxel_transformer_48",
}

DEFAULT_RUN_DIR = Path("runs/voxel_transformer_scaled")


class GeneratorApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Minecraft Structure Generator")
        self.root.geometry("1020x660")
        self.root.minsize(860, 560)

        self.run_dir = StringVar(value=str(DEFAULT_RUN_DIR))
        self.status = StringVar(value="Bereit")
        self.model_info = StringVar(value="Kein Modell geladen")
        self.generated_info = StringVar(value="Noch keine Struktur generiert")
        self.temperature = DoubleVar(value=0.85)
        self.top_k = IntVar(value=40)
        self.generated_grid: torch.Tensor | None = None
        self.id_to_block: list[str] | None = None
        self.model: SharedWeightVoxelTransformer | None = None
        self.prompt_tokenizer: PromptTokenizer | None = None
        self.grid_size: tuple[int, int, int] = (16, 16, 16)
        self.selected_grid_size = StringVar(value="16×16×16")

        self._configure_style()
        self._build()
        self.refresh_model_info()

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        # Deep dark theme colors
        bg_dark = "#0d0f14"
        bg_panel = "#141820"
        bg_input = "#1a1f2b"
        fg_primary = "#e8edf5"
        fg_muted = "#8892a8"
        fg_accent = "#60a5fa"
        fg_green = "#34d399"
        border_color = "#252b3a"

        self.root.configure(bg=bg_dark)

        style.configure(".", background=bg_dark, foreground=fg_primary, fieldbackground=bg_input)

        style.configure("Panel.TFrame", background=bg_panel, relief="flat", borderwidth=0)
        style.configure("Card.TFrame", background=bg_panel, relief="solid", borderwidth=1, bordercolor=border_color)

        style.configure("TLabel", background=bg_dark, foreground=fg_primary, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=bg_dark, foreground=fg_muted, font=("Segoe UI", 9))
        style.configure("Panel.TLabel", background=bg_panel, foreground=fg_primary, font=("Segoe UI", 10))
        style.configure("PanelMuted.TLabel", background=bg_panel, foreground=fg_muted, font=("Segoe UI", 9))
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), background=bg_dark, foreground=fg_primary)
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), background=bg_dark, foreground=fg_muted)
        style.configure("Accent.TLabel", background=bg_panel, foreground=fg_accent, font=("Segoe UI", 11, "bold"))
        style.configure("Green.TLabel", background=bg_panel, foreground=fg_green, font=("Segoe UI", 10))

        style.configure("Accent.TButton",
            background="#2563eb",
            foreground="#ffffff",
            padding=(18, 10),
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
            focusthickness=0,
        )
        style.map("Accent.TButton",
            background=[("active", "#1d4ed8"), ("disabled", "#1e293b")],
            foreground=[("disabled", "#475569")],
        )

        style.configure("TButton",
            padding=(14, 8),
            font=("Segoe UI", 9),
            background="#1e293b",
            foreground=fg_primary,
            borderwidth=0,
        )
        style.map("TButton",
            background=[("active", "#334155")],
        )

        style.configure("TEntry",
            padding=8,
            fieldbackground=bg_input,
            foreground=fg_primary,
            bordercolor=border_color,
        )

        style.configure("TSeparator", background=border_color)
        style.configure("TProgressbar",
            background=fg_accent,
            troughcolor=bg_input,
            bordercolor=bg_input,
            lightcolor=fg_accent,
            darkcolor=fg_accent,
        )
        style.configure("TScale",
            background=bg_panel,
            foreground=fg_primary,
            troughcolor=bg_input,
        )

        style.configure("TFrame", background=bg_dark)
        style.configure("TLabelframe", background=bg_panel, foreground=fg_primary)
        style.configure("TLabelframe.Label", background=bg_panel, foreground=fg_muted, font=("Segoe UI", 9))

    def _build(self) -> None:
        # Main container
        outer = ttk.Frame(self.root, padding=20)
        outer.pack(fill="both", expand=True)

        # ── Header ──
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 16))
        ttk.Label(header, text="Minecraft Structure Generator", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Shared-Weight Two-Pass Voxel Transformer  ·  GPU-beschleunigt",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        ttk.Separator(outer).pack(fill="x", pady=(0, 16))

        # ── Body: two columns ──
        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=4)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        # ── LEFT PANEL: Prompt + Controls ──
        left = ttk.Frame(body, style="Panel.TFrame", padding=18)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        left.rowconfigure(3, weight=1)

        # Prompteingabe
        ttk.Label(left, text="Baubeschreibung (Prompt)", style="Panel.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        prompt_frame = ttk.Frame(left, style="Panel.TFrame")
        prompt_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 14))
        prompt_frame.columnconfigure(0, weight=1)
        prompt_frame.rowconfigure(0, weight=1)

        self.prompt = Text(
            prompt_frame,
            height=10,
            wrap="word",
            bg="#0f131c",
            fg="#e8edf5",
            insertbackground="#60a5fa",
            relief="flat",
            padx=14,
            pady=14,
            font=("Segoe UI", 11),
            bd=1,
            highlightthickness=1,
            highlightbackground="#1e2a3a",
            highlightcolor="#2563eb",
        )
        self.prompt.grid(row=0, column=0, sticky="nsew")
        self.prompt.insert(
            END,
            "small medieval wooden cottage with stone foundation and steep oak roof",
        )

        # Controls row
        controls = ttk.Frame(left, style="Panel.TFrame")
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        self.generate_button = ttk.Button(
            controls, text="✦ Generieren", style="Accent.TButton", command=self.generate
        )
        self.generate_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.download_button = ttk.Button(
            controls, text="⬇ Schematic speichern", command=self.save_generated, state="disabled"
        )
        self.download_button.grid(row=0, column=1, sticky="ew")

        # Generation parameters
        params_frame = ttk.Frame(left, style="Panel.TFrame")
        params_frame.grid(row=3, column=0, sticky="nsew")
        params_frame.columnconfigure(1, weight=1)

        ttk.Label(params_frame, text="Parameter", style="Panel.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )

        # Temperature
        ttk.Label(params_frame, text="Temperatur:", style="PanelMuted.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 8)
        )
        temp_slider = ttk.Scale(
            params_frame,
            from_=0.1, to=1.5,
            variable=self.temperature,
            orient="horizontal",
            length=180,
        )
        temp_slider.grid(row=1, column=1, sticky="ew", padx=(0, 8))
        self.temp_label = ttk.Label(params_frame, text="0.85", style="Panel.TLabel", width=5)
        self.temp_label.grid(row=1, column=2, sticky="w")
        self.temperature.trace_add("write", lambda *_: self.temp_label.config(text=f"{self.temperature.get():.2f}"))

        # Top-K
        ttk.Label(params_frame, text="Top-K:", style="PanelMuted.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=(6, 0)
        )
        topk_slider = ttk.Scale(
            params_frame,
            from_=5, to=100,
            variable=self.top_k,
            orient="horizontal",
            length=180,
        )
        topk_slider.grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=(6, 0))
        self.topk_label = ttk.Label(params_frame, text="40", style="Panel.TLabel", width=5)
        self.topk_label.grid(row=2, column=2, sticky="w", pady=(6, 0))
        self.top_k.trace_add("write", lambda *_: self.topk_label.config(text=f"{self.top_k.get():d}"))

        # ── RIGHT PANEL: Model Info + Output ──
        right = ttk.Frame(body, style="Panel.TFrame", padding=18)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)

        # Model section
        ttk.Label(right, text="Modell", style="Panel.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )

        # Grid size selector
        size_row = ttk.Frame(right, style="Panel.TFrame")
        size_row.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        size_row.columnconfigure(1, weight=1)

        ttk.Label(size_row, text="Modell-Größe:", style="PanelMuted.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        size_combo = ttk.Combobox(
            size_row,
            textvariable=self.selected_grid_size,
            values=list(ALLOWED_GRID_SIZES.keys()),
            state="readonly",
            width=14,
        )
        size_combo.grid(row=0, column=1, sticky="w")
        size_combo.bind("<<ComboboxSelected>>", self._on_grid_size_changed)

        model_row = ttk.Frame(right, style="Panel.TFrame")
        model_row.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        model_row.columnconfigure(0, weight=1)

        run_dir_entry = ttk.Entry(model_row, textvariable=self.run_dir)
        run_dir_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(model_row, text="📁 Ordner", command=self.choose_run_dir).grid(row=0, column=1)

        # Model info card
        model_card = ttk.Frame(right, style="Card.TFrame", padding=14)
        model_card.grid(row=3, column=0, sticky="ew", pady=(0, 18))
        model_card.columnconfigure(0, weight=1)

        ttk.Label(model_card, textvariable=self.model_info, style="Panel.TLabel", wraplength=320).grid(
            row=0, column=0, sticky="w"
        )

        ttk.Separator(right).grid(row=4, column=0, sticky="ew", pady=(0, 18))

        # Output section
        ttk.Label(right, text="Generierte Ausgabe", style="Panel.TLabel").grid(
            row=5, column=0, sticky="w", pady=(0, 6)
        )

        output_card = ttk.Frame(right, style="Card.TFrame", padding=14)
        output_card.grid(row=6, column=0, sticky="ew", pady=(0, 14))
        output_card.columnconfigure(0, weight=1)

        ttk.Label(output_card, textvariable=self.generated_info, style="Green.TLabel", wraplength=320).grid(
            row=0, column=0, sticky="w"
        )

        # Progress bar
        self.progress = ttk.Progressbar(right, mode="indeterminate", length=320)
        self.progress.grid(row=7, column=0, sticky="ew")

        # ── Footer ──
        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(16, 0))
        footer.columnconfigure(0, weight=1)

        ttk.Label(footer, textvariable=self.status, style="Muted.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(footer, text="↻ Modellinfo aktualisieren", command=self.refresh_model_info).grid(
            row=0, column=1, sticky="e"
        )

    def _on_grid_size_changed(self, event=None) -> None:
        """Wird aufgerufen, wenn der Benutzer eine andere Grid-Größe im Dropdown auswählt.
        Aktualisiert das run_dir auf das passende Standardverzeichnis für diese Größe
        und lädt die Modellinfo neu."""
        size_label = self.selected_grid_size.get()
        grid = ALLOWED_GRID_SIZES.get(size_label)
        if grid is not None and grid in GRID_SIZE_RUN_DIRS:
            self.run_dir.set(str(Path(GRID_SIZE_RUN_DIRS[grid])))
        self.refresh_model_info()

    def choose_run_dir(self) -> None:
        directory = filedialog.askdirectory(initialdir="runs")
        if directory:
            self.run_dir.set(directory)
            self.refresh_model_info()

    def refresh_model_info(self) -> None:
        checkpoint_path = Path(self.run_dir.get()) / "model.pt"
        self.model = None
        self.prompt_tokenizer = None
        if not checkpoint_path.exists():
            self.model_info.set(f"❌ Nicht gefunden: {checkpoint_path}")
            return
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            grid = tuple(checkpoint["grid_size"])
            block_vocab = checkpoint["block_vocab_size"]
            d_model = checkpoint["d_model"]
            layers = checkpoint["layers"]
            nhead = checkpoint.get("nhead", 4)
            dim_ff = checkpoint.get("dim_feedforward", 512)
            epochs_done = checkpoint.get("epoch", "?")
            loss_val = checkpoint.get("loss", "?")

            self.grid_size = grid
            self.model_info.set(
                f"📐 Grid: {grid[0]}×{grid[1]}×{grid[2]}  ·  "
                f"🧱 Blöcke: {block_vocab}\n"
                f"🧠 d_model={d_model}  heads={nhead}  layers={layers}  ff={dim_ff}\n"
                f"📊 Epoche: {epochs_done}  ·  Loss: {loss_val}"
            )
        except Exception as exc:
            self.model_info.set(f"❌ Kann Modellinfo nicht lesen: {exc}")

    def generate(self) -> None:
        prompt = self.prompt.get("1.0", END).strip()
        if not prompt:
            messagebox.showwarning("Prompt fehlt", "Bitte gib einen Prompt ein.")
            return
        self.generate_button.configure(state="disabled")
        self.download_button.configure(state="disabled")
        self.progress.start(12)
        self.status.set("⏳ Generiere Struktur ...")
        threading.Thread(target=self._generate_worker, args=(prompt,), daemon=True).start()

    def _generate_worker(self, prompt: str) -> None:
        try:
            run_dir = Path(self.run_dir.get())
            checkpoint_path = run_dir / "model.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Kein Modell gefunden: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            prompt_tokenizer = PromptTokenizer.load(run_dir / "prompt_vocab.json")
            voxel_tokenizer = VoxelTokenizer.load(run_dir / "block_vocab.json")
            grid_size = tuple(checkpoint["grid_size"])

            # Prüfe Konsistenz zwischen Checkpoint und Tokenizer
            checkpoint_block_vocab = checkpoint["block_vocab_size"]
            tokenizer_vocab = len(voxel_tokenizer.id_to_block)
            if checkpoint_block_vocab < tokenizer_vocab:
                self.root.after(0, lambda: self.status.set(
                    f"⚠️ Achtung: Checkpoint ({checkpoint_block_vocab} Blöcke) < Tokenizer ({tokenizer_vocab} Blöcke). "
                    "Unbekannte Blöcke werden ignoriert."
                ))

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = SharedWeightVoxelTransformer(
                text_vocab_size=checkpoint["text_vocab_size"],
                block_vocab_size=checkpoint["block_vocab_size"],
                grid_size=grid_size,
                d_model=checkpoint["d_model"],
                nhead=checkpoint.get("nhead", 8),
                num_layers=checkpoint["layers"],
                dim_feedforward=checkpoint.get("dim_feedforward", 1024),
                dropout=0.0,
            ).to(device)
            model.load_state_dict(checkpoint["model_state"])

            prompt_ids = prompt_tokenizer.encode(prompt).unsqueeze(0).to(device)
            temp = self.temperature.get()
            topk = self.top_k.get()

            self.generated_grid = trim_token_grid(
                model.generate(prompt_ids, temperature=temp, top_k=topk)[0].cpu()
            )
            self.id_to_block = voxel_tokenizer.id_to_block
            unique_blocks = int(torch.unique(self.generated_grid).numel())

            self.root.after(0, self._generation_done, tuple(self.generated_grid.shape), unique_blocks, temp, topk)
        except Exception as exc:
            self.root.after(0, self._generation_failed, exc)

    def _generation_done(
        self,
        grid_size: tuple[int, int, int],
        unique_blocks: int,
        temp: float,
        topk: int,
    ) -> None:
        self.progress.stop()
        self.status.set("✅ Fertig")
        self.generated_info.set(
            f"✅ {grid_size[0]}×{grid_size[1]}×{grid_size[2]} Voxels\n"
            f"🧱 {unique_blocks} verschiedene Blocktypen\n"
            f"⚙️ Temperatur={temp:.2f}  Top-K={topk}"
        )
        self.generate_button.configure(state="normal")
        self.download_button.configure(state="normal")

    def _generation_failed(self, exc: Exception) -> None:
        self.progress.stop()
        self.status.set("❌ Generierung fehlgeschlagen")
        self.generate_button.configure(state="normal")
        messagebox.showerror("Fehler", str(exc))

    def save_generated(self) -> None:
        if self.generated_grid is None or self.id_to_block is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".schem",
            filetypes=[("Sponge schematic", "*.schem"), ("All files", "*.*")],
            initialfile="generated.schem",
        )
        if not path:
            return
        save_schem(path, self.generated_grid, self.id_to_block)
        self.status.set(f"💾 Gespeichert: {path}")


def main() -> None:
    root = Tk()
    GeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
