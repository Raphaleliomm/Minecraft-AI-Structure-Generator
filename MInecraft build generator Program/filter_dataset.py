from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from app.paths import PROJECT_ROOT
from dataset import read_schematic_size


def move_family(path: Path, source_dir: Path, target_dir: Path, dry_run: bool) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []
    for sibling in source_dir.glob(path.stem + ".*"):
        target = target_dir / sibling.name
        moved.append(target)
        if not dry_run:
            shutil.move(str(sibling), str(target))
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description="Move oversized Minecraft structures out of the active training set.")
    parser.add_argument("--source-dir", default=str(PROJECT_ROOT / "Trainingsdaten good thoroughly analyzed"))
    parser.add_argument("--target-dir", default=str(PROJECT_ROOT / "Trainingsdaten zu gross vorerst ausgelagert"))
    parser.add_argument("--max-voxels", type=int, default=100_000)
    parser.add_argument("--move-litematic", action=argparse.BooleanOptionalAction, default=True,
                        help="Move .litematic files (which the dataset parser cannot read) as well")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)

    for schem_path in sorted(source_dir.glob("*.schem")) + sorted(source_dir.glob("*.schematic")):
        # Nur der NBT-Header: das Dekodieren der Bloecke war fuer einen
        # 1000x1000-Build der Grund fuer Minuten Laufzeit und mehrere GB RAM.
        size = read_schematic_size(schem_path)
        voxels = size[0] * size[1] * size[2]
        if voxels > args.max_voxels:
            moved = move_family(schem_path, source_dir, target_dir, args.dry_run)
            action = "Wuerde verschieben" if args.dry_run else "Verschoben"
            print(f"{action}: {schem_path.stem} ({size}, {voxels} voxels) -> {len(moved)} Dateien")

    if args.move_litematic:
        for litematic_path in sorted(source_dir.glob("*.litematic")):
            moved = move_family(litematic_path, source_dir, target_dir, args.dry_run)
            action = "Wuerde verschieben" if args.dry_run else "Verschoben"
            print(f"{action}: {litematic_path.stem} (litematic noch nicht im Dataset-Parser) -> {len(moved)} Dateien")


if __name__ == "__main__":
    main()
