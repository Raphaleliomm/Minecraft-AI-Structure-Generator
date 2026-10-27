"""Kaggle training: one intermediate-capacity 48³ voxel-transformer."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


SIZES = {
    # Midpoint between the former small and medium presets: one model, not two.
    "intermediate": dict(d_model=112, nhead=4, num_layers=4, dim_feedforward=448),
}
DESIGNS = [("48_window8", (48, 48, 48), 8, 1, 0, 1)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_holdout(raw_examples, seed: int, count: int = 20, max_grid: int = 48):
    import dataset as data_module

    rng = random.Random(seed)
    by_path = {}
    for path, prompt, source in raw_examples:
        by_path.setdefault(Path(path).resolve(), []).append((Path(path), prompt, Path(source)))
    candidates = list(by_path)
    rng.shuffle(candidates)
    selected, selected_hashes = [], set()
    from dataset import load_schematic, trimmed_block_size
    for path in candidates:
        try:
            digest = _sha256(path)
            if digest in selected_hashes:
                continue
            schematic = load_schematic(path, max_voxels=400_000)
            if any(dimension > max_grid for dimension in trimmed_block_size(schematic)):
                continue
            options = by_path[path]
            selected.append(rng.choice(options))
            selected_hashes.add(digest)
            if len(selected) == count:
                break
        except Exception as exc:
            print(f"Holdout-Kandidat übersprungen: {path.name}: {exc}", flush=True)
    if len(selected) < count:
        raise RuntimeError(f"Nur {len(selected)} passende 48³-Val-Strukturen gefunden; benötigt: {count}")
    # Return hashes as well as paths to keep byte-identical copies out of train.
    path_hashes = {path: _sha256(path) for path in candidates}
    return selected, selected_hashes, path_hashes


def _make_dataset(root: Path, raw_examples, holdout_hashes, path_hashes, grid, tokenizer=None,
                  validation=False, progress_label=""):
    import dataset as data_module
    from dataset import MultiSourceSchematicDataset

    selected = []
    for path, prompt, source in raw_examples:
        digest = path_hashes[Path(path).resolve()]
        is_val = digest in holdout_hashes
        if is_val == validation:
            selected.append((path, prompt, source))
    original = data_module.discover_examples_many
    data_module.discover_examples_many = lambda _dirs: selected
    try:
        def progress(done, total):
            if done % 500 == 0 or done == total:
                print(f"{progress_label}{done}/{total} Struktureinträge geprüft", flush=True)
        ds = MultiSourceSchematicDataset(
            [str(root)], target_size=grid, voxel_tokenizer=(tokenizer.voxel_tokenizer if tokenizer else None),
            prompt_tokenizer=(tokenizer.prompt_tokenizer if tokenizer else None),
            max_voxels=400_000, augmentation_diversity=0,
            allow_vertical_movement=False, progress_callback=progress,
        )
    finally:
        data_module.discover_examples_many = original
    return ds


def _validation_metrics(model, dataset, selected, device, chunk=256):
    model.eval()
    scores = []
    for path, prompt, _source in selected:
        matches = [i for i, item in enumerate(dataset.examples)
                   if item.path.resolve() == Path(path).resolve() and item.prompt == prompt]
        if not matches:
            continue
        sample = dataset[matches[0]]
        prompt_ids = sample["prompt_ids"].unsqueeze(0).to(device)
        target = sample["voxel_ids"].reshape(-1).to(device)
        per_block = sample["per_block_weight"].to(device)
        per_air = sample["per_air_weight"].to(device)
        sample_weight = sample["sample_weight"].to(device)
        weight = torch.where(target == 0, per_air, per_block) * sample_weight
        with torch.no_grad():
            decoded = model.decoded(prompt_ids=prompt_ids)
            prediction = torch.empty_like(target)
            loss_sum = 0.0
            for start in range(0, target.numel(), chunk):
                end = min(start + chunk, target.numel())
                logits = model.output(decoded[:, start:end])
                prediction[start:end] = logits.argmax(-1).reshape(-1)
                nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                      target[start:end], reduction="none")
                loss_sum += float((nll * weight[start:end]).sum())
                del logits
            loss_sum /= max(1e-8, float(weight.sum()))
        target_nonair = target != 0
        pred_nonair = prediction != 0
        intersection = int((target_nonair & pred_nonair).sum())
        union = int((target_nonair | pred_nonair).sum())
        scores.append({
            "path": Path(path).name,
            "prompt": prompt,
            "token_accuracy": float((target == prediction).float().mean()),
            "nonair_iou": intersection / max(1, union),
            "nonair_precision": intersection / max(1, int(pred_nonair.sum())),
            "nonair_recall": intersection / max(1, int(target_nonair.sum())),
            "voxel_ce": loss_sum,
            "target_nonair_fraction": float(target_nonair.float().mean()),
            "pred_nonair_fraction": float(pred_nonair.float().mean()),
        })
    if not scores:
        return {"val_count": 0}
    return {"val_count": len(scores), **{
        key: sum(row[key] for row in scores) / len(scores)
        for key in ("token_accuracy", "nonair_iou", "nonair_precision", "nonair_recall",
                    "voxel_ce", "target_nonair_fraction", "pred_nonair_fraction")
    }, "samples": scores}


def _train_one(name, size_name, design, size_config, train_set, val_set, selected,
               output_dir, device, epochs=100, resume_root: Path | None = None):
    from model import SharedWeightVoxelTransformer
    from app.grad_accum import iter_micro_batches, per_sample_weight_mass, apply_optimizer_step
    from app.training_memory import plan_transformer_training

    design_name, grid, window, recurrence, experts, active = design
    args = dict(size_config)
    args.update(recurrent_depth=recurrence, window_size=window,
                moe_experts=experts, moe_active=active,
                tread_ratio=0.25, ssa_topk=16, voxel_mask_ratio=0.65)
    name = f"{design_name}_{size_name}"
    out = output_dir / name
    out.mkdir(parents=True, exist_ok=True)
    plan = plan_transformer_training(
        grid, 16, len(train_set.voxel_tokenizer.id_to_block), device=device,
        d_model=args["d_model"], nhead=args["nhead"], num_layers=args["num_layers"],
        dim_feedforward=args["dim_feedforward"], recurrent_depth=recurrence,
        window_size=window, allow_accumulation=True,
    )
    print(f"\n=== {name} ===", flush=True)
    print(plan, flush=True)
    if not plan.fits:
        result = {"model": name, "status": "does_not_fit",
                  "grid": grid, "window": window, "size": size_config}
        (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        torch.cuda.empty_cache()
        return result
    model = SharedWeightVoxelTransformer(
        text_vocab_size=len(train_set.prompt_tokenizer.token_to_id),
        block_vocab_size=len(train_set.voxel_tokenizer.id_to_block),
        grid_size=grid, dropout=0.1, **args,
    ).to(device)
    print(f"Parameters total={model.num_parameters:,} active={model.active_parameters:,}", flush=True)
    loader = DataLoader(train_set, batch_size=plan.effective_batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, persistent_workers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    total_steps = epochs * len(loader)
    warmup = min(100, max(1, total_steps // 20))
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    history = []
    best_val = float("inf")
    start_epoch = 1
    resume_candidates = [out / "latest_checkpoint.pt"]
    if resume_root is not None and resume_root.exists():
        resume_candidates.extend(p for p in resume_root.rglob("latest_checkpoint.pt")
                                 if p.parent.name == name)
    if Path("/kaggle/input").exists():
        resume_candidates.extend(p for p in Path("/kaggle/input").rglob("latest_checkpoint.pt")
                                 if p.parent.name == name)
    resume_path = next((p for p in resume_candidates if p.is_file()), None)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        history = checkpoint.get("history", [])
        best_val = checkpoint.get("best_val", float("inf"))
        start_epoch = int(checkpoint["epoch"]) + 1
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
        if checkpoint.get("python_rng_state") is not None:
            random.setstate(checkpoint["python_rng_state"])
        print(f"Resume {name}: Epoche {start_epoch}/{epochs} aus {resume_path}", flush=True)
    start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss, batches = 0.0, 0
        for batch in loader:
            masses = per_sample_weight_mass(batch["voxel_ids"], batch) if plan.accumulated else None
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for micro, share in iter_micro_batches(batch, plan.batch_size, sample_mass=masses):
                mb_target = micro["voxel_ids"].to(device, non_blocking=True).reshape(len(micro["prompt_ids"]), -1)
                mb_target = model.safe_clamp_target(mb_target)
                mb_prompt = micro["prompt_ids"].to(device, non_blocking=True)
                sw = micro["sample_weight"].to(device, non_blocking=True).view(-1, 1).expand_as(mb_target)
                bw = micro["per_block_weight"].to(device, non_blocking=True).view(-1, 1).expand_as(mb_target)
                aw = micro["per_air_weight"].to(device, non_blocking=True).view(-1, 1).expand_as(mb_target)
                mb_weights = torch.where(mb_target == 0, aw, bw) * sw
                loss = model.backward_weighted_loss(mb_target, mb_weights,
                    chunk_voxels=256, grad_scale=share, prompt_ids=mb_prompt)
                batch_loss += loss * share
            apply_optimizer_step(model, optimizer, max_norm=1.0)
            scheduler.step()
            train_loss += batch_loss
            batches += 1
        train_loss /= max(1, batches)
        val = _validation_metrics(model, val_set, selected, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val.get("voxel_ce"),
               "val_token_accuracy": val.get("token_accuracy"), "val_nonair_iou": val.get("nonair_iou"),
               "lr": scheduler.get_last_lr()[0], "seconds": time.time() - start}
        history.append(row)
        print(f"{name} epoch {epoch}/{epochs}: train={train_loss:.5f} "
              f"val={row['val_loss']} IoU={row['val_nonair_iou']}", flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if val.get("voxel_ce", float("inf")) < best_val:
            best_val = val["voxel_ce"]
            torch.save({"model_state": model.state_dict(), "grid_size": grid,
                        "text_vocab_size": len(train_set.prompt_tokenizer.token_to_id),
                        "block_vocab_size": len(train_set.voxel_tokenizer.id_to_block),
                        **args, "epoch": epoch, "val": val}, out / "best_model.pt")
            (out / "val_samples.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
        checkpoint = {
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "epoch": epoch,
            "best_val": best_val, "history": history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng_state": random.getstate(), "grid_size": grid,
            "text_vocab_size": len(train_set.prompt_tokenizer.token_to_id),
            "block_vocab_size": len(train_set.voxel_tokenizer.id_to_block), **args,
        }
        checkpoint_tmp = out / "latest_checkpoint.tmp"
        torch.save(checkpoint, checkpoint_tmp)
        checkpoint_tmp.replace(out / "latest_checkpoint.pt")
    result = {"model": name, "status": "complete", "parameters": model.num_parameters,
              "active_parameters": model.active_parameters, "grid": grid, "window": window,
              "recurrent_depth": recurrence, "moe_experts": experts, "moe_active": active,
              "size": size_config, "history": history, "best_val": best_val,
              "validation": _validation_metrics(model, val_set, selected, device)}
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    del model, loader, optimizer
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--resume-root", type=Path, default=None,
                        help="Optional directory containing per-model latest_checkpoint.pt files")
    args = parser.parse_args()
    import dataset as data_module
    from dataset import discover_examples_many
    root = args.data_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU ist nicht aktiviert; Benchmark wird nicht auf CPU gestartet.")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch={torch.__version__}", flush=True)
    raw = discover_examples_many([root])
    if not raw:
        raise RuntimeError(f"Keine prompt-annotierten Schematics unter {root}")
    selected, holdout_hashes, all_hashes = _validation_holdout(raw, args.seed, max_grid=48)
    split = {"seed": args.seed, "validation_count": len(selected),
             "validation": [{"path": str(p), "prompt": q} for p, q, _ in selected],
             "deduplicated_holdout_hashes": sorted(holdout_hashes),
             "raw_prompt_examples": len(raw), "unique_structure_hashes": len(set(all_hashes.values()))}
    (out / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Prompt-Beispiele: {len(raw)} | Val-Strukturen: {len(selected)} | "
          f"einzigartige Build-Hashes: {split['unique_structure_hashes']}", flush=True)

    results = []
    datasets = {}
    for grid_name, grid in (("48", (48, 48, 48)),):
        train_set = _make_dataset(root, raw, holdout_hashes, all_hashes, grid,
                                  progress_label=f"{grid_name}³ Train: ")
        val_set = _make_dataset(root, raw, holdout_hashes, all_hashes, grid, tokenizer=train_set,
                                validation=True, progress_label=f"{grid_name}³ Val: ")
        print(f"{grid_name}³: train={len(train_set)} val={len(val_set)} "
              f"blocks={len(train_set.voxel_tokenizer.id_to_block)}", flush=True)
        datasets[grid_name] = (train_set, val_set)

    for design in DESIGNS:
        grid_name = str(design[1][0])
        train_set, val_set = datasets[grid_name]
        for size_name, size_config in SIZES.items():
            name = f"{design[0]}_{size_name}"
            try:
                result = _train_one(name, size_name, design, size_config,
                                    train_set, val_set, selected, out, device, args.epochs,
                                    resume_root=args.resume_root)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                result = {"model": name, "status": "out_of_memory", "error": str(exc),
                          "grid": design[1], "size": size_config}
                folder = out / name
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(f"OOM bei {name}; nächster Lauf folgt.", flush=True)
            results.append(result)
            (out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
                fields = ["model", "status", "parameters", "active_parameters", "grid", "window",
                          "recurrent_depth", "moe_experts", "moe_active", "best_val"]
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results)
    print(f"Benchmark fertig: {len(results)} Modellläufe.", flush=True)


if __name__ == "__main__":
    main()
