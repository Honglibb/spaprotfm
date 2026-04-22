"""Cross-dataset SpaProtFM v2 training on a shared canonical IMC panel.

Trains a single MaskedUNetV2 (+Phikon-v2 pseudo-H&E) on pooled images from
multiple IMC datasets projected to a canonical 10-marker shared panel:
    [DNA1, DNA2, H3, SMA, CD20, CD3e, CD45, CD68, Ki67, cPARP]

Always-observed = first 4 (morphology). Bio targets = last 6.

Supports held-out datasets: images from held-out datasets are excluded from
train/val and used in full as an out-of-distribution test set. This lets us
evaluate zero-shot transfer to an unseen cohort.

No stage-2 fine-tune (the recent ablation showed it trades flexibility).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from spaprotfm.condition.phikon import PhikonEncoder
from spaprotfm.condition.pseudo_he import synthesize_pseudo_he_batch
from spaprotfm.data.bodenmiller import (
    DEFAULT_CANONICAL_MARKERS,
    DEFAULT_MARKER_ALIASES,
    load_imc_rds,
    project_to_canonical_panel,
)
from spaprotfm.data.normalization import normalize_image
from spaprotfm.data.tiling import tile_image
from spaprotfm.eval.metrics import mean_squared_error, pearson_correlation
from spaprotfm.models.spaprotfm_v0 import build_masked_input, fixed_mask
from spaprotfm.models.spaprotfm_v1 import random_mask_with_always_observed
from spaprotfm.models.spaprotfm_v2 import MaskedUNetV2

log = logging.getLogger(__name__)

DATASET_RDS = {
    "Damond": "/code/zkgy/hongliyin_computer/data/raw/imc/Damond_2019_Pancreas_images.rds",
    "HochSchulz": "/code/zkgy/hongliyin_computer/data/raw/imc/HochSchulz_2022_Melanoma_images_protein.rds",
    "Jackson": "/code/zkgy/hongliyin_computer/data/raw/imc/JacksonFischer_2020_BreastCancer_images_basel.rds",
}


def parse_names(arg: str) -> list[str]:
    if not arg.strip():
        return []
    return [x.strip() for x in arg.split(",") if x.strip()]


def encode_phikon_batch(full_tiles, dna_idx, bio_idx, phikon, device, phikon_dtype):
    rgb = synthesize_pseudo_he_batch(full_tiles.detach().cpu(), dna_idx, bio_idx)
    rgb = rgb.to(device)
    if phikon_dtype is not None:
        rgb = rgb.to(phikon_dtype)
    return phikon.encode(rgb)


def tile_set(imgs, patch_size, n_channels):
    if not imgs:
        return np.zeros((0, patch_size, patch_size, n_channels), dtype=np.float32)
    out = []
    for im in imgs:
        normed = normalize_image(im.image).astype(np.float32)
        tiles, _ = tile_image(normed, patch_size=patch_size, stride=patch_size)
        out.append(tiles)
    return np.concatenate(out, axis=0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-datasets", required=True,
                   help="Comma-separated dataset names (from Damond,HochSchulz,Jackson) "
                        "whose 80/10/10 splits contribute to train/val/in-dist-test.")
    p.add_argument("--heldout-datasets", default="",
                   help="Comma-separated dataset names whose images are ALL held out "
                        "as an out-of-distribution test set.")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--run-name", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-images-per-dataset", type=int, default=0)
    p.add_argument("--base", type=int, default=48)
    p.add_argument("--cond-dim", type=int, default=192)
    p.add_argument("--phikon-model", default="owkin/phikon-v2")
    p.add_argument("--no-cond", action="store_true")
    p.add_argument("--no-random-mask", action="store_true",
                   help="Baseline mode: train with fixed mask (no random masking). "
                        "Combined with --no-cond this produces a Murphy-equivalent U-Net.")
    p.add_argument("--save-checkpoint", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # ---- Canonical panel ----
    canonical = DEFAULT_CANONICAL_MARKERS
    n_channels = len(canonical)
    dna_channels = [0, 1]                # DNA1, DNA2
    always_observed = [0, 1, 2, 3]       # DNA1, DNA2, H3, SMA
    eval_obs = list(always_observed)     # transfer eval: observe morphology only
    target_idx = [i for i in range(n_channels) if i not in eval_obs]
    bio_idx_for_he = [i for i in range(n_channels) if i not in set(dna_channels)]

    log.info("Canonical panel (%d markers): %s", n_channels, canonical)
    log.info("Always-observed: %s = %s", always_observed, [canonical[i] for i in always_observed])
    log.info("Targets (predicted): %s = %s", target_idx, [canonical[i] for i in target_idx])

    # ---- Load train/val/in-dist-test splits per dataset, and held-out OOD test ----
    train_ds_names = parse_names(args.train_datasets)
    heldout_names = parse_names(args.heldout_datasets)
    if not train_ds_names:
        raise SystemExit("--train-datasets must contain at least one dataset name")
    unknown = set(train_ds_names + heldout_names) - set(DATASET_RDS)
    if unknown:
        raise SystemExit(f"Unknown dataset names: {unknown}")

    max_imgs = args.max_images_per_dataset or None

    # Build train / val / in-dist test by pooling 80/10/10 splits per train dataset
    train_imgs, val_imgs, indist_test = [], [], {}
    for name in train_ds_names:
        raw = load_imc_rds(DATASET_RDS[name], max_images=max_imgs)
        imgs = project_to_canonical_panel(raw, name)
        perm = rng.permutation(len(imgs))
        n_tr = int(0.8 * len(imgs))
        n_va = int(0.1 * len(imgs))
        train_imgs += [imgs[i] for i in perm[:n_tr]]
        val_imgs += [imgs[i] for i in perm[n_tr:n_tr + n_va]]
        indist_test[name] = [imgs[i] for i in perm[n_tr + n_va:]]
        log.info("  %s: train=%d val=%d in-dist-test=%d",
                 name, n_tr, n_va, len(imgs) - n_tr - n_va)

    ood_test: dict[str, list] = {}
    for name in heldout_names:
        raw = load_imc_rds(DATASET_RDS[name], max_images=max_imgs)
        imgs = project_to_canonical_panel(raw, name)
        ood_test[name] = imgs
        log.info("  %s (held-out): OOD-test=%d", name, len(imgs))

    log.info("Pooled: train=%d val=%d  in-dist-test=%s  ood-test=%s",
             len(train_imgs), len(val_imgs),
             {k: len(v) for k, v in indist_test.items()},
             {k: len(v) for k, v in ood_test.items()})

    # ---- Tile everything ----
    log.info("Tiling ...")
    x_train = tile_set(train_imgs, args.patch_size, n_channels)
    x_val = tile_set(val_imgs, args.patch_size, n_channels)
    indist_tiles = {k: tile_set(v, args.patch_size, n_channels) for k, v in indist_test.items()}
    ood_tiles = {k: tile_set(v, args.patch_size, n_channels) for k, v in ood_test.items()}
    log.info("tile counts: train=%d val=%d  in-dist=%s  ood=%s",
             len(x_train), len(x_val),
             {k: len(v) for k, v in indist_tiles.items()},
             {k: len(v) for k, v in ood_tiles.items()})

    x_tr_t = torch.from_numpy(x_train).permute(0, 3, 1, 2).float()
    x_va_t = torch.from_numpy(x_val).permute(0, 3, 1, 2).float()
    train_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr_t),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
    )

    # ---- Model + Phikon ----
    if args.no_cond:
        log.info("Ablation mode: --no-cond set, skipping Phikon load.")
        phikon = None
        phikon_dtype = None
        cond_in = 1024
        cond_grid = 14
    else:
        log.info("Loading Phikon-v2 (frozen) ...")
        phikon = PhikonEncoder(model_id=args.phikon_model).to(args.device)
        phikon_dtype = next(phikon.model.parameters()).dtype
        cond_in = phikon.hidden_dim
        cond_grid = phikon.grid_size

    model = MaskedUNetV2(
        n_channels=n_channels, base=args.base,
        cond_in=cond_in, cond_dim=args.cond_dim, cond_grid=cond_grid,
    ).to(args.device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    bio_pool_size = n_channels - len(always_observed)
    k_min, k_max = 1, bio_pool_size - 1
    log.info("Mask-k range (bio pool of %d): [%d, %d]", bio_pool_size, k_min, k_max)

    def encode_cond(full):
        if phikon is None:
            return None
        return encode_phikon_batch(full, dna_channels, bio_idx_for_he,
                                    phikon, args.device, phikon_dtype).float()

    history = {"train_loss": [], "val_loss_random_mask": [], "val_loss_fixed_mask": []}
    best_val = float("inf")
    best_state = None
    t0 = time.time()

    # ---- Stage 1 only (random-mask pre-training; no stage 2) ----
    log.info("=== Stage 1: random-mask pre-training (%d epochs) ===", args.epochs)
    for ep in range(args.epochs):
        model.train()
        losses = []
        for (xb,) in train_dl:
            xb = xb.to(args.device, non_blocking=True)
            B = xb.shape[0]
            cond = encode_cond(xb)
            if args.no_random_mask:
                mask = fixed_mask(n_channels, eval_obs, B, device=args.device)
            else:
                mask = random_mask_with_always_observed(
                    n_channels=n_channels, batch_size=B,
                    always_observed=always_observed,
                    k_min=k_min, k_max=k_max, device=args.device,
                )
            inp = build_masked_input(xb, mask)
            pred = model(inp, cond)
            loss_mask = (1.0 - mask).view(B, n_channels, 1, 1)
            sq = (pred - xb) ** 2 * loss_mask
            denom = loss_mask.sum() * pred.shape[-1] * pred.shape[-2] + 1e-8
            loss = sq.sum() / denom
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        tl = float(np.mean(losses))

        # Validation (pooled across train datasets, batched to avoid OOM)
        model.eval()
        vl_rand = vl_fix = float("nan")
        if len(x_va_t) > 0:
            with torch.no_grad():
                rand_num = rand_den = 0.0
                fix_num = fix_den = 0.0
                for i in range(0, len(x_va_t), args.batch_size):
                    xv = x_va_t[i:i + args.batch_size].to(args.device)
                    B = xv.shape[0]
                    cond_v = encode_cond(xv)
                    mv = random_mask_with_always_observed(
                        n_channels=n_channels, batch_size=B,
                        always_observed=always_observed,
                        k_min=k_min, k_max=k_max, device=args.device,
                    )
                    invr = build_masked_input(xv, mv)
                    pvr = model(invr, cond_v)
                    lm = (1.0 - mv).view(B, n_channels, 1, 1)
                    rand_num += float((((pvr - xv) ** 2) * lm).sum())
                    rand_den += float(lm.sum() * pvr.shape[-1] * pvr.shape[-2])

                    mvf = fixed_mask(n_channels, eval_obs, B, device=args.device)
                    invf = build_masked_input(xv, mvf)
                    pvf = model(invf, cond_v)
                    lmf = (1.0 - mvf).view(B, n_channels, 1, 1)
                    fix_num += float((((pvf - xv) ** 2) * lmf).sum())
                    fix_den += float(lmf.sum() * pvf.shape[-1] * pvf.shape[-2])
                vl_rand = rand_num / (rand_den + 1e-8)
                vl_fix = fix_num / (fix_den + 1e-8)

        history["train_loss"].append(tl)
        history["val_loss_random_mask"].append(vl_rand)
        history["val_loss_fixed_mask"].append(vl_fix)
        log.info("ep %d train=%.4f val_rand=%.4f val_fixed=%.4f",
                 ep, tl, vl_rand, vl_fix)

        if not np.isnan(vl_fix) and vl_fix < best_val:
            best_val = vl_fix
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        if args.save_checkpoint:
            torch.save(best_state, args.out_dir / "model.pt")
            log.info("Saved checkpoint.")

    # ---- Evaluation helper: per-dataset PCC on fixed eval_obs ----
    def eval_tiles(tiles: np.ndarray, tag: str) -> dict:
        if len(tiles) == 0:
            return {"n_tiles": 0}
        xt = torch.from_numpy(tiles).permute(0, 3, 1, 2).float()
        preds = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(xt), args.batch_size):
                xb = xt[i:i + args.batch_size].to(args.device)
                B = xb.shape[0]
                cond_b = encode_cond(xb)
                m = fixed_mask(n_channels, eval_obs, B, device=args.device)
                inp = build_masked_input(xb, m)
                preds.append(model(inp, cond_b).cpu().numpy())
        pred = np.concatenate(preds, axis=0)
        pred = np.moveaxis(pred, 1, -1)
        pred_tgt = pred[..., target_idx]
        y_tgt = tiles[..., target_idx]
        N, H, W, C_tgt = pred_tgt.shape
        pcc = pearson_correlation(
            pred_tgt.reshape(N * H, W, C_tgt),
            y_tgt.reshape(N * H, W, C_tgt),
            per_channel=True,
        )
        mse = mean_squared_error(pred_tgt, y_tgt)
        pcc_dict = {canonical[target_idx[i]]: float(pv) for i, pv in enumerate(pcc)}
        mean_pcc = float(np.nanmean(pcc))
        log.info("EVAL %s  n_tiles=%d  mean_pcc=%.3f  mse=%.4f", tag, len(tiles), mean_pcc, mse)
        return {
            "n_tiles": int(len(tiles)),
            "mean_pcc": mean_pcc,
            "mse": float(mse),
            "pcc_per_marker": pcc_dict,
        }

    eval_results = {
        "in_dist": {k: eval_tiles(v, f"in-dist/{k}") for k, v in indist_tiles.items()},
        "ood": {k: eval_tiles(v, f"ood/{k}") for k, v in ood_tiles.items()},
    }

    metrics = {
        "model": "spaprotfm_v2_xds" + ("_no_cond" if args.no_cond else ""),
        "run_name": args.run_name,
        "seed": args.seed,
        "train_datasets": train_ds_names,
        "heldout_datasets": heldout_names,
        "canonical_markers": canonical,
        "always_observed_idx": always_observed,
        "eval_observed_channels": [canonical[i] for i in eval_obs],
        "eval_target_channels": [canonical[i] for i in target_idx],
        "n_train_images": len(train_imgs),
        "n_val_images": len(val_imgs),
        "n_train_tiles": int(len(x_train)),
        "n_val_tiles": int(len(x_val)),
        "epochs": args.epochs,
        "base": args.base,
        "cond_dim": args.cond_dim,
        "phikon_model": args.phikon_model if phikon is not None else None,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "mask_k_min": k_min,
        "mask_k_max": k_max,
        "use_cond": phikon is not None,
        "best_val_loss_fixed_mask": best_val,
        "eval": eval_results,
        "train_wall_seconds": time.time() - t0,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    log.info("Wrote metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
