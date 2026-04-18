"""Train SpaProtFM v2: MaskedUNetV1 + pseudo-H&E conditioning via Phikon-v2.

Same 2-stage training recipe as v1:
  stage 1: random-mask pre-training (DNA channels always observed)
  stage 2: fixed-mask fine-tuning on the eval panel

At every forward pass we synthesize a pseudo-H&E RGB from the *full* tile's
DNA channels and feed it through a frozen Phikon-v2 encoder; the 14x14
feature map is fused at the U-Net bottleneck. DNA channels are forced to be
always-observed because the pseudo-H&E is derived from them.
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
from spaprotfm.data.bodenmiller import load_imc_rds
from spaprotfm.data.normalization import normalize_image
from spaprotfm.data.tiling import tile_image
from spaprotfm.eval.metrics import mean_squared_error, pearson_correlation
from spaprotfm.models.spaprotfm_v0 import build_masked_input, fixed_mask
from spaprotfm.models.spaprotfm_v1 import random_mask_with_always_observed
from spaprotfm.models.spaprotfm_v2 import MaskedUNetV2

log = logging.getLogger(__name__)


def parse_indices(arg: str) -> list[int]:
    if not arg.strip():
        return []
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def encode_phikon_batch(
    full_tiles: torch.Tensor,
    dna_idx: list[int],
    bio_idx: list[int] | None,
    phikon: PhikonEncoder,
    device: str | torch.device,
    phikon_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Synthesize pseudo-H&E and run through Phikon; returns (B, 1024, 14, 14)."""
    rgb = synthesize_pseudo_he_batch(full_tiles.detach().cpu(), dna_idx, bio_idx)
    rgb = rgb.to(device)
    if phikon_dtype is not None:
        rgb = rgb.to(phikon_dtype)
    return phikon.encode(rgb)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rds", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--dna-channels", required=True,
                   help="Comma-separated channel indices for DNA (pseudo-H&E hematoxylin source). "
                        "E.g. '36,37' for Damond DNA1/DNA2.")
    p.add_argument("--n-eval-observed", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30,
                   help="Stage 1: random-mask pre-training epochs")
    p.add_argument("--finetune-epochs", type=int, default=5,
                   help="Stage 2: fixed-mask fine-tuning epochs")
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--mask-k-min", type=int, default=3)
    p.add_argument("--mask-k-max", type=int, default=0)
    p.add_argument("--non-bio-channels", type=str, default="",
                   help="Extra channels that should never be masked (e.g. Ru in Jackson). "
                        "DNA channels are always implicitly always-observed.")
    p.add_argument("--base", type=int, default=48)
    p.add_argument("--cond-dim", type=int, default=192)
    p.add_argument("--panel-sweep", action="store_true")
    p.add_argument("--sweep-sizes", type=str, default="3,7,10,15,20")
    p.add_argument("--sweep-trials", type=int, default=3)
    p.add_argument("--sweep-seed", type=int, default=0)
    p.add_argument("--phikon-model", default="owkin/phikon-v2")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    dna_channels = parse_indices(args.dna_channels)
    extra_always = parse_indices(args.non_bio_channels)
    always_observed = sorted(set(dna_channels) | set(extra_always))
    if not dna_channels:
        raise SystemExit("--dna-channels is required and must be non-empty")

    log.info("Loading %s ...", args.rds)
    images = load_imc_rds(str(args.rds), max_images=args.max_images or None)
    n_images = len(images)
    channel_names = images[0].channel_names
    n_channels = len(channel_names)
    log.info("Loaded %d images, %d channels", n_images, n_channels)
    log.info("DNA channels (pseudo-H&E source): %s = %s",
             dna_channels, [channel_names[i] for i in dna_channels])
    log.info("Always-observed channels (mask=1 always): %s = %s",
             always_observed, [channel_names[i] for i in always_observed])

    # Image-level split (same seed as v1 for apples-to-apples comparison)
    perm = rng.permutation(n_images)
    n_train = int(0.8 * n_images)
    n_val = int(0.1 * n_images)
    train_imgs = [images[i] for i in perm[:n_train]]
    val_imgs = [images[i] for i in perm[n_train:n_train + n_val]]
    test_imgs = [images[i] for i in perm[n_train + n_val:]]

    def tile_set(imgs):
        if not imgs:
            return np.zeros((0, args.patch_size, args.patch_size, n_channels), dtype=np.float32)
        out = []
        for im in imgs:
            normed = normalize_image(im.image).astype(np.float32)
            tiles, _ = tile_image(normed, patch_size=args.patch_size, stride=args.patch_size)
            out.append(tiles)
        return np.concatenate(out, axis=0)

    log.info("Tiling train/val/test ...")
    x_train = tile_set(train_imgs)
    x_val = tile_set(val_imgs)
    x_test = tile_set(test_imgs)
    log.info("tiles train=%d val=%d test=%d", len(x_train), len(x_val), len(x_test))

    x_tr_t = torch.from_numpy(x_train).permute(0, 3, 1, 2).float()
    x_va_t = torch.from_numpy(x_val).permute(0, 3, 1, 2).float()
    x_te_t = torch.from_numpy(x_test).permute(0, 3, 1, 2).float()

    train_ds = torch.utils.data.TensorDataset(x_tr_t)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                           shuffle=True, num_workers=0, pin_memory=True)

    log.info("Loading Phikon-v2 (frozen) ...")
    phikon = PhikonEncoder(model_id=args.phikon_model).to(args.device)
    # Use fp32 to keep simple; Phikon is small relative to our U-Net.
    phikon_dtype = next(phikon.model.parameters()).dtype
    cond_in = phikon.hidden_dim
    cond_grid = phikon.grid_size

    model = MaskedUNetV2(
        n_channels=n_channels, base=args.base,
        cond_in=cond_in, cond_dim=args.cond_dim, cond_grid=cond_grid,
    ).to(args.device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    bio_pool_size = n_channels - len(always_observed)
    k_max = args.mask_k_max if args.mask_k_max > 0 else (bio_pool_size - 1)
    log.info("Mask-k range (bio pool of %d): [%d, %d]", bio_pool_size, args.mask_k_min, k_max)

    eval_obs = list(range(args.n_eval_observed))
    log.info("Eval observed (first %d): %s", args.n_eval_observed,
             [channel_names[i] for i in eval_obs])

    # Bio-idx for pseudo-H&E eosin: use all non-DNA channels (reasonable default)
    bio_idx_for_he = [i for i in range(n_channels) if i not in set(dna_channels)]

    history = {
        "stage1_train_loss": [], "stage1_val_loss_random_mask": [],
        "stage1_val_loss_fixed_mask": [],
        "stage2_train_loss": [], "stage2_val_loss_fixed_mask": [],
    }
    best_val = float("inf")
    best_state = None
    t0 = time.time()

    def encode_cond(full: torch.Tensor) -> torch.Tensor:
        return encode_phikon_batch(full, dna_channels, bio_idx_for_he,
                                    phikon, args.device, phikon_dtype)

    # ==================== STAGE 1 ====================
    log.info("=== Stage 1: random-mask pre-training (%d epochs) ===", args.epochs)
    for ep in range(args.epochs):
        model.train()
        losses = []
        for (xb,) in train_dl:
            xb = xb.to(args.device, non_blocking=True)
            B = xb.shape[0]
            cond = encode_cond(xb).float()
            mask = random_mask_with_always_observed(
                n_channels=n_channels, batch_size=B,
                always_observed=always_observed if always_observed else None,
                k_min=args.mask_k_min, k_max=k_max, device=args.device,
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

        # Validation
        model.eval()
        vl_rand = float("nan")
        vl_fix = float("nan")
        if len(x_va_t) > 0:
            with torch.no_grad():
                xv = x_va_t.to(args.device)
                B = xv.shape[0]
                cond_v = encode_cond(xv).float()
                mv = random_mask_with_always_observed(
                    n_channels=n_channels, batch_size=B,
                    always_observed=always_observed if always_observed else None,
                    k_min=args.mask_k_min, k_max=k_max, device=args.device,
                )
                inv_ = build_masked_input(xv, mv)
                pv = model(inv_, cond_v)
                lm = (1.0 - mv).view(B, n_channels, 1, 1)
                vl_rand = float((((pv - xv) ** 2) * lm).sum() / (lm.sum() * pv.shape[-1] * pv.shape[-2] + 1e-8))

                mvf = fixed_mask(n_channels, eval_obs, B, device=args.device)
                invf = build_masked_input(xv, mvf)
                pvf = model(invf, cond_v)
                lmf = (1.0 - mvf).view(B, n_channels, 1, 1)
                vl_fix = float((((pvf - xv) ** 2) * lmf).sum() / (lmf.sum() * pvf.shape[-1] * pvf.shape[-2] + 1e-8))

        history["stage1_train_loss"].append(tl)
        history["stage1_val_loss_random_mask"].append(vl_rand)
        history["stage1_val_loss_fixed_mask"].append(vl_fix)
        log.info("S1 ep %d train=%.4f val_rand=%.4f val_fixed=%.4f",
                 ep, tl, vl_rand, vl_fix)

        if not np.isnan(vl_fix) and vl_fix < best_val:
            best_val = vl_fix
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ==================== STAGE 2 ====================
    log.info("=== Stage 2: fixed-mask fine-tuning (%d epochs) ===", args.finetune_epochs)
    for pg in opt.param_groups:
        pg["lr"] = args.lr * 0.1

    for ep in range(args.finetune_epochs):
        model.train()
        losses = []
        for (xb,) in train_dl:
            xb = xb.to(args.device, non_blocking=True)
            B = xb.shape[0]
            cond = encode_cond(xb).float()
            mask = fixed_mask(n_channels, eval_obs, B, device=args.device)
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

        model.eval()
        vl_fix = float("nan")
        if len(x_va_t) > 0:
            with torch.no_grad():
                xv = x_va_t.to(args.device)
                B = xv.shape[0]
                cond_v = encode_cond(xv).float()
                mvf = fixed_mask(n_channels, eval_obs, B, device=args.device)
                invf = build_masked_input(xv, mvf)
                pvf = model(invf, cond_v)
                lmf = (1.0 - mvf).view(B, n_channels, 1, 1)
                vl_fix = float((((pvf - xv) ** 2) * lmf).sum() / (lmf.sum() * pvf.shape[-1] * pvf.shape[-2] + 1e-8))

        history["stage2_train_loss"].append(tl)
        history["stage2_val_loss_fixed_mask"].append(vl_fix)
        log.info("S2 ep %d train=%.4f val_fixed=%.4f", ep, tl, vl_fix)

        if not np.isnan(vl_fix) and vl_fix < best_val:
            best_val = vl_fix
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # ==================== TEST ====================
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x_te_t), args.batch_size):
            xb = x_te_t[i:i + args.batch_size].to(args.device)
            B = xb.shape[0]
            cond_b = encode_cond(xb).float()
            m = fixed_mask(n_channels, eval_obs, B, device=args.device)
            inp = build_masked_input(xb, m)
            preds.append(model(inp, cond_b).cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    pred = np.moveaxis(pred, 1, -1)

    target_idx = [i for i in range(n_channels) if i not in eval_obs]
    pred_tgt = pred[..., target_idx]
    y_tgt = x_test[..., target_idx]
    N, H, W, C_tgt = pred_tgt.shape
    pcc_per_marker = pearson_correlation(
        pred_tgt.reshape(N * H, W, C_tgt),
        y_tgt.reshape(N * H, W, C_tgt),
        per_channel=True,
    )
    mse = mean_squared_error(pred_tgt, y_tgt)

    all_target_pcc = float(np.nanmean(pcc_per_marker))
    non_bio_set = set(extra_always)
    bio_target_local_idx = [
        local_i for local_i, global_i in enumerate(target_idx)
        if global_i not in non_bio_set
    ]
    bio_pcc = (
        float(np.nanmean([pcc_per_marker[i] for i in bio_target_local_idx]))
        if bio_target_local_idx else all_target_pcc
    )

    log.info("=== RESULTS for %s ===", args.dataset_name)
    log.info("Test mean PCC all targets (first %d observed): %.3f",
             args.n_eval_observed, all_target_pcc)
    log.info("Test mean PCC bio targets: %.3f", bio_pcc)
    log.info("Test MSE: %.4f", mse)
    log.info("Train wall: %.1f s", time.time() - t0)

    pcc_dict = {channel_names[target_idx[i]]: float(pv)
                for i, pv in enumerate(pcc_per_marker)}

    metrics = {
        "model": "spaprotfm_v2_masked_unet_phikon",
        "dataset": args.dataset_name,
        "rds_path": str(args.rds),
        "n_images": n_images,
        "n_train_images": len(train_imgs),
        "n_val_images": len(val_imgs),
        "n_test_images": len(test_imgs),
        "n_channels": n_channels,
        "patch_size": args.patch_size,
        "n_train_tiles": int(len(x_train)),
        "n_val_tiles": int(len(x_val)),
        "n_test_tiles": int(len(x_test)),
        "epochs": args.epochs,
        "finetune_epochs": args.finetune_epochs,
        "base": args.base,
        "cond_dim": args.cond_dim,
        "cond_in": cond_in,
        "phikon_model": args.phikon_model,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "mask_k_min": args.mask_k_min,
        "mask_k_max": k_max,
        "dna_channels": dna_channels,
        "dna_channel_names": [channel_names[i] for i in dna_channels],
        "non_bio_channels": extra_always,
        "always_observed_channels": always_observed,
        "eval_observed_n": args.n_eval_observed,
        "eval_observed_channels": [channel_names[i] for i in eval_obs],
        "eval_target_channels": [channel_names[i] for i in target_idx],
        "best_val_loss_fixed_mask": best_val,
        "test_mse": mse,
        "test_mean_pcc_all_targets": all_target_pcc,
        "test_mean_pcc_bio_targets": bio_pcc,
        "test_mean_pcc": bio_pcc if extra_always else all_target_pcc,
        "test_pcc_per_marker": pcc_dict,
        "train_wall_seconds": time.time() - t0,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.out_dir / "training_history.json").write_text(json.dumps(history, indent=2))

    # ==================== PANEL-SIZE SWEEP ====================
    if args.panel_sweep:
        log.info("=== Panel-size sweep ===")
        sweep_rng = np.random.default_rng(args.sweep_seed)
        sizes = [int(s) for s in args.sweep_sizes.split(",")]
        always_set = set(always_observed)
        # The pool for random selection = bio channels not already always-observed
        # Observed panel = always_observed ∪ sampled_from_pool
        pool = [i for i in range(n_channels) if i not in always_set]
        sweep_results = []
        for size in sizes:
            # Effective # of sampled channels = size - len(always_observed) so that
            # the total observed count equals `size`.
            extra_needed = size - len(always_observed)
            if extra_needed < 0:
                log.warning("size %d < |always_observed|=%d, skipping", size, len(always_observed))
                continue
            if extra_needed > len(pool):
                log.warning("size %d > pool+always %d, capping", size, len(pool) + len(always_observed))
                extra_needed = len(pool)
            for trial in range(args.sweep_trials):
                if extra_needed > 0:
                    picks = sweep_rng.choice(pool, size=extra_needed, replace=False).tolist()
                else:
                    picks = []
                obs = sorted(set(always_observed) | set(int(x) for x in picks))
                preds_sw = []
                model.eval()
                with torch.no_grad():
                    for i in range(0, len(x_te_t), args.batch_size):
                        xb = x_te_t[i:i + args.batch_size].to(args.device)
                        B = xb.shape[0]
                        cond_b = encode_cond(xb).float()
                        m = fixed_mask(n_channels, obs, B, device=args.device)
                        inp = build_masked_input(xb, m)
                        preds_sw.append(model(inp, cond_b).cpu().numpy())
                pred_sw = np.concatenate(preds_sw, axis=0)
                pred_sw = np.moveaxis(pred_sw, 1, -1)
                tgt_idx = [i for i in range(n_channels) if i not in obs]
                bio_tgt_idx = [i for i in tgt_idx if i not in set(extra_always)]
                if tgt_idx:
                    N_sw, H_sw, W_sw, _ = pred_sw.shape
                    pcc = pearson_correlation(
                        pred_sw[..., tgt_idx].reshape(N_sw * H_sw, W_sw, len(tgt_idx)),
                        x_test[..., tgt_idx].reshape(N_sw * H_sw, W_sw, len(tgt_idx)),
                        per_channel=True,
                    )
                else:
                    pcc = np.array([float("nan")])
                if bio_tgt_idx:
                    pcc_bio = pearson_correlation(
                        pred_sw[..., bio_tgt_idx].reshape(N_sw * H_sw, W_sw, len(bio_tgt_idx)),
                        x_test[..., bio_tgt_idx].reshape(N_sw * H_sw, W_sw, len(bio_tgt_idx)),
                        per_channel=True,
                    )
                else:
                    pcc_bio = pcc
                sweep_results.append({
                    "dataset": args.dataset_name,
                    "panel_size": len(obs),
                    "trial": trial,
                    "observed_indices": [int(x) for x in obs],
                    "observed_names": [channel_names[i] for i in obs],
                    "n_targets": len(tgt_idx),
                    "n_bio_targets": len(bio_tgt_idx),
                    "mean_pcc_all": float(np.nanmean(pcc)),
                    "mean_pcc_bio": float(np.nanmean(pcc_bio)),
                })
                log.info("  size=%d trial=%d → mean_pcc_all=%.3f mean_pcc_bio=%.3f",
                         len(obs), trial,
                         sweep_results[-1]["mean_pcc_all"],
                         sweep_results[-1]["mean_pcc_bio"])

        (args.out_dir / "panel_sweep.json").write_text(json.dumps(sweep_results, indent=2))
        log.info("Wrote panel_sweep.json (%d points)", len(sweep_results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
