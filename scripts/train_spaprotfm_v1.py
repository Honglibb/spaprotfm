"""Train SpaProtFM v1 (masked U-Net with per-channel bias + two-stage training) on Bodenmiller IMC data."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from spaprotfm.data.bodenmiller import load_imc_rds
from spaprotfm.data.normalization import normalize_image
from spaprotfm.data.tiling import tile_image
from spaprotfm.eval.metrics import mean_squared_error, pearson_correlation
from spaprotfm.models.spaprotfm_v0 import build_masked_input, fixed_mask
from spaprotfm.models.spaprotfm_v1 import MaskedUNetV1, random_mask_with_always_observed

log = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rds", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--n-eval-observed", type=int, default=10,
                   help="At eval time, observe first N channels (matches baseline)")
    p.add_argument("--epochs", type=int, default=30,
                   help="Stage 1: random-mask pre-training epochs")
    p.add_argument("--finetune-epochs", type=int, default=5,
                   help="Stage 2: fixed-mask fine-tuning epochs")
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--mask-k-min", type=int, default=3)
    p.add_argument("--mask-k-max", type=int, default=0,
                   help="0 = use n_bio_channels-1 as max")
    p.add_argument("--non-bio-channels", type=str, default="",
                   help="Comma-separated list of channel indices that are always observed (never masked). "
                        "E.g. Ruthenium calibration probes in JacksonFischer.")
    p.add_argument("--base", type=int, default=48, help="U-Net base channel width")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Parse non-bio-channels
    non_bio_channels: list[int] = []
    if args.non_bio_channels.strip():
        non_bio_channels = [int(x.strip()) for x in args.non_bio_channels.split(",") if x.strip()]

    log.info("Loading %s ...", args.rds)
    images = load_imc_rds(str(args.rds), max_images=args.max_images or None)
    n_images = len(images)
    channel_names = images[0].channel_names
    n_channels = len(channel_names)
    log.info("Loaded %d images, %d channels", n_images, n_channels)

    if non_bio_channels:
        non_bio_names = [channel_names[i] for i in non_bio_channels if i < n_channels]
        log.info("Non-bio channels (always observed, never masked): %s = %s",
                 non_bio_channels, non_bio_names)

    # Image-level split (same as baseline)
    perm = rng.permutation(n_images)
    n_train = int(0.8 * n_images)
    n_val = int(0.1 * n_images)
    train_imgs = [images[i] for i in perm[:n_train]]
    val_imgs = [images[i] for i in perm[n_train:n_train + n_val]]
    test_imgs = [images[i] for i in perm[n_train + n_val:]]

    def tile_set(imgs: list) -> np.ndarray:
        if not imgs:
            return np.zeros((0, args.patch_size, args.patch_size, n_channels), dtype=np.float32)
        out = []
        for im in imgs:
            normed = normalize_image(im.image).astype(np.float32)
            tiles, _ = tile_image(normed, patch_size=args.patch_size, stride=args.patch_size)
            out.append(tiles)
        return np.concatenate(out, axis=0)

    log.info("Tiling train images (%d) ...", len(train_imgs))
    x_train = tile_set(train_imgs)
    log.info("Tiling val images (%d) ...", len(val_imgs))
    x_val = tile_set(val_imgs)
    log.info("Tiling test images (%d) ...", len(test_imgs))
    x_test = tile_set(test_imgs)
    log.info("tiles train=%d val=%d test=%d", len(x_train), len(x_val), len(x_test))

    # Build tensors
    x_tr_t = torch.from_numpy(x_train).permute(0, 3, 1, 2).float()
    x_va_t = torch.from_numpy(x_val).permute(0, 3, 1, 2).float()
    x_te_t = torch.from_numpy(x_test).permute(0, 3, 1, 2).float()

    train_ds = torch.utils.data.TensorDataset(x_tr_t)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                           shuffle=True, num_workers=2, pin_memory=True)

    model = MaskedUNetV1(n_channels=n_channels, base=args.base).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Compute k_max for the bio-channel pool (channels outside non_bio_channels)
    bio_pool_size = n_channels - len(non_bio_channels)
    k_max = args.mask_k_max if args.mask_k_max > 0 else (bio_pool_size - 1)
    log.info("Mask-k range (bio pool of %d): [%d, %d]", bio_pool_size, args.mask_k_min, k_max)

    eval_obs = list(range(args.n_eval_observed))
    log.info("Eval observed (fixed for comparability): first %d channels = %s",
             args.n_eval_observed, [channel_names[i] for i in eval_obs])

    history: dict = {
        "stage1_train_loss": [], "stage1_val_loss_random_mask": [], "stage1_val_loss_fixed_mask": [],
        "stage2_train_loss": [], "stage2_val_loss_fixed_mask": [],
    }
    best_val = float("inf")
    best_state = None
    t0 = time.time()

    # ==================== STAGE 1: Random-mask pre-training ====================
    log.info("=== Stage 1: random-mask pre-training (%d epochs) ===", args.epochs)
    for ep in range(args.epochs):
        model.train()
        losses = []
        for (xb,) in train_dl:
            xb = xb.to(args.device, non_blocking=True)
            B = xb.shape[0]
            mask = random_mask_with_always_observed(
                n_channels=n_channels,
                batch_size=B,
                always_observed=non_bio_channels if non_bio_channels else None,
                k_min=args.mask_k_min,
                k_max=k_max,
                device=args.device,
            )
            inp = build_masked_input(xb, mask)
            pred = model(inp)
            # Loss only on UNobserved (mask == 0) positions
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
        with torch.no_grad():
            xv = x_va_t.to(args.device)
            B = xv.shape[0]
            # Random mask val
            mv = random_mask_with_always_observed(
                n_channels=n_channels,
                batch_size=B,
                always_observed=non_bio_channels if non_bio_channels else None,
                k_min=args.mask_k_min,
                k_max=k_max,
                device=args.device,
            )
            inv_ = build_masked_input(xv, mv)
            pv = model(inv_)
            lm = (1.0 - mv).view(B, n_channels, 1, 1)
            vl_rand = float((((pv - xv) ** 2) * lm).sum() / (lm.sum() * pv.shape[-1] * pv.shape[-2] + 1e-8))

            # Fixed eval mask val (apples-to-apples with baseline)
            mvf = fixed_mask(n_channels, eval_obs, B, device=args.device)
            invf = build_masked_input(xv, mvf)
            pvf = model(invf)
            lmf = (1.0 - mvf).view(B, n_channels, 1, 1)
            vl_fix = float((((pvf - xv) ** 2) * lmf).sum() / (lmf.sum() * pvf.shape[-1] * pvf.shape[-2] + 1e-8))

        history["stage1_train_loss"].append(tl)
        history["stage1_val_loss_random_mask"].append(vl_rand)
        history["stage1_val_loss_fixed_mask"].append(vl_fix)
        log.info("S1 ep %d  train=%.4f  val_random=%.4f  val_fixed=%.4f", ep, tl, vl_rand, vl_fix)

        if vl_fix < best_val:
            best_val = vl_fix
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ==================== STAGE 2: Fixed-mask fine-tuning ====================
    log.info("=== Stage 2: fixed-mask fine-tuning (%d epochs) on eval config ===", args.finetune_epochs)

    # Lower LR for fine-tuning
    for pg in opt.param_groups:
        pg["lr"] = args.lr * 0.1

    for ep in range(args.finetune_epochs):
        model.train()
        losses = []
        for (xb,) in train_dl:
            xb = xb.to(args.device, non_blocking=True)
            B = xb.shape[0]
            # Always use the eval-time fixed mask in stage 2
            mask = fixed_mask(n_channels, eval_obs, B, device=args.device)
            inp = build_masked_input(xb, mask)
            pred = model(inp)
            loss_mask = (1.0 - mask).view(B, n_channels, 1, 1)
            sq = (pred - xb) ** 2 * loss_mask
            denom = loss_mask.sum() * pred.shape[-1] * pred.shape[-2] + 1e-8
            loss = sq.sum() / denom
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        tl = float(np.mean(losses))

        # Validation with fixed mask
        model.eval()
        with torch.no_grad():
            xv = x_va_t.to(args.device)
            B = xv.shape[0]
            mvf = fixed_mask(n_channels, eval_obs, B, device=args.device)
            invf = build_masked_input(xv, mvf)
            pvf = model(invf)
            lmf = (1.0 - mvf).view(B, n_channels, 1, 1)
            vl_fix = float((((pvf - xv) ** 2) * lmf).sum() / (lmf.sum() * pvf.shape[-1] * pvf.shape[-2] + 1e-8))

        history["stage2_train_loss"].append(tl)
        history["stage2_val_loss_fixed_mask"].append(vl_fix)
        log.info("S2 ep %d  train=%.4f  val_fixed=%.4f", ep, tl, vl_fix)

        if vl_fix < best_val:
            best_val = vl_fix
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # === Test-time evaluation with FIXED 10 observed channels ===
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x_te_t), args.batch_size):
            xb = x_te_t[i:i + args.batch_size].to(args.device)
            B = xb.shape[0]
            m = fixed_mask(n_channels, eval_obs, B, device=args.device)
            inp = build_masked_input(xb, m)
            preds.append(model(inp).cpu().numpy())
    pred = np.concatenate(preds, axis=0)  # (N, C, H, W)
    pred = np.moveaxis(pred, 1, -1)        # (N, H, W, C)

    # Compute PCC ONLY on the unobserved channels (target)
    target_idx = [i for i in range(n_channels) if i not in eval_obs]
    pred_tgt = pred[..., target_idx]
    y_tgt = x_test[..., target_idx]

    # Flatten for pearson_correlation: (N*H, W, C)
    N, H, W, C_tgt = pred_tgt.shape
    pred_flat = pred_tgt.reshape(N * H, W, C_tgt)
    y_flat = y_tgt.reshape(N * H, W, C_tgt)
    pcc_per_marker = pearson_correlation(pred_flat, y_flat, per_channel=True)
    mse = mean_squared_error(pred_tgt, y_tgt)

    # Headline PCC: mean over ALL target channels
    all_target_pcc = float(np.nanmean(pcc_per_marker))

    # Biological target PCC: exclude non-bio channels from the target set
    # target_idx are the unobserved channels; bio targets = those not in non_bio_channels
    non_bio_set = set(non_bio_channels)
    bio_target_local_idx = [
        local_i for local_i, global_i in enumerate(target_idx)
        if global_i not in non_bio_set
    ]
    if bio_target_local_idx:
        bio_pcc = float(np.nanmean([pcc_per_marker[i] for i in bio_target_local_idx]))
    else:
        bio_pcc = all_target_pcc

    log.info("=== RESULTS for %s ===", args.dataset_name)
    log.info("Test mean PCC all targets (eval-time first %d observed): %.3f",
             args.n_eval_observed, all_target_pcc)
    log.info("Test mean PCC bio targets only: %.3f", bio_pcc)
    log.info("Test MSE: %.4f", mse)
    log.info("Train wall: %.1f s", time.time() - t0)

    pcc_dict = {
        channel_names[target_idx[i]]: float(pv)
        for i, pv in enumerate(pcc_per_marker)
    }

    metrics = {
        "model": "spaprotfm_v1_masked_unet",
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
        "lr": args.lr,
        "batch_size": args.batch_size,
        "mask_k_min": args.mask_k_min,
        "mask_k_max": k_max,
        "non_bio_channels": non_bio_channels,
        "non_bio_channel_names": [channel_names[i] for i in non_bio_channels if i < n_channels],
        "eval_observed_n": args.n_eval_observed,
        "eval_observed_channels": [channel_names[i] for i in eval_obs],
        "eval_target_channels": [channel_names[i] for i in target_idx],
        "best_val_loss_fixed_mask": best_val,
        "test_mse": mse,
        "test_mean_pcc_all_targets": all_target_pcc,
        "test_mean_pcc_bio_targets": bio_pcc,
        # Headline metric (bio targets if available, else all targets)
        "test_mean_pcc": bio_pcc if non_bio_channels else all_target_pcc,
        "test_pcc_per_marker": pcc_dict,
        "train_wall_seconds": time.time() - t0,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
