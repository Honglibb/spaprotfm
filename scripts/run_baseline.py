"""Train + evaluate Murphy U-Net baseline on Damond IMC.

Usage:
    uv run python scripts/run_baseline.py \
        --rds /code/zkgy/hongliyin_computer/data/raw/imc/damond_2019_pancreas_images.rds \
        --out-dir results/baseline_damond \
        --n-observed 10 \
        --epochs 20 \
        --patch-size 128 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from spaprotfm.baselines.murphy_unet import MurphyUNetBaseline
from spaprotfm.data.bodenmiller import load_imc_rds
from spaprotfm.data.normalization import normalize_image
from spaprotfm.data.tiling import tile_image
from spaprotfm.eval.metrics import mean_squared_error, pearson_correlation

log = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rds", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n-observed", type=int, default=10,
                   help="Number of observed markers (first N channels)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-images", type=int, default=0,
                   help="For debugging: limit images loaded (0 = all)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    log.info("Loading %s ...", args.rds)
    images = load_imc_rds(str(args.rds), max_images=args.max_images or None)
    log.info("Loaded %d images", len(images))
    channel_names = images[0].channel_names
    n_channels = len(channel_names)
    log.info("Panel: %d channels — %s", n_channels, channel_names)

    n_obs = args.n_observed
    obs_idx = list(range(n_obs))
    tgt_idx = list(range(n_obs, n_channels))
    log.info("Observed (%d): %s", n_obs, [channel_names[i] for i in obs_idx])
    log.info("Target (%d): %s", len(tgt_idx), [channel_names[i] for i in tgt_idx])

    # Normalize + tile each image, aggregate
    tiles_list = []
    for img in images:
        normed = normalize_image(img.image).astype(np.float32)
        tiles, _ = tile_image(normed, patch_size=args.patch_size, stride=args.patch_size)
        tiles_list.append(tiles)
    all_tiles = np.concatenate(tiles_list, axis=0)
    log.info("Total tiles: %d of shape %s", all_tiles.shape[0], all_tiles.shape[1:])

    # 80/10/10 split by tile
    n_total = all_tiles.shape[0]
    idx = rng.permutation(n_total)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    tr, va, te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    x_all = all_tiles[:, :, :, obs_idx]
    y_all = all_tiles[:, :, :, tgt_idx]

    log.info("train=%d  val=%d  test=%d", len(tr), len(va), len(te))

    model = MurphyUNetBaseline(
        in_channels=len(obs_idx), out_channels=len(tgt_idx),
        device=args.device, base=32,
    )
    history = model.fit(
        x_all[tr], y_all[tr], x_all[va], y_all[va],
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
    )

    # Final test metrics
    pred = model.predict(x_all[te], batch_size=args.batch_size)
    y_te = y_all[te]

    # pearson_correlation(per_channel=True) expects (H, W, C); reshape (N,H,W,C) → (N*H, W, C)
    N, H, W, C = pred.shape
    pred_flat = pred.reshape(N * H, W, C)
    y_te_flat = y_te.reshape(N * H, W, C)
    pcc_per_marker = pearson_correlation(pred_flat, y_te_flat, per_channel=True)
    mse = mean_squared_error(pred, y_te)

    metrics = {
        "dataset": str(args.rds),
        "n_images": len(images),
        "patch_size": args.patch_size,
        "n_tiles_total": int(n_total),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "n_observed_markers": n_obs,
        "n_target_markers": len(tgt_idx),
        "observed_markers": [channel_names[i] for i in obs_idx],
        "target_markers": [channel_names[i] for i in tgt_idx],
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "best_val_loss": min(history["val_loss"]),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "test_mse": mse,
        "test_mean_pcc": float(np.nanmean(pcc_per_marker)),
        "test_pcc_per_marker": pcc_per_marker.tolist(),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    log.info("=== RESULTS ===")
    log.info("Test MSE: %.4f", mse)
    log.info("Test mean PCC: %.3f", metrics["test_mean_pcc"])
    log.info("Per-marker PCC: %s", {channel_names[tgt_idx[i]]: round(float(pv), 3) for i, pv in enumerate(pcc_per_marker)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
