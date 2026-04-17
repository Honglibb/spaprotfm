"""Train + evaluate Murphy U-Net baseline on Bodenmiller IMC datasets.

Usage:
    uv run python scripts/run_baseline.py \
        --rds /code/zkgy/hongliyin_computer/data/raw/imc/Damond_2019_Pancreas_images.rds \
        --out-dir results/baseline_damond \
        --dataset-name damond_pancreas \
        --n-observed 10 \
        --epochs 30 \
        --patch-size 128 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import time
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
    p.add_argument("--dataset-name", default="",
                   help="Human-readable dataset identifier saved into metrics.json")
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

    t_start = time.time()

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

    # --- Image-level 80/10/10 split (fix: prevent tile leakage across splits) ---
    n_images = len(images)
    image_idx = rng.permutation(n_images)
    n_train_img = int(0.8 * n_images)
    n_val_img = int(0.1 * n_images)
    train_img_idx = image_idx[:n_train_img]
    val_img_idx   = image_idx[n_train_img:n_train_img + n_val_img]
    test_img_idx  = image_idx[n_train_img + n_val_img:]

    log.info("Image split — train: %d  val: %d  test: %d",
             len(train_img_idx), len(val_img_idx), len(test_img_idx))

    # Save the image-id assignment for reproducibility
    def img_name(i: int) -> str:
        img = images[i]
        # Use .name attribute if available, else index
        return getattr(img, "name", None) or getattr(img, "image_name", None) or str(i)

    image_ids_split = {
        "train": [img_name(i) for i in train_img_idx.tolist()],
        "val":   [img_name(i) for i in val_img_idx.tolist()],
        "test":  [img_name(i) for i in test_img_idx.tolist()],
    }
    (args.out_dir / "image_ids.json").write_text(json.dumps(image_ids_split, indent=2))
    log.info("Image ID split written to %s/image_ids.json", args.out_dir)

    def tile_subset(idx_list):
        """Tile the images at the given indices and return (X, Y) arrays."""
        xs, ys = [], []
        for i in idx_list:
            img = images[i]
            normed = normalize_image(img.image).astype(np.float32)
            tiles, _ = tile_image(normed, patch_size=args.patch_size, stride=args.patch_size)
            xs.append(tiles[:, :, :, obs_idx])
            ys.append(tiles[:, :, :, tgt_idx])
        if not xs:
            # Empty split (can happen with very few images in smoke tests)
            dummy_shape = (0, args.patch_size, args.patch_size)
            return (np.zeros(dummy_shape + (len(obs_idx),), dtype=np.float32),
                    np.zeros(dummy_shape + (len(tgt_idx),), dtype=np.float32))
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    log.info("Tiling train images ...")
    x_train, y_train = tile_subset(train_img_idx)
    log.info("Tiling val images ...")
    x_val,   y_val   = tile_subset(val_img_idx)
    log.info("Tiling test images ...")
    x_test,  y_test  = tile_subset(test_img_idx)

    n_total = x_train.shape[0] + x_val.shape[0] + x_test.shape[0]
    log.info("Total tiles: %d  (train=%d  val=%d  test=%d)",
             n_total, x_train.shape[0], x_val.shape[0], x_test.shape[0])

    model = MurphyUNetBaseline(
        in_channels=len(obs_idx), out_channels=len(tgt_idx),
        device=args.device, base=32,
    )
    history = model.fit(
        x_train, y_train, x_val, y_val,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
    )

    # Final test metrics
    pred = model.predict(x_test, batch_size=args.batch_size)

    # pearson_correlation(per_channel=True) expects (H, W, C); reshape (N,H,W,C) → (N*H, W, C)
    N, H, W, C = pred.shape
    pred_flat  = pred.reshape(N * H, W, C)
    y_te_flat  = y_test.reshape(N * H, W, C)
    pcc_per_marker = pearson_correlation(pred_flat, y_te_flat, per_channel=True)
    mse = mean_squared_error(pred, y_test)

    t_elapsed = time.time() - t_start

    metrics = {
        "dataset_name": args.dataset_name or str(args.rds),
        "rds_path": str(args.rds),
        "n_images": n_images,
        "n_train_images": int(len(train_img_idx)),
        "n_val_images": int(len(val_img_idx)),
        "n_test_images": int(len(test_img_idx)),
        "patch_size": args.patch_size,
        "n_tiles_total": int(n_total),
        "n_train_tiles": int(x_train.shape[0]),
        "n_val_tiles": int(x_val.shape[0]),
        "n_test_tiles": int(x_test.shape[0]),
        "n_observed_markers": n_obs,
        "n_target_markers": len(tgt_idx),
        "observed_markers": [channel_names[i] for i in obs_idx],
        "target_markers": [channel_names[i] for i in tgt_idx],
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "best_val_loss": min(history["val_loss"]),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "test_mse": mse,
        "test_mean_pcc": float(np.nanmean(pcc_per_marker)),
        "test_pcc_per_marker": {
            channel_names[tgt_idx[i]]: float(pv)
            for i, pv in enumerate(pcc_per_marker)
        },
        "train_wall_seconds": round(t_elapsed, 1),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    log.info("=== RESULTS ===")
    log.info("Dataset: %s", metrics["dataset_name"])
    log.info("Test MSE: %.4f", mse)
    log.info("Test mean PCC: %.3f", metrics["test_mean_pcc"])
    log.info("Per-marker PCC: %s",
             {channel_names[tgt_idx[i]]: round(float(pv), 3) for i, pv in enumerate(pcc_per_marker)})
    log.info("Wall-clock: %.1f s", t_elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
