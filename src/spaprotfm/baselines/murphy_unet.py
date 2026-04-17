"""Panel-extension baseline using a vanilla U-Net (Murphy-style)."""

from __future__ import annotations

import logging

import numpy as np
import torch

from spaprotfm.baselines._vanilla_unet import UNet

logger = logging.getLogger(__name__)


class MurphyUNetBaseline:
    """Fit/predict with a U-Net mapping (N, H, W, C_in) → (N, H, W, C_out).

    Uses Adam + MSE. Stores the best-val-loss state dict internally.
    """

    def __init__(self, in_channels: int, out_channels: int, device: str = "cuda:0", base: int = 32):
        self.device = device
        self.model = UNet(in_channels=in_channels, out_channels=out_channels, base=base).to(device)

    def fit(
        self,
        x_train: np.ndarray, y_train: np.ndarray,
        x_val: np.ndarray, y_val: np.ndarray,
        epochs: int = 30, lr: float = 1e-4, batch_size: int = 16,
    ) -> dict:
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        to_t = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).float()
        x_tr, y_tr = to_t(x_train), to_t(y_train)
        x_va, y_va = to_t(x_val).to(self.device), to_t(y_val).to(self.device)

        ds = torch.utils.data.TensorDataset(x_tr, y_tr)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

        history: dict = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_state = None

        for ep in range(epochs):
            self.model.train()
            losses: list[float] = []
            for xb, yb in dl:
                xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            tl = float(np.mean(losses))

            self.model.eval()
            with torch.no_grad():
                vl = float(loss_fn(self.model(x_va), y_va).item())
            history["train_loss"].append(tl)
            history["val_loss"].append(vl)
            logger.info("epoch %d  train=%.4f  val=%.4f", ep, tl, vl)

            if vl < best_val:
                best_val = vl
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def predict(self, x: np.ndarray, batch_size: int = 16) -> np.ndarray:
        self.model.eval()
        x_t = torch.from_numpy(x).permute(0, 3, 1, 2).float()
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(x_t), batch_size):
                xb = x_t[i:i + batch_size].to(self.device)
                outs.append(self.model(xb).cpu().numpy())
        out = np.concatenate(outs, axis=0)  # (N, C_out, H, W)
        return np.moveaxis(out, 1, -1)
