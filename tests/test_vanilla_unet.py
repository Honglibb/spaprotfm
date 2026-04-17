import torch
from spaprotfm.baselines._vanilla_unet import UNet


def test_unet_forward_shape():
    m = UNet(in_channels=10, out_channels=28, base=16)
    x = torch.randn(2, 10, 128, 128)
    out = m(x)
    assert out.shape == (2, 28, 128, 128)


def test_unet_backward_runs():
    m = UNet(in_channels=4, out_channels=6, base=8)
    x = torch.randn(1, 4, 128, 128, requires_grad=False)
    target = torch.randn(1, 6, 128, 128)
    loss = torch.nn.functional.mse_loss(m(x), target)
    loss.backward()
    # Check a grad exists on the final layer
    assert m.out.weight.grad is not None
