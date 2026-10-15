"""Fast CPU unit test of the MF-vs-iMF JVP ablation (no MNIST needed)."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comparison_flow import MFIMFComparison


class TinyMeanVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.15))
        self.b = nn.Parameter(torch.tensor(0.31))

    def forward(self, z, t, r, y=None, w=None, return_v=False,
                use_flash_attention=False):
        t = t[:, None, None, None]
        r = r[:, None, None, None]
        return self.a * z + self.b * z.square() + (t - r) * z


def evaluate(variant):
    torch.manual_seed(17)
    np.random.seed(17)
    net = TinyMeanVelocity()
    flow = MFIMFComparison(
        channels=1, image_size=4, num_classes=10,
        normalizer=['minmax', None, None], time_dist=['uniform', 0, 1],
        variant=variant, equal_time_prob=0.5,
        adaptive_l2_gamma=1.0)
    x = torch.rand(12, 1, 4, 4)
    labels = torch.arange(12) % 10
    loss, metrics = flow.loss(net, x, labels)
    assert torch.isfinite(loss)
    assert set(metrics) == {
        'mf_loss', 'mf_v_mse', 'non_boundary_mse', 'non_boundary_count',
        'tangent_rms', 'jvp_rms'
    }
    for value in metrics.values():
        assert torch.isfinite(value), f'Nonfinite metric: {value}'
    assert metrics['non_boundary_count'].item() > 0
    loss.backward()
    assert net.a.grad is not None and torch.isfinite(net.a.grad)
    assert net.b.grad is not None and torch.isfinite(net.b.grad)
    assert torch.allclose(loss.detach(), metrics['mf_v_mse'])
    return loss.item()


def test_jvp_tangent_matters():
    """Equal settings, same random batch, but different JVP tangent."""
    mf = evaluate('mf')
    imf = evaluate('imf')
    assert abs(mf - imf) > 1e-6, 'MF/iMF must use different JVP directions'


def main():
    mf = evaluate('mf')
    imf = evaluate('imf')
    assert abs(mf - imf) > 1e-6, 'MF/iMF must use different JVP directions'
    print(f'MF/iMF smoke test passed: MF loss={mf:.5f}, iMF loss={imf:.5f}')


if __name__ == '__main__':
    main()
