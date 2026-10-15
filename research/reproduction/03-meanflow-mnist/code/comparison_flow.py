"""MF vs iMF, isolated JVP-direction comparison (MNIST).

Equation from Improved Mean Flows, Sec. 4.1 and Fig. 3:
    z_t = (1-t)x + t*e, v_c = e-x
    V_theta = u_theta(z_t,r,t) + (t-r) stopgrad(JVP(u_theta; direction))
    loss = ||V_theta - v_c||^2
MF direction: v_c; iMF direction: u_theta(z_t,t,t).

Both methods use the same architecture and training recipe. Paper Fig. 3
is ImageNet latent MeanFlow-B/2, NOT this small MNIST/DiT experiment.

Crucial numerical precaution: compute the *entire* JVP subgraph,
including boundary velocity, with autocast disabled in FP32.
No thresholding/clamping of loss, v, JVP, or its diagnostics.
"""
from functools import partial

import torch
from einops import rearrange

from meanflow import MeanFlow, adaptive_l2_loss


class MFIMFComparison(MeanFlow):
    def __init__(self, variant='imf', equal_time_prob=0.5, **kwargs):
        if variant not in ('mf', 'imf'):
            raise ValueError("variant must be 'mf' or 'imf'")
        if not (0 <= equal_time_prob <= 1):
            raise ValueError('equal_time_prob must be within [0,1]')
        if kwargs.get('cfg_scale') is not None:
            raise ValueError('Comparison must disable CFG: cfg_scale=None')
        super().__init__(mode='i-meanflow', cfg_scale=None, **kwargs)
        self.variant = variant
        self.mode = variant
        self.equal_time_prob = equal_time_prob

    def sample_t_r(self, batch_size, device, sample_cfg=True):
        t, r, w = super().sample_t_r(batch_size, device, sample_cfg=False)
        r = torch.where(torch.rand(batch_size, device=device) < self.equal_time_prob,
                        t, r)
        return t, r, w

    def loss(self, model, x, c=None):
        t, r, w = self.sample_t_r(x.shape[0], x.device)
        t4 = rearrange(t, 'b -> b 1 1 1')
        r4 = rearrange(r, 'b -> b 1 1 1')

        # The target and the interpolation path are always calculated in FP32.
        x0 = self.normer.norm(x.float())
        noise = torch.randn_like(x0)
        z = (1 - t4) * x0 + t4 * noise
        vc = noise - x0

        # Autocast can make forward-mode numerical derivatives unreliable.
        # Nested autocast(enabled=False) forces both JVP and boundary v to FP32,
        # even if outer train_repro.py has enabled BF16.
        with torch.no_grad():
            with torch.autocast(device_type=z.device.type, enabled=False):
                z32, t32, r32, w32 = z.float(), t.float(), r.float(), w.float()
                if self.variant == 'mf':
                    tangent = vc.detach()
                else:
                    tangent = model(z32, t32, t32, y=c, w=w32,
                                    return_v=False, use_flash_attention=False).detach()
                fn = partial(model, y=c, w=w32, return_v=False,
                             use_flash_attention=False)
                _, dudt = torch.autograd.functional.jvp(
                    fn, (z32, t32, r32),
                    (tangent, torch.ones_like(t32), torch.zeros_like(r32)),
                    create_graph=False, strict=False,
                )

        # The only parameter-gradient path for the objective is u_theta.
        # Gradient does not propagate back through either the JVP or tangent.
        u_pred = model(z, t, r, y=c, w=w, return_v=False)
        estimated_v = u_pred.float() + (t4 - r4) * dudt.detach()
        error = estimated_v - vc.detach()
        loss = adaptive_l2_loss(
            error, gamma=self.adaptive_l2_gamma, c=self.adaptive_l2_c
        )

        mse_each = error.detach().square().flatten(1).mean(dim=1)
        is_non_boundary = (t > r)
        # Exact non-boundary sample MSE for the paper-Fig.-3-style plot.
        non_boundary_mse = (mse_each[is_non_boundary].mean()
                            if is_non_boundary.any()
                            else mse_each.new_zeros(()))
        metrics = {
            'mf_loss': loss.detach(),
            'mf_v_mse': mse_each.mean().detach(),
            'non_boundary_mse': non_boundary_mse.detach(),
            'non_boundary_count': is_non_boundary.sum().detach(),
            'tangent_rms': tangent.detach().float().square().mean().sqrt(),
            'jvp_rms': dudt.detach().float().square().mean().sqrt(),
        }
        return loss, metrics
