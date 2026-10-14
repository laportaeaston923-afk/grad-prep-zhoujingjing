"""Controlled original-MF vs improved-MF ablation (no CFG, no v-head).

Both variants share the same mean-velocity model, data path, objective
and time sampling. Only the JVP tangent differs:
  MF  : e - x          (sample/conditional velocity)
  iMF : u(z_t, t, t)   (predicted marginal velocity, boundary condition)

This mirrors the algorithmic distinction in Improved Mean Flows, Sec. 4.1,
without the paper's auxiliary v-head or flexible-guidance experiments.
"""

from functools import partial

import torch
from einops import rearrange

from meanflow import MeanFlow, adaptive_l2_loss, stopgrad


class MFIMFComparison(MeanFlow):
    def __init__(self, variant='imf', equal_time_prob=0.5, **kwargs):
        if variant not in ('mf', 'imf'):
            raise ValueError("variant must be 'mf' or 'imf'")
        if not (0.0 <= equal_time_prob <= 1.0):
            raise ValueError('equal_time_prob must be between 0 and 1')
        if kwargs.get('cfg_scale') is not None:
            raise ValueError('This controlled comparison requires cfg_scale=None')
        super().__init__(mode='i-meanflow', cfg_scale=None, **kwargs)
        self.variant = variant
        self.mode = variant
        self.equal_time_prob = equal_time_prob

    def sample_t_r(self, batch_size, device, sample_cfg=True):
        t, r, w = super().sample_t_r(batch_size, device, sample_cfg=False)
        same_time = torch.rand(batch_size, device=device) < self.equal_time_prob
        r = torch.where(same_time, t, r)
        return t, r, w

    def loss(self, model, x, c=None):
        batch_size = x.shape[0]
        t, r, w = self.sample_t_r(batch_size, x.device)
        t4 = rearrange(t, 'b -> b 1 1 1')
        r4 = rearrange(r, 'b -> b 1 1 1')

        e = torch.randn_like(x)
        x = self.normer.norm(x)
        z = (1 - t4) * x + t4 * e
        conditional_v = e - x

        with torch.no_grad():
            if self.variant == 'imf':
                # Reuse u-network at r=t: v_theta(z_t,t)=u_theta(z_t,t,t).
                tangent = model(z, t, t, y=c, w=w, return_v=False)
            else:
                tangent = conditional_v

            fn = partial(model, y=c, w=w, return_v=False,
                         use_flash_attention=False)
            _, dudt = torch.autograd.functional.jvp(
                fn, (z, t, r),
                (tangent, torch.ones_like(t), torch.zeros_like(r)),
                create_graph=False,
            )

        u_pred = model(z, t, r, y=c, w=w, return_v=False)
        # MF's stopped u-target and iMF's stopped V-loss are equivalent
        # under fixed JVP tangent; the only treatment difference is tangent.
        estimated_v = u_pred + (t4 - r4) * stopgrad(dudt)
        error = estimated_v - stopgrad(conditional_v)
        loss = adaptive_l2_loss(error,
                                gamma=self.adaptive_l2_gamma,
                                c=self.adaptive_l2_c)
        # Figure 3 compares the non-boundary samples (t != r).
        per_sample_mse = error.square().flatten(1).mean(dim=1)
        non_boundary = (t > r).to(per_sample_mse.dtype)
        non_boundary_mse = ((per_sample_mse * non_boundary).sum()
                            / non_boundary.sum().clamp_min(1))
        metrics = {
            'mf_loss': stopgrad(loss),
            'mf_v_mse': stopgrad(error.square().mean()),
            'non_boundary_mse': stopgrad(non_boundary_mse),
        }
        return loss, metrics
