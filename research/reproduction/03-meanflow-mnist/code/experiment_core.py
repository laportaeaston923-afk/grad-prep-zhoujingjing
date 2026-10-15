"""Paper-aligned MeanFlow/iMF objectives on the *upstream MFDiT* backbone.

Fig.3 modes: mf, imf
  - no CFG, no adaptive weighting, no auxiliary v-head loss
  - same DiT architecture, data, seed, schedule
  - MF: JVP(u; tangent=e-x) [paper Eq.9]
  - iMF: JVP(u; tangent=u(z,t,t)) [paper Eq.12, boundary condition]
  - both use plain v-loss; log ONLY samples with t != r for Fig.3.
CFG mode: imf_cfg
  - iMF JVP tangent = v_c predicted by the upstream auxiliary v-head
  - v target follows paper Algorithm 2
  - auxiliary flow-matching v-head regression, as in the source repository
  - omega is an explicit DiT conditioning input.
Source: https://github.com/haidog-yaqub/MeanFlow (MIT).
"""
import torch
import torch.nn.functional as F
from functools import partial


def sample_times(batch_size, device, diagonal_probability=0.5):
    """Independent logit-normal(-0.4,1) endpoints ordered r<=t."""
    pair=torch.sigmoid(torch.randn(batch_size,2,device=device)-0.4)
    t=pair.max(dim=1).values
    r=pair.min(dim=1).values
    diagonal=(torch.rand(batch_size,device=device)<diagonal_probability)
    r=torch.where(diagonal,t,r)
    return t,r,~diagonal


def sample_cfg(batch_size, device, max_cfg=4.0, cls_drop=0.1):
    # Draw a continuous, explicit omega scale (paper Sec.4.2).
    w=torch.empty(batch_size,device=device).uniform_(1.,max_cfg)
    dropped=torch.rand(batch_size,device=device)<cls_drop
    return torch.where(dropped,torch.ones_like(w),w),dropped


def training_loss(model,x,labels,mode,diagonal_probability=0.5,max_cfg=4.0,cls_drop=0.1):
    """Returns differentiable loss + detached statistics.

    All image inputs are [0,1]. Time convention z_t=(1-t)*x+t*noise.
    The underlying model signature is the original MFDiT(z, t, r, ...).
    """
    if mode not in ("mf","imf","imf_cfg"):
        raise ValueError(f"Unknown training mode: {mode}")
    n=x.shape[0]
    device=x.device
    x=x*2-1
    noise=torch.randn_like(x)
    t,r,offdiag=sample_times(n,device,diagonal_probability)
    delta=(t-r).view(n,1,1,1)
    z=(1-t[:,None,None,None])*x+t[:,None,None,None]*noise
    vc=noise-x
    c=labels
    w=torch.ones(n,device=device)
    if mode=="imf_cfg":
        w,drop=sample_cfg(n,device,max_cfg,cls_drop)
        c=torch.where(drop,torch.full_like(c,10),c)

    # Compute directional derivative of the *u-head* with no parameter graph.
    # torch.autograd.functional.jvp needs the direction but no parameter grads.
    with torch.no_grad():
        if mode=="mf":
            tangent=vc                             # Eq.(9): original MF
            target=vc
        elif mode=="imf":
            # Eq.(12) boundary: v(z,t)=u(z,t,t), no extra v-head.
            tangent=model(z,t,t,y=c,w=w,return_v=False)
            target=vc
        else:
            # Upstream dual-head DiT: auxiliary v-head for CFG.
            # Boundary is evaluated at r=t to avoid r leaking into v(t).
            _,v_c=model(z,t,t,y=c,w=w,return_v=True)
            _,v_u=model(z,t,t,y=torch.full_like(c,10),
                        w=torch.ones_like(w),return_v=True)
            tangent=v_c
            # Paper Algorithm 2: v_tgt = (e-x)+(1-1/omega)*(v_c-v_u)
            target=vc+(1-1/w)[:,None,None,None]*(v_c-v_u)

        # Input ordering in upstream MFDiT is (z,t,r), hence time tangents (1,0).
        u_fn=partial(model,y=c,w=w,return_v=False,use_flash_attention=False)
        _,dudt=torch.autograd.functional.jvp(
            u_fn,(z,t,r),(tangent.detach(),torch.ones_like(t),torch.zeros_like(r)),
            create_graph=False,strict=False
        )
    u,v_aux=model(z,t,r,y=c,w=w,return_v=True)
    V=u+delta*dudt.detach()
    per_example=(V-target.detach()).square().flatten(1).mean(1)
    meanflow_loss=per_example.mean()
    aux_loss=torch.zeros((),device=device)
    if mode=="imf_cfg":
        # Auxiliary instantaneous velocity head (NOT part of Fig.3).
        aux_loss=F.mse_loss(v_aux,target.detach())
    full=meanflow_loss+aux_loss
    with torch.no_grad():
        # Recording both groups distinguishes u-regression from JVP instability.
        # Diagnostics never change the original paper v-loss objective.
        diag=~offdiag
        off_values=per_example[offdiag].float()
        diag_values=per_example[diag].float()
        off_count=int(off_values.numel())
        diag_count=int(diag_values.numel())
        off_sum=float(off_values.sum())
        off_sumsq=float(off_values.square().sum())
        diag_sum=float(diag_values.sum())
        diag_sumsq=float(diag_values.square().sum())
        # JVP can have large tails even while t=r regression is fine.
        stats=dict(
            loss=float(full),v_loss=float(meanflow_loss),
            auxiliary_v_loss=float(aux_loss),
            offdiag_count=off_count,offdiag_sum=off_sum,offdiag_sumsq=off_sumsq,
            diag_count=diag_count,diag_sum=diag_sum,diag_sumsq=diag_sumsq,
            jvp_rms=float(dudt.float().square().mean().sqrt()),
            jvp_max=float(dudt.float().abs().amax()),
            weighted_jvp_rms=float((delta*dudt).float().square().mean().sqrt()),
            weighted_jvp_max=float((delta*dudt).float().abs().amax()),
            u_rms=float(u.float().square().mean().sqrt()),
            u_max=float(u.float().abs().amax()),
            tangent_rms=float(tangent.float().square().mean().sqrt()),
            compound_rms=float(V.float().square().mean().sqrt()),
            offdiag_batch_max=float(off_values.max()) if off_count else 0.0,
            delta_mean=float((t-r).mean()))
    return full,stats


@torch.no_grad()
def sample_one_step(model,noise,labels,omega=1.0):
    """One DiT forward evaluation: z0=z1-u(z1,t=1,r=0,omega)."""
    b=len(noise)
    t=torch.ones(b,device=noise.device)
    r=torch.zeros_like(t)
    w=torch.full_like(t,float(omega))
    u=model(noise,t,r,y=labels,w=w,return_v=False)
    z0=noise-u
    return z0,u


def to_images(z):
    return z.clamp(-1,1).add(1).mul(0.5)
