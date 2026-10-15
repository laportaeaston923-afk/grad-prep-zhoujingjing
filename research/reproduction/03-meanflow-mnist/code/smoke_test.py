"""Preflight: exercise the REAL upstream MFDiT, both JVP directions and CFG.

No MNIST download. Runs a small CPU model and tests loss, gradient, EMA, and
learning-rate schedule. Passing this does NOT prove Fig3 convergence.
"""
import math
import torch
from models.dit import MFDiT
from experiment_core import training_loss,sample_one_step
from train import step_lr,ema_update
import copy


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model=MFDiT(input_size=32,patch_size=8,in_channels=1,dim=64,
                depth=1,num_heads=4,num_classes=10,max_cfg=4.0)
    x=torch.rand(4,1,32,32)
    labels=torch.tensor([1,2,3,4])
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5,betas=(.9,.95))
    ema=copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    assert step_lr(1,2e-5,2000,20000,.2)>0
    assert abs(step_lr(2000,2e-5,2000,20000,.2)-2e-5)<1e-12
    for mode in ("mf","imf","imf_cfg"):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss,stats=training_loss(model,x,labels,mode,diagonal_probability=.25)
        assert torch.isfinite(loss),f"{mode}: non-finite loss"
        for name in ("jvp_rms","jvp_max","weighted_jvp_rms",
                     "weighted_jvp_max","u_rms","offdiag_sum","diag_sum"):
            assert math.isfinite(stats[name]),f"{mode}: bad {name}"
        assert stats["offdiag_count"]+stats["diag_count"]==len(x)
        loss.backward()
        grad=model.final_layer_u.linear.weight.grad
        assert grad is not None and torch.isfinite(grad).all(),mode
        total_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        assert torch.isfinite(total_norm)
        opt.step()
        ema_update(ema,model,.999)
        out,u=sample_one_step(ema,torch.randn_like(x),labels,omega=2.)
        assert out.shape==x.shape and torch.isfinite(out).all()
        print(f"{mode:7s} loss={float(loss.detach()):.4f} "
              f"offdiag={stats['offdiag_count']} "
              f"JVP_RMS={stats['jvp_rms']:.3f} 1-NFE={tuple(out.shape)}")
    print("SMOKE TEST PASSED — real original DiT, not a convergence claim.")


if __name__=="__main__":
    main()
