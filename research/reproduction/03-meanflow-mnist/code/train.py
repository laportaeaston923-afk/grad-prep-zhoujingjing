"""Paper-aligned MNIST MF/iMF DiT training, with numerical diagnostics.

The loss and JVP directions are defined in experiment_core.py.
Warmup, conservative LR and EMA are TRAINING CONFIGURATION changes, not
changes to the paper's Fig.3 plain-MSE objective. No data are fabricated.
"""
import argparse
import copy
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import build_dataset
from experiment_core import training_loss
from models.dit import MFDiT

ROOT=Path(__file__).resolve().parents[1]


def args_parser():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("mf","imf","imf_cfg"),required=True)
    p.add_argument("--steps",type=int,default=20000,
                   help="Final step number; can increase when resuming.")
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--dim",type=int,default=256)
    p.add_argument("--depth",type=int,default=6)
    p.add_argument("--heads",type=int,default=8)
    p.add_argument("--patch-size",type=int,default=4)
    p.add_argument("--lr",type=float,default=2e-5,
                   help="Peak LR, deliberately below the prior unstable 1e-4.")
    p.add_argument("--warmup-steps",type=int,default=2000)
    p.add_argument("--schedule-steps",type=int,default=20000,
                   help="Fixed scheduler horizon; leave unchanged when resuming.")
    p.add_argument("--min-lr-ratio",type=float,default=0.2)
    p.add_argument("--grad-clip",type=float,default=1.0)
    p.add_argument("--ema-decay",type=float,default=0.999)
    p.add_argument("--log-every",type=int,default=100)
    p.add_argument("--save-every",type=int,default=1000)
    p.add_argument("--abort-offdiag-threshold",type=float,default=250.0,
                   help="Stop after two consecutive log windows above this "
                        "limit. Set to 0 to disable. It does NOT clip the loss.")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--workers",type=int,default=0)
    p.add_argument("--device",default="auto")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--data-root",default=str(ROOT/"data"))
    p.add_argument("--runs-root",default=str(ROOT/"runs_dit_stable"))
    a=p.parse_args()
    if min(a.steps,a.batch_size,a.log_every,a.save_every,a.schedule_steps)<1:
        p.error("steps, batch-size, log-every, save-every, schedule-steps must be > 0")
    if a.warmup_steps<0 or a.warmup_steps>=a.schedule_steps:
        p.error("warmup-steps must be >=0 and < schedule-steps")
    if a.lr<=0 or a.grad_clip<=0 or not (0<=a.ema_decay<1):
        p.error("lr and grad-clip must be positive; EMA decay in [0,1)")
    if not(0<a.min_lr_ratio<=1):p.error("min-lr-ratio must be in (0,1]")
    if a.dim%4 or a.dim%a.heads or 32%a.patch_size:
        p.error("dim must be divisible by 4 and heads; patch-size must divide 32")
    return a


def step_lr(step,lr,warmup,schedule_steps,min_ratio):
    # step is one-indexed: 1,2,... Use a fixed horizon for honest resume.
    if warmup and step<=warmup:
        return lr*step/warmup
    progress=min(1.,max(0.,(step-warmup)/max(1,schedule_steps-warmup)))
    return lr*(min_ratio+(1-min_ratio)*0.5*(1+math.cos(math.pi*progress)))


@torch.no_grad()
def ema_update(ema,model,decay):
    # Original MFDiT is unchanged; EMA is only used for sample quality.
    for target,source in zip(ema.parameters(),model.parameters()):
        target.lerp_(source,1-decay)
    for target,source in zip(ema.buffers(),model.buffers()):
        target.copy_(source)


def main():
    a=args_parser()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(min(8,torch.get_num_threads()))
    device=torch.device("cuda" if a.device=="auto" and torch.cuda.is_available()
                        else ("cpu" if a.device=="auto" else a.device))
    # Use float32 for correct, more predictable JVP calculations.
    cfg={"dataset":"mnist","data_root":a.data_root,"image_size":32}
    ds=build_dataset(cfg)  # original upstream MNIST Resize((32,32))
    dl=DataLoader(ds,batch_size=a.batch_size,shuffle=True,drop_last=True,
                  num_workers=a.workers,pin_memory=device.type=="cuda",
                  generator=torch.Generator().manual_seed(a.seed))
    model_config=dict(input_size=32,patch_size=a.patch_size,in_channels=1,
                      dim=a.dim,depth=a.depth,num_heads=a.heads,
                      num_classes=10,max_cfg=4.0)
    model=MFDiT(**model_config).to(device)
    ema=copy.deepcopy(model).to(device).eval()
    ema.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(0.9,0.95),
                          weight_decay=0.0)
    hparams={k:getattr(a,k) for k in
             ("lr","warmup_steps","schedule_steps","min_lr_ratio",
              "grad_clip","ema_decay","batch_size")}
    out=Path(a.runs_root)/a.mode
    out.mkdir(parents=True,exist_ok=True)
    ckpt=out/"last.pt"
    log=out/"train.csv"
    step=0

    if ckpt.exists():
        if not a.resume:
            raise FileExistsError(
                f"{ckpt} already exists. No model is overwritten automatically. "
                "Use --resume only for runs trained by THIS stabilized script; "
                "or specify a different --runs-root.")
        saved=torch.load(ckpt,map_location=device,weights_only=False)
        if saved.get("version")!="dit-stable-v2":
            raise ValueError("Incompatible old checkpoint; do not resume "
                             "the unstable run into this new experiment.")
        if saved["mode"]!=a.mode or saved["model_config"]!=model_config or \
           saved["hparams"]!=hparams:
            raise ValueError("The DiT dimensions or scheduler hyperparameters "
                             "do not match this checkpoint.")
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema_model"])
        opt.load_state_dict(saved["optimizer"])
        step=int(saved["step"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if torch.cuda.is_available() and "cuda_rng" in saved:
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        print("Resumed step",step,
              "(the shuffled data iterator restarts; not bit-exact continuation)")
    else:
        if a.resume:raise FileNotFoundError(ckpt)
        with open(log,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([
                "step","learning_rate","total_loss","v_loss","auxiliary_v_loss",
                "offdiag_v_loss","offdiag_std","offdiag_samples",
                "diag_v_loss","diag_std","diag_samples",
                "jvp_rms","jvp_max","weighted_jvp_rms","weighted_jvp_max",
                "u_rms","u_max","tangent_rms",
                "compound_rms","offdiag_batch_max","delta_mean",
                "grad_norm_mean","grad_norm_max","grad_clip_fraction",
                "elapsed_seconds"])

    def save():
        data=dict(version="dit-stable-v2",model=model.state_dict(),
                  ema_model=ema.state_dict(),optimizer=opt.state_dict(),
                  model_config=model_config,hparams=hparams,
                  step=step,mode=a.mode,seed=a.seed,
                  torch_rng=torch.get_rng_state(),
                  numpy_rng=np.random.get_state(),
                  python_rng=random.getstate())
        if torch.cuda.is_available():
            data["cuda_rng"]=torch.cuda.get_rng_state_all()
        tmp=ckpt.with_suffix(".pt.tmp")
        torch.save(data,tmp);tmp.replace(ckpt)

    scalar_keys=("loss","v_loss","auxiliary_v_loss",
                 "jvp_rms","weighted_jvp_rms","u_rms","tangent_rms",
                 "compound_rms","delta_mean")
    max_keys=("jvp_max","weighted_jvp_max","u_max","offdiag_batch_max")
    count_keys=("offdiag_count","diag_count")
    sum_keys=("offdiag_sum","offdiag_sumsq","diag_sum","diag_sumsq")
    acc={key:0.0 for key in scalar_keys+sum_keys+count_keys}
    maxima={key:0.0 for key in max_keys}
    nlogged=0;grad_total=0.;grad_max=0.;nclipped=0
    lr=0.
    start=time.perf_counter()
    it=iter(dl)
    model.train()
    exceeded=0
    print(f"{a.mode} starting at {step}/{a.steps}; device={device}; "
          f"peak_lr={a.lr}, warmup={a.warmup_steps}, clip={a.grad_clip}",flush=True)
    while step<a.steps:
        try:x,y=next(it)
        except StopIteration:it=iter(dl);x,y=next(it)
        x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
        lr=step_lr(step+1,a.lr,a.warmup_steps,a.schedule_steps,a.min_lr_ratio)
        for group in opt.param_groups:group["lr"]=lr
        opt.zero_grad(set_to_none=True)
        loss,stat=training_loss(model,x,y,a.mode)
        if not torch.isfinite(loss):
            save()
            raise FloatingPointError(f"Non-finite loss at step {step}; checkpoint saved")
        loss.backward()
        gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip))
        if not math.isfinite(gn):
            opt.zero_grad(set_to_none=True)
            save()
            raise FloatingPointError(f"Non-finite gradient at step {step}; stopped")
        opt.step()
        ema_update(ema,model,a.ema_decay)
        step+=1;nlogged+=1
        grad_total+=gn;grad_max=max(grad_max,gn)
        nclipped+=int(gn>a.grad_clip)
        for key in scalar_keys+sum_keys+count_keys:acc[key]+=stat[key]
        for key in max_keys:maxima[key]=max(maxima[key],stat[key])
        if step%a.log_every==0 or step==a.steps:
            n=acc["offdiag_count"];d=acc["diag_count"]
            offmean=acc["offdiag_sum"]/n if n else float("nan")
            offstd=math.sqrt(max(0.,acc["offdiag_sumsq"]/n-offmean**2)) if n else float("nan")
            diagmean=acc["diag_sum"]/d if d else float("nan")
            diagstd=math.sqrt(max(0.,acc["diag_sumsq"]/d-diagmean**2)) if d else float("nan")
            row=[step,lr,acc["loss"]/nlogged,acc["v_loss"]/nlogged,
                 acc["auxiliary_v_loss"]/nlogged,offmean,offstd,int(n),
                 diagmean,diagstd,int(d),
                 acc["jvp_rms"]/nlogged,maxima["jvp_max"],
                 acc["weighted_jvp_rms"]/nlogged,maxima["weighted_jvp_max"],
                 acc["u_rms"]/nlogged,maxima["u_max"],
                 acc["tangent_rms"]/nlogged,acc["compound_rms"]/nlogged,
                 maxima["offdiag_batch_max"],acc["delta_mean"]/nlogged,
                 grad_total/nlogged,grad_max,nclipped/nlogged,
                 round(time.perf_counter()-start,1)]
            with open(log,"a",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow(row)
            print(f"{a.mode:7s} step={step:6d}/{a.steps} "
                  f"offdiag={offmean:.4f} diag={diagmean:.4f} "
                  f"JVP_RMS={row[11]:.3f} ΔJVP_RMS={row[13]:.3f} "
                  f"grad={row[21]:.2f} lr={lr:.2e}",flush=True)
            if a.abort_offdiag_threshold>0 and math.isfinite(offmean) and \
               offmean>a.abort_offdiag_threshold:
                exceeded+=1
            else:exceeded=0
            acc={key:0.0 for key in acc}
            maxima={key:0.0 for key in maxima}
            nlogged=0;grad_total=0.;grad_max=0.;nclipped=0
            if exceeded>=2:
                save()
                raise RuntimeError(
                    f"Two consecutive {a.log_every}-step intervals exceeded "
                    f"offdiag v-loss {a.abort_offdiag_threshold}. Stopped early "
                    f"at {step} with checkpoint+diagnostic CSV; inspect JVP/MAX, "
                    "gradient norm and time embedding scale before retry.")
        if step%a.save_every==0 or step==a.steps:save()
    print("COMPLETE",a.mode,"saved",ckpt,log)


if __name__=="__main__":
    main()
