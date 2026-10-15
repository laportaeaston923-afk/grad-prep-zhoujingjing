"""Run matched DiT experiments. MNIST, checkpoint/resume, real CSV stats.

Uses ORIGINAL upstream models/dit.py + data.py and experiment_core.py.
"""
import argparse,csv,math,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import build_dataset
from models.dit import MFDiT
from experiment_core import training_loss

ROOT=Path(__file__).resolve().parents[1]


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("mf","imf","imf_cfg"),required=True)
    p.add_argument("--steps",type=int,default=20000)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--dim",type=int,default=256,
                   help="DiT width (upstream MNIST uses 512)")
    p.add_argument("--depth",type=int,default=6,
                   help="DiT depth (upstream MNIST uses 12)")
    p.add_argument("--heads",type=int,default=8)
    p.add_argument("--patch-size",type=int,default=4,
                   help="patch=4 is faster; upstream MNIST uses patch=2")
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--log-every",type=int,default=100)
    p.add_argument("--save-every",type=int,default=2000)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--workers",type=int,default=0)
    p.add_argument("--device",default="auto")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--data-root",default=str(ROOT/"data"))
    p.add_argument("--runs-root",default=str(ROOT/"runs_dit"))
    args=p.parse_args()
    if args.steps<1 or args.batch_size<1 or args.log_every<1 or args.save_every<1:
        p.error("steps, batch-size, log-every, save-every must be positive")
    if args.dim%4 or args.dim%args.heads or 32%args.patch_size:
        p.error("dim must be divisible by 4 and heads; patch-size must divide 32")
    return args


def main():
    a=parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    device=torch.device("cuda" if a.device=="auto" and torch.cuda.is_available()
                        else ("cpu" if a.device=="auto" else a.device))
    # Keep JVP in fp32 to avoid unsupported half-precision forward-AD kernels.
    torch.set_num_threads(min(8,torch.get_num_threads()))
    if device.type=="cuda":
        torch.backends.cuda.matmul.allow_tf32=True
    cfg={"dataset":"mnist","data_root":a.data_root,"image_size":32}
    ds=build_dataset(cfg)  # upstream resize 28->32, ToTensor
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=True,drop_last=True,
                      num_workers=a.workers,pin_memory=device.type=="cuda",
                      generator=torch.Generator().manual_seed(a.seed))
    model_cfg=dict(input_size=32,patch_size=a.patch_size,in_channels=1,
                   dim=a.dim,depth=a.depth,num_heads=a.heads,
                   num_classes=10,max_cfg=4.0)
    model=MFDiT(**model_cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.0)
    out=Path(a.runs_root)/a.mode
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/"last.pt"
    log_path=out/"train.csv"
    step=0
    if checkpoint.exists():
        if not a.resume:
            raise FileExistsError(
                f"{checkpoint} exists. Use --resume or choose new --runs-root. "
                "Existing models are never silently overwritten.")
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        if saved["mode"]!=a.mode or saved["model_config"]!=model_cfg or \
           saved["batch_size"]!=a.batch_size:
            raise ValueError("Resume config differs from original checkpoint")
        model.load_state_dict(saved["model"]);opt.load_state_dict(saved["optimizer"])
        step=int(saved["step"])
        if "torch_rng" in saved:torch.set_rng_state(saved["torch_rng"].cpu())
        if "numpy_rng" in saved:np.random.set_state(saved["numpy_rng"])
        print(f"Resumed {a.mode} at step {step}; data sampler order restarts.")
    else:
        with open(log_path,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(["step","total_loss","v_loss","auxiliary_v_loss",
                                    "offdiag_v_loss","offdiag_std","offdiag_samples",
                                    "elapsed_seconds"])

    it=iter(loader);start=time.perf_counter()
    totals={"loss":0.0,"v_loss":0.0,"auxiliary_v_loss":0.0,
            "offdiag_sum":0.0,"offdiag_sumsq":0.0,"offdiag_count":0}
    nlogged=0
    model.train()
    while step<a.steps:
        try:x,y=next(it)
        except StopIteration:it=iter(loader);x,y=next(it)
        x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss,st=training_loss(model,x,y,a.mode)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss in {a.mode} at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
        opt.step();step+=1
        nlogged+=1
        for key in totals:totals[key]+=st[key]
        if step%a.log_every==0 or step==a.steps:
            n=totals["offdiag_count"]
            m=totals["offdiag_sum"]/n if n else float("nan")
            std=math.sqrt(max(0.,totals["offdiag_sumsq"]/n-m*m)) if n else float("nan")
            row=[step,totals["loss"]/nlogged,totals["v_loss"]/nlogged,
                 totals["auxiliary_v_loss"]/nlogged,m,std,n,
                 round(time.perf_counter()-start,1)]
            with open(log_path,"a",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow(row)
            print(f"{a.mode:7s} {step:>6d}/{a.steps} "
                  f"v_loss={row[2]:.4f} offdiag={m:.4f} ±{std:.4f}",flush=True)
            totals={k:(0 if k=="offdiag_count" else 0.0) for k in totals}
            nlogged=0
        if step%a.save_every==0 or step==a.steps:
            state=dict(model=model.state_dict(),optimizer=opt.state_dict(),
                       step=step,mode=a.mode,model_config=model_cfg,
                       batch_size=a.batch_size,seed=a.seed,
                       torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state())
            tmp=checkpoint.with_suffix(".pt.tmp")
            torch.save(state,tmp);tmp.replace(checkpoint)
    print("Model and logs saved:",checkpoint,log_path)


if __name__=="__main__":
    main()
