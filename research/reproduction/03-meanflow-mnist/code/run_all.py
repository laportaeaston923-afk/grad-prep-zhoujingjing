"""Run stable Fig3 reproduction first, then optional CFG.

The previous unstable checkpoints in runs_dit/ are NEVER loaded or overwritten.
New runs in runs_dit_stable/. All files are actual training output.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def launch(script,*args):
    cmd=[sys.executable,str(HERE/script),*map(str,args)]
    print("\n>>> "+" ".join(cmd),flush=True)
    subprocess.run(cmd,check=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--steps",type=int,default=20000)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--dim",type=int,default=256)
    p.add_argument("--depth",type=int,default=6)
    p.add_argument("--heads",type=int,default=8)
    p.add_argument("--patch-size",type=int,default=4)
    p.add_argument("--lr",type=float,default=2e-5)
    p.add_argument("--warmup-steps",type=int,default=2000)
    p.add_argument("--schedule-steps",type=int,default=20000)
    p.add_argument("--min-lr-ratio",type=float,default=0.2)
    p.add_argument("--grad-clip",type=float,default=1.)
    p.add_argument("--ema-decay",type=float,default=0.999)
    p.add_argument("--abort-offdiag-threshold",type=float,default=250.)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--runs-root",default=str(ROOT/"runs_dit_stable"))
    p.add_argument("--results",default=str(ROOT/"results"/"dit_stable"))
    p.add_argument("--device",default="auto")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--fig3-only",action="store_true",
                   help="Train only MF+iMF; plot Fig3, one-step and two-model comparison")
    a=p.parse_args()
    if a.steps<=a.warmup_steps and a.steps>100:
        print("NOTE: only warmup steps will run in this short experiment.")
    modes=("mf","imf") if a.fig3_only else ("mf","imf","imf_cfg")
    for mode in modes:
        cmd=["--mode",mode,"--steps",a.steps,"--batch-size",a.batch_size,
             "--dim",a.dim,"--depth",a.depth,"--heads",a.heads,
             "--patch-size",a.patch_size,"--device",a.device,
             "--lr",a.lr,"--warmup-steps",a.warmup_steps,
             "--schedule-steps",a.schedule_steps,"--min-lr-ratio",a.min_lr_ratio,
             "--grad-clip",a.grad_clip,"--ema-decay",a.ema_decay,
             "--seed",a.seed,"--runs-root",a.runs_root,
             "--abort-offdiag-threshold",a.abort_offdiag_threshold]
        # An existing run resumes; a missing mode (often CFG after a Fig3-only
        # pilot) starts fresh. Nothing ever silently overwrites a checkpoint.
        checkpoint=Path(a.runs_root)/mode/"last.pt"
        if a.resume and checkpoint.exists():
            cmd.append("--resume")
        launch("train.py",*cmd)
    cmd=["--device",a.device,"--runs-root",a.runs_root,"--results",a.results]
    if a.fig3_only:cmd.append("--fig3-only")
    launch("plot_results.py",*cmd)
    print("Actual results:",a.results)
    print("Optional quality metrics: python code/evaluate.py --train-classifier "
          "--runs-root",a.runs_root,"--results",a.results)


if __name__=="__main__":
    main()
