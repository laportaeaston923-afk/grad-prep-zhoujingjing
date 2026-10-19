"""Generate four requested TRUE-result figures using trained upstream DiT checkpoints."""
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from models.dit import MFDiT
from experiment_core import sample_one_step,to_images

ROOT=Path(__file__).resolve().parents[1]


def load(runs,mode,device):
    file=runs/mode/"last.pt"
    if not file.exists():
        raise FileNotFoundError(f"{file} is missing. Train {mode} first.")
    state=torch.load(file,map_location=device,weights_only=False)
    if state["mode"]!=mode:raise ValueError("Wrong checkpoint mode")
    model=MFDiT(**state["model_config"]).to(device)
    model.load_state_dict(state.get("ema_model",state["model"]),strict=True);model.eval()
    return model,state


def plot_rows(tensors,labels,title,path):
    nrows=len(tensors);ncols=tensors[0].shape[0]
    fig,axes=plt.subplots(nrows,ncols,figsize=(ncols*1.15,nrows*1.4+.5),
                          squeeze=False)
    for i,arr in enumerate(tensors):
        for j in range(ncols):
            axes[i,j].imshow(arr[j,0].detach().cpu().numpy(),
                             cmap="gray",vmin=0,vmax=1)
            axes[i,j].axis("off")
            if i==0:axes[i,j].set_title(str(j),fontsize=10)
        axes[i,0].text(-.17,.5,labels[i],rotation=90,ha="right",va="center",
                       transform=axes[i,0].transAxes,fontsize=10)
    fig.suptitle(title,fontsize=12)
    fig.subplots_adjust(left=.1,right=.99,top=.88,bottom=.035,
                        wspace=.035,hspace=.16)
    fig.savefig(path,dpi=180);plt.close(fig)


def loss_plot(runs,dest):
    fig,ax=plt.subplots(figsize=(8,4.8))
    for mode,label,color in (("mf","Original MF","tab:blue"),
                             ("imf","Improved MF","tab:orange")):
        with open(runs/mode/"train.csv",newline="",encoding="utf-8") as f:
            rows=list(csv.DictReader(f))
        if not rows:raise RuntimeError(f"No logs for {mode}")
        steps=np.asarray([float(q["step"]) for q in rows])
        val=np.asarray([float(q["offdiag_v_loss"]) for q in rows])
        ax.plot(steps,val,alpha=.32,lw=.8,color=color)
        kernel=np.ones(min(9,len(val)))/min(9,len(val))
        if len(val)>=9:
            smooth=np.convolve(val,kernel,mode="valid")
            ax.plot(steps[4:-4],smooth,label=label,color=color,lw=2)
        else:
            ax.plot(steps,val,label=label,color=color,lw=2)
    ax.set(xlabel="Training steps",ylabel="Off-diagonal plain v-loss (MSE)",
           title="Fig. 3-style MF vs iMF: matched MNIST DiT, no CFG")
    ax.grid(alpha=.15);ax.legend();fig.tight_layout()
    fig.savefig(dest,dpi=180);plt.close(fig)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--runs-root",default=str(ROOT/"runs_dit_stable"))
    p.add_argument("--results",default=str(ROOT/"results"/"dit_stable"))
    p.add_argument("--fig3-only",action="store_true",
                   help="No CFG checkpoint needed; generate Fig3, MF-vs-iMF and 1-step")
    p.add_argument("--device",default="auto")
    p.add_argument("--seed",type=int,default=2026)
    a=p.parse_args()
    dev=torch.device("cuda" if a.device=="auto" and torch.cuda.is_available()
                     else ("cpu" if a.device=="auto" else a.device))
    runs,out=Path(a.runs_root),Path(a.results)
    modes=("mf","imf") if a.fig3_only else ("mf","imf","imf_cfg")
    checkpoints={k:load(runs,k,dev) for k in modes}
    out.mkdir(parents=True,exist_ok=True)
    loss_plot(runs,out/"fig3_loss_dit.png")
    # Identical noise, labels and number of evaluations for every comparison.
    torch.manual_seed(a.seed)
    label=torch.arange(10,device=dev)
    noise=torch.randn(10,1,32,32,device=dev)
    mf=checkpoints["mf"][0];imf=checkpoints["imf"][0]
    z_imf,u_imf=sample_one_step(imf,noise,label,1.)
    velocity_vis=(u_imf/(2*max(1.,u_imf.abs().max().item()))+.5).clamp(0,1)
    plot_rows([to_images(noise),velocity_vis,to_images(z_imf)],
              ["Gaussian noise z1","avg velocity u*","z0=z1-u"],
              "iMF: one DiT forward (1-NFE); *velocity display is normalized",
              out/"one_step_process_dit.png")
    z_mf,_=sample_one_step(mf,noise,label,1.)
    plot_rows([to_images(z_mf),to_images(z_imf)],
              [f"MF ({checkpoints['mf'][1]['step']:,})",f"iMF ({checkpoints['imf'][1]['step']:,})"],
              "Matched DiT, noise and class; 1-NFE (training steps in row labels)",
              out/"mf_vs_imf_dit.png")
    if not a.fig3_only:
        cfg=checkpoints["imf_cfg"][0]
        scales=[1.,1.5,2.,3.,4.,6.,8.]
        z_cfg=[to_images(sample_one_step(cfg,noise,label,scale)[0])
               for scale in scales]
        plot_rows(z_cfg,[f"CFG = {scale:g}" for scale in scales],
                  "Same iMF DiT, same noise; flexible CFG in one evaluation",
                  out/"cfg_scales_dit.png")
    print(f"Saved {'three figures (CFG skipped)' if a.fig3_only else 'four figures'} "
          f"from actual DiT checkpoints in {out}")


if __name__=="__main__":
    main()
