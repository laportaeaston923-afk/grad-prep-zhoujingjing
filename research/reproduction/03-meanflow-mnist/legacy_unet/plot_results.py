"""Draw four genuine result figures from actual checkpoints and CSV logs."""
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from model import TinyVelocityUNet,sample_one_step
ROOT=Path(__file__).resolve().parents[1]

def load_model(runs,mode,device):
    path=runs/mode/"last.pt"
    if not path.is_file():raise FileNotFoundError(f"Train {mode} first: {path}")
    s=torch.load(path,map_location=device,weights_only=False)
    model=TinyVelocityUNet(s["width"]).to(device)
    model.load_state_dict(s["model"]);model.eval()
    return model,s["step"]

def to_img(x):return x.clamp(-1,1).add(1).mul(.5)

def plot_grid(rows,names,path,title):
    nr=len(rows);nc=len(rows[0])
    fig,axs=plt.subplots(nr,nc,figsize=(nc*1.15,nr*1.35+.5),squeeze=False)
    for i,row in enumerate(rows):
        for j in range(nc):
            axs[i,j].imshow(row[j,0].detach().cpu().numpy(),cmap="gray",vmin=0,vmax=1)
            axs[i,j].axis("off")
            if i==0:axs[i,j].set_title(str(j) if nc==10 else "",fontsize=9)
        axs[i,0].text(-.22,.5,names[i],rotation=90,transform=axs[i,0].transAxes,
                      ha="right",va="center",fontsize=9)
    fig.suptitle(title,fontsize=12)
    fig.subplots_adjust(left=.10,right=.99,top=.87,bottom=.03,hspace=.12,wspace=.04)
    fig.savefig(path,dpi=170);plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--runs",default=str(ROOT/"runs"))
    p.add_argument("--results",default=str(ROOT/"results"))
    p.add_argument("--device",default="auto")
    p.add_argument("--seed",type=int,default=2026)
    args=p.parse_args()
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available()
                        else ("cpu" if args.device=="auto" else args.device))
    runs,out=Path(args.runs),Path(args.results)
    out.mkdir(parents=True,exist_ok=True)
    models={m:load_model(runs,m,device) for m in ("mf","imf","imf_cfg")}
    fig,ax=plt.subplots(figsize=(7,4.1))
    for mode,label in (("mf","Original MF on MNIST"),("imf","iMF on MNIST")):
        path=runs/mode/"loss.csv"
        if not path.exists():raise FileNotFoundError(path)
        with open(path,newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
        ax.plot([int(row["step"]) for row in rows],
                [float(row["fig3_offdiag_loss"]) for row in rows],
                linewidth=1.4,label=label)
    ax.set(xlabel="Training steps",ylabel="Off-diagonal v-loss (plain MSE)",
           title="Fig. 3-style loss comparison: MNIST")
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(out/"fig3_loss_mnist.png",dpi=180);plt.close(fig)
    torch.manual_seed(args.seed)
    labels=torch.arange(10,device=device)
    noise=torch.randn(10,1,28,28,device=device)
    z,u=sample_one_step(models["imf"][0],labels,noise,1.)
    u_display=(.5+u/(2*max(1.,float(u.abs().max().item())))).clamp(0,1)
    plot_grid([to_img(noise),u_display,to_img(z)],["z1: noise","u: velocity","z0: image"],
              out/"one_step_process.png",
              f'iMF MNIST 1-NFE (step {models["imf"][1]}); intermediate is velocity visualization')
    compare=[to_img(sample_one_step(models[m][0],labels,noise,1.)[0]) for m in ("mf","imf")]
    plot_grid(compare,["MF","iMF"],out/"mf_vs_imf.png",
              "Matched noise and class labels: 1-step MF vs iMF")
    scales=(1.,1.5,2.,3.,4.)
    grid=[to_img(sample_one_step(models["imf_cfg"][0],labels,noise,w)[0]) for w in scales]
    plot_grid(grid,[f"w={w:g}" for w in scales],out/"cfg_scales.png",
              "Same iMF model and noise: flexible CFG (1 NFE)")
    print("Saved 4 genuine experiment plots in",out)

if __name__=="__main__":main()
