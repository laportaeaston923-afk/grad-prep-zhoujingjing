"""Train matched MF, iMF, and iMF+flexible CFG on actual MNIST."""
import argparse,csv,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from model import TinyVelocityUNet,loss_batch

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["mf","imf","imf_cfg"],required=True)
    p.add_argument("--steps",type=int,default=20000)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--width",type=int,default=32)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default="auto")
    p.add_argument("--workers",type=int,default=0)
    p.add_argument("--log-every",type=int,default=50)
    p.add_argument("--save-every",type=int,default=2000)
    p.add_argument("--data-root",default=str(ROOT/"data"))
    p.add_argument("--out-root",default=str(ROOT/"runs"))
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if args.steps<1:p.error("--steps must be positive")
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(min(8,torch.get_num_threads()))
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available()
         else ("cpu" if args.device=="auto" else args.device))
    ds=datasets.MNIST(args.data_root,train=True,transform=transforms.ToTensor(),download=True)
    rng=torch.Generator().manual_seed(args.seed)
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,
                  drop_last=True,generator=rng,pin_memory=device.type=="cuda")
    model=TinyVelocityUNet(args.width).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0)
    out=Path(args.out_root)/args.mode;out.mkdir(parents=True,exist_ok=True)
    ckpt=out/"last.pt";step=0
    if args.resume and ckpt.is_file():
        state=torch.load(ckpt,map_location=device,weights_only=False)
        if state["mode"]!=args.mode or state["width"]!=args.width:
            raise ValueError("Checkpoint mode or width differs")
        model.load_state_dict(state["model"]);opt.load_state_dict(state["optimizer"])
        step=state["step"];print("Resuming step",step,"(data order restarts)")
    elif args.resume:raise FileNotFoundError(ckpt)
    if not args.resume:
        with open(out/"loss.csv","w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(["step","loss","fig3_offdiag_loss","seconds_elapsed"])
    start=time.perf_counter();s1=s2=count=0.;model.train();it=iter(dl)
    while step<args.steps:
        try:x,y=next(it)
        except StopIteration:it=iter(dl);x,y=next(it)
        x,y=x.to(device),y.to(device)
        opt.zero_grad(set_to_none=True)
        loss,diag=loss_batch(model,x,y,args.mode)
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite loss")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        opt.step();step+=1
        s1+=diag["loss"];s2+=diag["fig3_offdiag_loss"];count+=1
        if step%args.log_every==0 or step==args.steps:
            row=[step,s1/count,s2/count,round(time.perf_counter()-start,1)]
            with open(out/"loss.csv","a",newline="",encoding="utf-8") as f:csv.writer(f).writerow(row)
            print(f'{args.mode:8s} step={step:6d} loss={row[1]:.4f} offdiag={row[2]:.4f}',flush=True)
            s1=s2=count=0.
        if step%args.save_every==0 or step==args.steps:
            torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),
                        "step":step,"mode":args.mode,"width":args.width,
                        "seed":args.seed,"dataset":"MNIST","image_size":28},ckpt)
    print("Saved",ckpt)

if __name__=="__main__":main()
