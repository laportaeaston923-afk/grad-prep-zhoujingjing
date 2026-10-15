"""Evaluate MNIST generation with an independent digit classifier.

The reported Frechet distance uses MNIST classifier features; it is NOT
standard ImageNet Inception FID and is not comparable with the paper's FID.
"""
import argparse,csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from scipy.linalg import sqrtm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import TinyVelocityUNet,sample_one_step

ROOT=Path(__file__).resolve().parents[1]

class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Flatten(),nn.Linear(64*7*7,128),nn.ReLU())
        self.head=nn.Linear(128,10)
    def forward(self,x):
        features=self.encoder(x)
        return self.head(features),features

def train_classifier(net,root,device,epochs,batch_size,path):
    ds=datasets.MNIST(root,train=True,transform=transforms.ToTensor(),download=True)
    dl=DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=0)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    net.train()
    for ep in range(epochs):
        hit=seen=0
        for x,y in dl:
            x,y=x.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            logits,_=net(x)
            loss=F.cross_entropy(logits,y);loss.backward();opt.step()
            hit+=(logits.argmax(dim=1)==y).sum().item();seen+=len(x)
        print(f'Classifier epoch {ep+1}/{epochs}, train accuracy={hit/seen:.3%}',flush=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(net.state_dict(),path)

def moments(parts):
    x=np.concatenate(parts,axis=0).astype(np.float64)
    return x.mean(axis=0),np.cov(x,rowvar=False)

def frechet(m1,c1,m2,c2):
    root=sqrtm(c1@c2)
    if np.iscomplexobj(root):
        if np.max(np.abs(root.imag))>1e-4:raise ValueError("Complex covariance root")
        root=root.real
    return float(max(0.,np.dot(m1-m2,m1-m2)+np.trace(c1+c2-2*root)))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-classifier",action="store_true")
    p.add_argument("--classifier-epochs",type=int,default=4)
    p.add_argument("--num-samples",type=int,default=2000)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--data-root",default=str(ROOT/"data"))
    p.add_argument("--runs",default=str(ROOT/"runs"))
    p.add_argument("--results",default=str(ROOT/"results"))
    p.add_argument("--device",default="auto")
    p.add_argument("--seed",type=int,default=2026)
    a=p.parse_args()
    if a.num_samples<130:p.error("Choose at least 130 generated samples")
    device=torch.device("cuda" if a.device=="auto" and torch.cuda.is_available()
                        else ("cpu" if a.device=="auto" else a.device)
    torch.manual_seed(a.seed)
    checkpoint=Path(a.runs)/"mnist_classifier.pt"
    net=DigitClassifier().to(device)
    if a.train_classifier:
        train_classifier(net,a.data_root,device,a.classifier_epochs,a.batch_size,checkpoint)
    if not checkpoint.exists():raise FileNotFoundError(f'Run with --train-classifier: {checkpoint}')
    net.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
    net.eval()
    test=datasets.MNIST(a.data_root,train=False,transform=transforms.ToTensor(),download=True)
    dl=DataLoader(test,batch_size=a.batch_size,shuffle=False)
    real=[];hit=seen=0
    with torch.inference_mode():
        for x,y in dl:
            if seen>=a.num_samples:break
            x=x[:a.num_samples-seen].to(device);y=y[:len(x)].to(device)
            logit,feat=net(x)
            hit+=(logit.argmax(dim=1)==y).sum().item();seen+=len(x)
            real.append(feat.cpu().numpy())
    classifier_accuracy=hit/seen
    print(f'Held-out classifier accuracy={classifier_accuracy:.2%}')
    real_mean,real_cov=moments(real)
    rng=torch.Generator(device="cpu").manual_seed(a.seed)
    labels=torch.arange(a.num_samples)%10
    noise=torch.randn(a.num_samples,1,28,28,generator=rng)
    rows=[]
    for mode,scale in [("mf",1.),("imf",1.),("imf_cfg",1.),("imf_cfg",2.),
                       ("imf_cfg",3.),("imf_cfg",4.)]:
        path=Path(a.runs)/mode/"last.pt"
        if not path.exists():raise FileNotFoundError(path)
        state=torch.load(path,map_location=device,weights_only=False)
        model=TinyVelocityUNet(width=state["width"]).to(device)
        model.load_state_dict(state["model"]);model.eval()
        features=[];preds=[]
        with torch.inference_mode():
            for k in range(0,a.num_samples,a.batch_size):
                c=labels[k:k+a.batch_size].to(device)
                z=noise[k:k+a.batch_size].to(device)
                img,_=sample_one_step(model,c,z,scale)
                img=(img.clamp(-1,1)+1)/2
                logits,feat=net(img)
                features.append(feat.cpu().numpy())
                preds.append((logits.argmax(1)==c).float().cpu())
        mu,cov=moments(features)
        item={"model":mode,"cfg_scale":scale,"steps":state["step"],
              "mnist_feature_frechet":frechet(real_mean,real_cov,mu,cov),
              "label_consistency":float(torch.cat(preds).mean()),
              "classifier_test_accuracy":classifier_accuracy,"num_samples":a.num_samples}
        print(item,flush=True);rows.append(item)
    out=Path(a.results);out.mkdir(parents=True,exist_ok=True)
    with open(out/"metrics.csv","w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    fig,ax=plt.subplots(figsize=(8,4))
    names=[f'{r["model"]}\nw={r["cfg_scale"]:g}' for r in rows]
    ax.bar(names,[r["mnist_feature_frechet"] for r in rows])
    ax.set_ylabel("MNIST feature Frechet (lower is better)")
    ax.set_title("MNIST feature-distance comparison (NOT standard Inception FID)")
    ax.grid(axis="y",alpha=.2);fig.tight_layout()
    fig.savefig(out/"mf_vs_imf_metrics.png",dpi=180);plt.close(fig)

if __name__=="__main__":main()
