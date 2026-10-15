"""Independent MNIST feature-distance and class-consistency evaluation.

This is NOT standard Inception FID, and is NOT comparable to ImageNet FID.
Classifier is fitted on MNIST train split; real moments and classifier test
accuracy measured on held-out MNIST test images. Equal noise for all generators.
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
from models.dit import MFDiT
from experiment_core import sample_one_step,to_images

ROOT=Path(__file__).resolve().parents[1]

class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Flatten(),nn.Linear(64*8*8,128),nn.ReLU())
        self.head=nn.Linear(128,10)
    def forward(self,x):
        feat=self.encoder(x)
        return self.head(feat),feat

def train_classifier(net,root,device,epochs,batch,path):
    ds=datasets.MNIST(root,train=True,download=True,
                      transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()]))
    dl=DataLoader(ds,batch_size=batch,shuffle=True,num_workers=0)
    optim=torch.optim.Adam(net.parameters(),lr=.001)
    net.train()
    for ep in range(epochs):
        total=correct=0
        for x,y in dl:
            x,y=x.to(device),y.to(device)
            optim.zero_grad(set_to_none=True)
            logits,_=net(x)
            loss=F.cross_entropy(logits,y)
            loss.backward();optim.step()
            correct+=(logits.argmax(1)==y).sum().item();total+=len(x)
        print(f"Classifier ep {ep+1}/{epochs} train acc={correct/total:.2%}",flush=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(net.state_dict(),path)

def moments(chunks):
    x=np.concatenate(chunks).astype(np.float64)
    return x.mean(axis=0),np.cov(x,rowvar=False)

def frechet(mu1,cov1,mu2,cov2):
    covmean=sqrtm(cov1@cov2)
    if np.iscomplexobj(covmean):
        if np.max(np.abs(covmean.imag))>1e-4:
            raise ValueError("Unstable matrix square root")
        covmean=covmean.real
    return float(max(0.,sum((mu1-mu2)**2)+np.trace(cov1+cov2-2*covmean)))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-classifier",action="store_true")
    p.add_argument("--classifier-epochs",type=int,default=4)
    p.add_argument("--samples",type=int,default=2000)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--device",default="auto")
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--runs-root",default=str(ROOT/"runs_dit_stable"))
    p.add_argument("--results",default=str(ROOT/"results"/"dit_stable"))
    p.add_argument("--data-root",default=str(ROOT/"data"))
    a=p.parse_args()
    if a.samples<130 or a.samples>10000:p.error("--samples must be between 130 and 10000")
    device=torch.device("cuda" if a.device=="auto" and torch.cuda.is_available()
                        else ("cpu" if a.device=="auto" else a.device))
    torch.manual_seed(a.seed)
    runs=Path(a.runs_root);out=Path(a.results)
    classifier_path=runs/"mnist_32_classifier.pt"
    net=DigitClassifier().to(device)
    if a.train_classifier:
        train_classifier(net,a.data_root,device,a.classifier_epochs,a.batch_size,classifier_path)
    if not classifier_path.exists():
        raise FileNotFoundError(f"Run with --train-classifier first: {classifier_path}")
    net.load_state_dict(torch.load(classifier_path,map_location=device,weights_only=True))
    net.eval()
    test=datasets.MNIST(a.data_root,train=False,download=True,
                        transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()]))
    dl=DataLoader(test,batch_size=a.batch_size,shuffle=False,num_workers=0)
    real=[];accuracy_n=accuracy_d=0
    with torch.inference_mode():
        for x,y in dl:
            if accuracy_d>=a.samples:break
            x=x[:a.samples-accuracy_d].to(device)
            y=y[:len(x)].to(device)
            logits,feat=net(x)
            accuracy_n+=(logits.argmax(1)==y).sum().item()
            accuracy_d+=len(x)
            real.append(feat.cpu().numpy())
    real_mu,real_cov=moments(real)
    test_acc=accuracy_n/accuracy_d
    gen=torch.Generator().manual_seed(a.seed)
    noise=torch.randn(a.samples,1,32,32,generator=gen)
    labels=torch.arange(a.samples)%10
    results=[]
    for mode,w in (("mf",1.),("imf",1.),("imf_cfg",1.),("imf_cfg",2.),("imf_cfg",3.),("imf_cfg",4.)):
        ckpt=runs/mode/"last.pt"
        if not ckpt.exists():raise FileNotFoundError(ckpt)
        sd=torch.load(ckpt,map_location=device,weights_only=False)
        model=MFDiT(**sd["model_config"]).to(device)
        model.load_state_dict(sd.get("ema_model",sd["model"]));model.eval()
        feats=[];matched=total=0
        with torch.inference_mode():
            for i in range(0,a.samples,a.batch_size):
                c=labels[i:i+a.batch_size].to(device)
                z=noise[i:i+a.batch_size].to(device)
                generated,_=sample_one_step(model,z,c,w)
                logits,features=net(to_images(generated))
                matched+=(logits.argmax(1)==c).sum().item()
                total+=len(c);feats.append(features.cpu().numpy())
        mu,cov=moments(feats)
        row={"model":mode,"cfg_scale":w,"training_steps":sd["step"],
             "mnist_feature_frechet":frechet(real_mu,real_cov,mu,cov),
             "label_consistency":matched/total,"classifier_test_accuracy":test_acc,
             "samples":a.samples}
        results.append(row);print(row,flush=True)
    out.mkdir(parents=True,exist_ok=True)
    with open(out/"metrics.csv","w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader();writer.writerows(results)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar([f'{r["model"]}\nw={r["cfg_scale"]:g}' for r in results],
           [r["mnist_feature_frechet"] for r in results])
    ax.set_ylabel("MNIST classifier feature distance (smaller is better)")
    ax.set_title("DiT: MNIST feature Fréchet (NOT ImageNet Inception FID)")
    ax.grid(axis="y",alpha=.2);fig.tight_layout()
    fig.savefig(out/"mf_vs_imf_metrics_dit.png",dpi=170);plt.close(fig)


if __name__=="__main__":
    main()
