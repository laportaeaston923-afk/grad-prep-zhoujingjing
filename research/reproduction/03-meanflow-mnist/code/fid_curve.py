"""Actual pytorch-fid pool3 (2048-d) on MNIST, grayscale repeated to RGB.

Fixed held-out test reference, seed, class balance and sample count for all runs.
This is MNIST FID-N, not the paper's ImageNet FID-50K.
"""
import os
from pathlib import Path
os.environ.setdefault('TORCH_HOME',str(Path(__file__).resolve().parents[1]/'work'/'torch_cache'))
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('MKL_NUM_THREADS','8')
os.environ['OPENBLAS_NUM_THREADS']='8'
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms
from pytorch_fid.inception import InceptionV3
from scipy.linalg import eigh,eigvalsh
from models.dit import MFDiT
from experiment_core import sample_one_step,to_images
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_real_root_cache=None
def calculate_frechet_distance(mu1,sigma1,mu2,sigma2):
    # Equivalent PSD form of the standard FID covariance trace. Avoids slow
    # complex Schur sqrtm for singular 2048-d sample covariance (N=2000).
    global _real_root_cache
    if _real_root_cache is None or _real_root_cache[0] is not sigma1:
        values,vectors=eigh((sigma1+sigma1.T)*.5,check_finite=True)
        if values.min() < -1e-7*max(1.,values.max()):
            raise ValueError('Reference covariance is not positive semidefinite')
        root=(vectors*np.sqrt(np.maximum(values,0)))@vectors.T
        _real_root_cache=(sigma1,root)
    root=_real_root_cache[1]
    product=root@sigma2@root
    values=eigvalsh((product+product.T)*.5,check_finite=True)
    if values.min() < -1e-7*max(1.,values.max()):
        raise ValueError('Generated covariance is not positive semidefinite')
    score=(mu1-mu2)@(mu1-mu2)+np.trace(sigma1)+np.trace(sigma2)-2*np.sqrt(np.maximum(values,0)).sum()
    return float(max(0.,score))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs-root',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--results',required=True)
    p.add_argument('--samples',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--steps',type=int,nargs='*')
    a=p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    dev=torch.device('cuda')
    out=Path(a.results);out.mkdir(parents=True,exist_ok=True)
    runs=Path(a.runs_root)
    extractor=InceptionV3([3]).to(dev).eval()
    def features(images):
        with torch.inference_mode():
            return extractor(images.repeat(1,3,1,1).to(dev))[0].flatten(1).cpu().numpy()
    cache_dir=Path(os.environ['TORCH_HOME']).parent/'fid_reference_cache'
    cache_dir.mkdir(parents=True,exist_ok=True)
    cache=cache_dir/f'real_pool3_{a.samples}_seed2026.npz'
    legacy_cache=out/cache.name
    if legacy_cache.exists() and not cache.exists():legacy_cache.replace(cache)
    if cache.exists():
        ref=np.load(cache);mu,cov=ref['mu'],ref['cov']
    else:
        ds=datasets.MNIST(a.data_root,train=False,download=False,
            transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()]))
        # Random fixed held-out subset rather than the ordered first test images.
        ids=torch.randperm(len(ds),generator=torch.Generator().manual_seed(2026))[:a.samples]
        real=[]
        for x,_ in DataLoader(Subset(ds,ids.tolist()),batch_size=a.batch_size):
            real.append(features(x))
        real=np.concatenate(real).astype(np.float64)
        mu,cov=real.mean(0),np.cov(real,rowvar=False)
        np.savez(cache,mu=mu,cov=cov,indices=ids.numpy())
        print('Reference moments ready',flush=True)
    file=out/'fid_history.csv'
    rows=list(csv.DictReader(file.open(newline=''))) if file.exists() else []
    keys={(r['mode'],int(r['step']),float(r['cfg_scale']),int(r['samples'])) for r in rows}
    noise=torch.randn(a.samples,1,32,32,generator=torch.Generator().manual_seed(2026))
    labels=torch.arange(a.samples)%10
    for mode in ('mf','imf','imf_cfg'):
        for ckpt in sorted((runs/mode).glob('step_*.pt')):
            step=int(ckpt.stem.split('_')[-1])
            if a.steps and step not in a.steps:continue
            scales=(1.,1.5,2.,2.5,3.,4.,5.,6.,8.) if mode=='imf_cfg' else (1.,)
            pending=[w for w in scales if (mode,step,w,a.samples) not in keys]
            if not pending:continue
            state=torch.load(ckpt,map_location='cpu',weights_only=False)
            net=MFDiT(**state['model_config']).to(dev).eval()
            net.load_state_dict(state.get('ema_model',state['model']))
            del state
            for w in pending:
                start=time.perf_counter();chunks=[]
                with torch.inference_mode():
                    for i in range(0,a.samples,a.batch_size):
                        z,_=sample_one_step(net,noise[i:i+a.batch_size].to(dev),
                            labels[i:i+a.batch_size].to(dev),w)
                        chunks.append(features(to_images(z)))
                fake=np.concatenate(chunks).astype(np.float64)
                score=calculate_frechet_distance(mu,cov,fake.mean(0),np.cov(fake,rowvar=False))
                row=dict(mode=mode,step=step,cfg_scale=w,samples=a.samples,
                         fid=score,seed=2026,weights='ema',feature='pytorch-fid_pool3_2048',
                         reference='MNIST_test_fixed_random_subset',nfe=1)
                rows.append(row)
                with file.open('w',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
                print(f'FID {mode} step={step} w={w}: {score:.4f} ({time.perf_counter()-start:.1f}s)',flush=True)
                plot(rows,out)
            del net;torch.cuda.empty_cache()
    (out/'fid_protocol.json').write_text(json.dumps(dict(samples=a.samples,seed=2026,
        feature='FID-specific Inception-v3 pool3 2048',grayscale='repeat RGB',resize='pytorch-fid bilinear 299',
        input='MNIST 32x32, clipped [0,1]',reference='fixed random held-out test subset',
        inference='EMA, 1 NFE',paper_comparable=False),indent=2))

def plot(rows,out):
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for mode,label in [('mf','MF'),('imf','iMF'),('imf_cfg','iMF + flexible CFG (w=2)')]:
        rs=sorted([r for r in rows if r['mode']==mode and float(r['cfg_scale'])==(2. if mode=='imf_cfg' else 1.)],key=lambda r:int(r['step']))
        if rs:axes[0].plot([int(r['step']) for r in rs],[float(r['fid']) for r in rs],'o-',label=label)
    axes[0].set(xlabel='Training steps',ylabel='MNIST Inception FID (lower is better)',title='1-NFE FID during training')
    cfg=[r for r in rows if r['mode']=='imf_cfg']
    for step in sorted({int(r['step']) for r in cfg}):
        rs=sorted([r for r in cfg if int(r['step'])==step],key=lambda r:float(r['cfg_scale']))
        axes[1].plot([float(r['cfg_scale']) for r in rs],[float(r['fid']) for r in rs],'o-',label=f'{step:,} steps')
    axes[1].set(xlabel='CFG scale',ylabel='MNIST Inception FID (lower is better)',title='Flexible guidance: fixed model and noise')
    for ax in axes:
        ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=8)
    fig.suptitle(f"MNIST FID-{rows[0]['samples']}; EMA, seed 2026; not ImageNet FID-50K",fontsize=10)
    fig.tight_layout();fig.savefig(out/'fid_and_cfg_curves.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
