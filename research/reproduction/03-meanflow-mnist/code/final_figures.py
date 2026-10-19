import csv,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser()
p.add_argument('--results',default=str(Path(__file__).resolve().parents[1]/'results'/'dit_stable'))
OUT=Path(p.parse_args().results)
def read(name):
    with (OUT/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
fig,axes=plt.subplots(1,3,figsize=(16,4.7))
for mode,label,color in [('mf','MF','tab:blue'),('imf','iMF','tab:orange')]:
    rows=read(f'{mode}_train.csv')
    x=np.array([int(r['step']) for r in rows]);y=np.array([float(r['offdiag_v_loss']) for r in rows])
    axes[0].plot(x,y,color=color,alpha=.22,lw=.7)
    axes[0].plot(x[4:-4],np.convolve(y,np.ones(9)/9,'valid'),color=color,label=label,lw=1.8)
axes[0].set(title='MF vs iMF: plain off-diagonal loss',xlabel='Training steps',ylabel='v-MSE (t != r)')
rows=read('fid_history.csv')
for mode,label,w in [('mf','MF',1.),('imf','iMF',1.),('imf_cfg','iMF + CFG (w=2)',2.)]:
    rs=sorted([r for r in rows if r['mode']==mode and float(r['cfg_scale'])==w],key=lambda r:int(r['step']))
    axes[1].plot([int(r['step']) for r in rs],[float(r['fid']) for r in rs],'o-',label=label,ms=4)
axes[1].set(title='1-NFE FID during training',xlabel='Training steps',ylabel='MNIST Inception FID-2000')
for step in (4000,8000,12000,16000,20000):
    rs=sorted([r for r in rows if r['mode']=='imf_cfg' and int(r['step'])==step],key=lambda r:float(r['cfg_scale']))
    axes[2].plot([float(r['cfg_scale']) for r in rs],[float(r['fid']) for r in rs],'o-',label=f'{step//1000}k steps',ms=3)
axes[2].set(title='Flexible CFG across checkpoints',xlabel='CFG scale',ylabel='MNIST Inception FID-2000')
for ax in axes:
    ax.grid(alpha=.18);ax.legend(fontsize=8)
fig.suptitle('MNIST / DiT | matched MF-iMF settings | EMA, 1-NFE | FID seed 2026; not ImageNet FID-50K',fontsize=11)
fig.text(.012,.012,'Loss: faint raw 100-step logs + 9-point moving average. All recorded FID points retained.',fontsize=8,color='#555555')
fig.tight_layout(rect=[0,.035,1,.94])
fig.savefig(OUT/'comparison_curves.png',dpi=200)
fig.savefig(OUT/'comparison_curves.svg')
plt.close(fig)
