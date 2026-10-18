"""Generate scientific result figures from exported real model outputs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
DEFAULT={'ddpm':100,'ddim':20,'vae':1,'ldm':20,'dit':20,'fm':20,'meanflow':1}
NAMES={'ddpm':'DDPM','ddim':'DDIM','vae':'VAE','ldm':'LDM','dit':'Tiny DiT','fm':'Flow Matching','meanflow':'MeanFlow'}

def load_run(name):
    text=(ROOT/'src'/'data'/f'{name}.js').read_text(encoding='utf-8')
    return json.loads(text.split('=',1)[1].rstrip(';'))

if __name__=='__main__':
    meta=json.loads((ROOT/'results'/'metrics.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(7,10,figsize=(12,9),facecolor='#f6f7fb')
    for row,(key,nfe) in enumerate(DEFAULT.items()):
        run=load_run(f'{key}-{nfe}')
        for col in range(10):
            ax=axes[row,col];ax.imshow(np.array(run['samples'][col]).reshape(8,8),cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'Class {col}',fontsize=9)
            if col==0:ax.set_ylabel(f'{NAMES[key]}\nNFE {nfe}',fontsize=9,labelpad=12)
    fig.suptitle('GenCompare | real 8x8 Digits generation / seed 42 / first sample per class',fontsize=13)
    fig.tight_layout(rect=[0,.01,1,.97]);fig.savefig(ROOT/'results'/'generation_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for key,nfe in DEFAULT.items():
        r=next(r for r in meta['records'] if r['id']==f'{key}-{nfe}')
        axes[0].errorbar(r['batch_ms'],r['mmd2'],xerr=r['batch_ms_std'],yerr=r['mmd2_std'],fmt='o',capsize=3)
        axes[0].annotate(NAMES[key],(r['batch_ms'],r['mmd2']),xytext=(5,6),textcoords='offset points',fontsize=8)
    axes[0].set(xlabel='200-image batch inference time (ms)',ylabel='Pixel RBF MMD squared (lower is better)',title='Default configurations / 3 sampling seeds')
    x=np.arange(len(DEFAULT));values=[next(r for r in meta['records'] if r['id']==f'{key}-{nfe}')['label_consistency'] for key,nfe in DEFAULT.items()]
    axes[1].bar(x,values,color='#7763d7');axes[1].set_xticks(x,[NAMES[k] for k in DEFAULT],rotation=35,ha='right');axes[1].set(ylim=(0,1.08),ylabel='Conditional label consistency',title='Auxiliary classifier is a proxy, not image quality')
    fig.savefig(ROOT/'results'/'benchmark.png',dpi=150);plt.close(fig)
    source=(ROOT/'src'/'data'/'meta.js').read_text(encoding='utf-8').split('=',1)[1].split(';\nwindow.GEN_RUNS')[0]
    web=json.loads(source)
    fig,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
    for ax,(key,loss) in zip(axes.flat,web['loss'].items()):
        ax.plot(np.arange(len(loss))*25,loss,color='#7763d7');ax.set(title=NAMES[key],xlabel='Optimizer step',ylabel='Model-specific loss')
    fig.suptitle('Training curves | distinct objectives, do not compare raw loss magnitudes')
    fig.savefig(ROOT/'results'/'training_curves.png',dpi=150);plt.close(fig)
    print('3 result figures generated')
