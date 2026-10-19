"""Figures from actual model evaluation, in a separate plotting process."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
    cases=json.loads((ROOT/'results'/'cases.json').read_text(encoding='utf-8'))
    for kind,data in cases.items():
        clean=np.array(data['clean']).reshape(-1,28,28);mask=np.array(data['mask']).reshape(-1,28,28)
        baseline=np.array(data['baseline']).reshape(-1,28,28);recovered=np.array(data['recovered']).reshape(-1,28,28)
        fig,axes=plt.subplots(4,8,figsize=(12,6.2))
        rows=[clean,clean*(1-mask),baseline,recovered]
        for row,values in enumerate(rows):
            for i in range(8):
                ax=axes[row,i];ax.imshow(values[i],cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
                if i==0:ax.set_ylabel(['Reference','Masked input','Harmonic','DDIM 50'][row],fontsize=9,labelpad=12)
                if row==0:ax.set_title('Test #'+str(data['indices'][i]),fontsize=8)
        fig.suptitle('DigitRepair | '+kind+' mask / actual held-out inference',fontsize=12);fig.tight_layout();fig.savefig(ROOT/'results'/f'recovery_{kind}.png',dpi=150);plt.close(fig)
    metrics=json.loads((ROOT/'results'/'metrics.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,kind in zip(axes,['center','brush','half']):
        values=[r for r in metrics['records'] if r['mask']==kind]
        labels=[r['method']+(str(r['steps']) if 'steps' in r else '') for r in values]
        ax.bar(np.arange(len(values)),[r['mean_masked_rmse'] for r in values],color=['#b5b2a8','#7f927c','#dfa76e','#c87931','#779caa'])
        ax.set_xticks(np.arange(len(values)),labels,rotation=35,ha='right',fontsize=8);ax.set_title(kind+' masks');ax.set_ylabel('Mean masked RMSE [0,1]')
    fig.suptitle('80 official MNIST test images / no best-candidate selection');fig.savefig(ROOT/'results'/'benchmark.png',dpi=150);plt.close(fig)
    training=json.loads((ROOT/'results'/'training.json').read_text(encoding='utf-8'))
    fig,ax=plt.subplots(figsize=(9,3.5),constrained_layout=True);ax.plot(np.arange(len(training['loss']))*25,training['loss'],color='#c87931');ax.set(xlabel='Optimizer step',ylabel='Weighted noise prediction MSE',title='Conditional U-Net training / 25-step averages');fig.savefig(ROOT/'results'/'training_loss.png',dpi=150);plt.close(fig)
    print('5 result figures generated')
