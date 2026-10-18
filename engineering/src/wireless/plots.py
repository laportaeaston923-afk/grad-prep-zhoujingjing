"""Scientific figures from actual inference arrays, no illustrative fake results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(output, clean, observed, recovered, baseline, losses, frames, metrics):
    plt.rcParams.update({'figure.facecolor':'#0d1420','axes.facecolor':'#121e2c','text.color':'#e4eef8',
                         'axes.labelcolor':'#aabed0','xtick.color':'#aabed0','ytick.color':'#aabed0',
                         'axes.edgecolor':'#304258','font.size':10,'savefig.facecolor':'#0d1420'})
    fig, axes = plt.subplots(3,1,figsize=(11,7),constrained_layout=True)
    for ax, arr, title in zip(axes,[clean,observed,recovered],['Clean simulated CSI','Corrupted CSI: noise + packet loss','Conditional DDPM recovery']):
        image=ax.imshow(arr[0],aspect='auto',origin='lower',extent=[0,12.8,0,8],cmap='magma',vmin=-1,vmax=1)
        ax.set_title(title,loc='left'); ax.set_ylabel('Subcarrier')
    axes[-1].set_xlabel('Time (s)');fig.colorbar(image,ax=axes,label='Calibrated amplitude')
    fig.savefig(output/'csi_heatmaps.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    t=np.arange(128)/10
    for a,label,color in [(observed,'Observed','#657e99'),(baseline,'Moving average','#efbc66'),(clean,'Clean','#ffffff'),(recovered,'Conditional DDPM','#50dbcb')]:
        axes[0].plot(t,a[0,0],label=label,color=color,lw=1.5,alpha=.9)
    axes[0].set(xlabel='Time (s)',ylabel='Amplitude',title='First holdout window / subcarrier 0')
    axes[0].legend(fontsize=8)
    values=metrics['rmse'];axes[1].bar(list(values),list(values.values()),color=['#657e99','#efbc66','#50dbcb'])
    axes[1].set(ylabel='RMSE (lower is better)',title='128 independent simulated test windows')
    for i,v in enumerate(values.values()):axes[1].text(i,v,f'{v:.4f}',ha='center',va='bottom')
    fig.savefig(output/'recovery_comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.5),constrained_layout=True)
    ax.plot(losses,color='#50dbcb',alpha=.2,lw=.6)
    n=50;avg=np.convolve(losses,np.ones(n)/n,mode='valid');ax.plot(np.arange(n-1,len(losses)),avg,color='#50dbcb',lw=1.5)
    ax.set(xlabel='Optimizer step',ylabel='Noise prediction MSE',title='Conditional DDPM training / 50-step moving average')
    fig.savefig(output/'training_loss.png',dpi=150);plt.close(fig)
    chosen=[0,4,10,16,len(frames)-1]
    fig,axes=plt.subplots(1,5,figsize=(13,2.5),constrained_layout=True)
    for ax,idx in zip(axes,chosen):
        step,values=frames[idx];ax.imshow(values[0],aspect='auto',origin='lower',cmap='magma',vmin=-1,vmax=1);ax.set_title(f't = {step}');ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Actual reverse diffusion states / first holdout window')
    fig.savefig(output/'diffusion_trajectory.png',dpi=150);plt.close(fig)


if __name__ == '__main__':
    import sys
    import json
    from pathlib import Path
    output = Path(sys.argv[1])
    d = np.load(output/'evaluation.npz')
    frames = [(int(t), value[None]) for t,value in zip(d['frame_steps'], d['frame_values'])]
    render(output, d['clean'], d['observed'], d['recovered'], d['baseline'], d['loss'], frames,
           json.loads((output/'metrics.json').read_text(encoding='utf-8')))
