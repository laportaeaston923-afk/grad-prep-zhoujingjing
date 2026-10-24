"""Plot fixed held-out samples: observed CSI, recovery, forecast and true targets."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from run import ROOT,data,backbone,Token_Classifier,seed

RUNS={'WiGesture':'wigesture_paper_v2','WiFall':'wifall_paper_v2','WiCount':'wicount_paper_v2'}

def infer(x,t,model,mask,device):
    x=torch.tensor(x,device=device); t=torch.tensor(t,device=device)
    valid=x!=-1000; visible=valid&~torch.tensor(mask,device=device).unsqueeze(-1)
    n=visible.sum(1,keepdim=True).clamp_min(1)
    avg=(x*visible).sum(1,keepdim=True)/n
    std=(((x-avg).square()*visible).sum(1,keepdim=True)/n).sqrt()
    z=(x-avg)/(std+1e-8); noise=torch.randn_like(z); z[~visible]=noise[~visible]
    with torch.no_grad(): y=model(z,t)*std+avg
    return y.cpu().numpy()

def draw(ax,x,y,mask,t,future,title):
    channel=26; truth=x[:,channel].copy(); truth[truth==-1000]=np.nan
    ax.plot(t,truth,color='#657383',lw=1.6,label='True CSI')
    observed=~mask&np.isfinite(truth)
    ax.scatter(t[observed],truth[observed],s=13,color='#137eab',label='Observed input',zorder=3)
    if future:
        start=80; ax.axvspan(t[start],t[-1],color='#f4c88b',alpha=.27,label='Forecast region')
        ax.plot(t[start:],y[start:,channel],color='#d94736',lw=1.9,label='Model forecast')
    else:
        known=mask&np.isfinite(truth)
        ax.scatter(t[known],y[known,channel],s=38,marker='x',lw=1.5,color='#d94736',label='Recovered missing values',zorder=4)
    targets=mask[:,None]&(x!=-1000)
    mse=float(np.mean((y[targets]-x[targets])**2))
    compact=title.replace(', test sample ',' #')
    ax.set(title=f'{compact}\nMasked MSE over 52 carriers: {mse:.3f}',xlabel='Time (s)',ylabel='Amplitude (carrier 26)')
    ax.grid(alpha=.18)
    return mse

def main():
    torch.set_num_threads(4); seed(42)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dest=ROOT/'reports/generated_examples'; dest.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    overview,overview_axes=plt.subplots(3,2,figsize=(14,10),constrained_layout=True)
    heat,heat_axes=plt.subplots(3,3,figsize=(14,9),constrained_layout=True)
    metadata={}
    for row,(dataset,run) in enumerate(RUNS.items()):
        ds=data('test',None,False,dataset=dataset)
        x=ds.tensors[0][:3].numpy(); t=ds.tensors[-1][:3].numpy()
        rng=np.random.default_rng(42); recovery_mask=np.zeros((3,100),dtype=bool)
        for i in range(3): recovery_mask[i,rng.choice(100,15,replace=False)]=True
        future_mask=np.zeros((3,100),dtype=bool); future_mask[:,80:]=True
        estimates=[]
        for name,mask in [('pretrain',recovery_mask),('prediction',future_mask)]:
            seed(42); model=Token_Classifier(backbone(device),52).to(device)
            model.load_state_dict(torch.load(ROOT/'runs'/run/f'{name}.pth',map_location=device,weights_only=True)); model.eval()
            estimates.append(infer(x,t,model,mask,device)); del model
        recovered,predicted=estimates
        np.savez_compressed(dest/f'{dataset}_fixed_test_samples.npz',ground_truth=x,timestamp=t,recovery=recovered,prediction=predicted,recovery_mask=recovery_mask,future_mask=future_mask,test_indices=np.arange(3),channel_shown=26)
        fig,axes=plt.subplots(3,2,figsize=(14,10),constrained_layout=True)
        rows=[]
        for i in range(3):
            values=[]
            for col,(y,mask,future,task) in enumerate([(recovered,recovery_mask,False,'15% random recovery'),(predicted,future_mask,True,'Final 20% prediction')]):
                title=f'{dataset}, test sample {i}: {task}'
                value=draw(axes[i,col],x[i],y[i],mask[i],t[i]/1e6,future,title); values.append(value)
                if i==0: draw(overview_axes[row,col],x[i],y[i],mask[i],t[i]/1e6,future,title)
            rows.append(dict(test_index=i,recovery_mse=values[0],prediction_mse=values[1]))
        handles,labels=axes[0,0].get_legend_handles_labels(); fig.legend(handles,labels,loc='outside upper left',ncol=3)
        fig.savefig(dest/f'{dataset}_waveforms.png',dpi=170); plt.close(fig)
        truth=x[0].copy(); truth[truth==-1000]=np.nan
        completed=x[0].copy(); completed[future_mask[0]]=predicted[0,future_mask[0]]; completed[completed==-1000]=np.nan
        error=np.abs(completed-truth); error[:80]=np.nan
        lo,hi=np.nanpercentile(truth,[1,99])
        for col,(values,title) in enumerate([(truth,'True CSI'),(completed,'Observed + forecast'),(error,'Absolute forecast error')]):
            ax=heat_axes[row,col]
            im=ax.imshow(values.T,aspect='auto',origin='lower',cmap='magma' if col==2 else 'viridis',vmin=0 if col==2 else lo,vmax=None if col==2 else hi)
            ax.axvline(79.5,color='white',ls='--',lw=1.2)
            ax.set(title=f'{dataset}: {title}',xlabel='Packet index (forecast starts at 80)',ylabel='Subcarrier index'); heat.colorbar(im,ax=ax,shrink=.8)
        metadata[dataset]=dict(run=run,samples=rows)
    for ax in overview_axes[0]: ax.legend(fontsize=8,loc='best')
    overview.savefig(dest/'overview.png',dpi=170); plt.close(overview)
    heat.savefig(dest/'prediction_heatmaps.png',dpi=170); plt.close(heat)
    names=['wigesture_paper_v2','wigesture_no_gan','wigesture_no_arl','wigesture_no_time','wigesture_scratch']
    records=[json.loads((ROOT/'runs'/n/'results.json').read_text()) for n in names]
    labels=['Full','No GAN','No ARL','No time','From scratch*']
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    for ax,stage,title,key,indices in [(axes[0],'pretrain','Recovery MSE (lower is better)','mse',range(4)),(axes[1],'prediction','20% forecast MSE (lower is better)','mse',range(4)),(axes[2],'action','Gesture accuracy (higher is better)','accuracy',range(5))]:
        values=[records[i][stage][key]*(100 if key=='accuracy' else 1) for i in indices]
        bars=ax.bar([labels[i] for i in indices],values,color=['#147d92','#7fb7be','#dda15e','#a06a80','#6a8a5d'][:len(values)])
        ax.set_title(title); ax.grid(axis='y',alpha=.2); ax.tick_params(axis='x',rotation=20)
        ax.bar_label(bars,labels=[f'{v:.2f}'+('%' if key=='accuracy' else '') for v in values],padding=3,fontsize=9)
        ax.set_ylim(0,max(values)*1.17)
    fig.suptitle('WiGesture ablations: 30 epochs, one seed\n*From scratch trains the whole model; pretrained classification freezes the backbone',fontsize=11)
    fig.savefig(ROOT/'reports/ablations_complete.png',dpi=170); plt.close(fig)
    (dest/'sample_metrics.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (dest/'说明.md').write_text('本目录展示保存权重在真实测试数据上的输出，不是示意图。固定使用每个数据集的第 0、1、2 个测试窗口、零起始第 26 个子载波，没有按效果挑选样本。蓝点为可见输入，灰线为真实信号，红叉为随机遮挡位置的恢复值，红线为末尾 20% 的模型预测，浅橙色区域为预测区。原始缺失位置没有真实值，不计入误差。标题误差按该窗口全部 52 个子载波的有效被遮挡位置计算；单样本效果不代表总体精度。npz 文件保存真实值、预测值和掩码，可自行重画。\n',encoding='utf-8')
    print(dest)

if __name__=='__main__': main()
