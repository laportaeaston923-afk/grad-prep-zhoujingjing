"""Paper-style strips of actual CSI-BERT2 full-window recovery outputs."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from run import ROOT,data,backbone,Token_Classifier,seed
from visualize_outputs import infer,RUNS

def main():
    torch.set_num_threads(4); seed(42)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dest=ROOT/'reports/full_reconstruction'; dest.mkdir(parents=True,exist_ok=True)
    for dataset,run in RUNS.items():
        ds=data('test',None,False,dataset=dataset)
        x=ds.tensors[0][:1].numpy(); t=ds.tensors[-1][:1].numpy()
        model=Token_Classifier(backbone(device),52).to(device)
        model.load_state_dict(torch.load(ROOT/'runs'/run/'pretrain.pth',map_location=device,weights_only=True)); model.eval()
        # No additional artificial masking: replace real missing packets with noise.
        y=infer(x,t,model,np.zeros((1,100),dtype=bool),device)
        observed=x!=-1000; filled=np.where(observed,x,y)
        original=x[0].copy(); original[~observed[0]]=np.nan
        lo,hi=np.nanmin(original),np.nanmax(original)
        fig,axes=plt.subplots(3,1,figsize=(10,5.8),sharex=True,constrained_layout=True)
        rows=[(original,'Original CSI'),(filled[0],'CSI-BERT2: fill missing only'),(y[0],'CSI-BERT2: reconstruct all')]
        for ax,(values,label) in zip(axes,rows):
            im=ax.imshow(values.T,origin='lower',aspect='auto',cmap='viridis',interpolation='nearest',vmin=lo,vmax=hi)
            ax.set_yticks([0,26,51]); ax.set_ylabel('Subcarrier')
            ax.text(1.02,.5,label,transform=ax.transAxes,ha='left',va='center',fontsize=10)
        axes[-1].set_xlabel('Packet index'); fig.suptitle(f'{dataset}: fixed test sample 0, 100 packets, 52 subcarriers',fontsize=12)
        fig.colorbar(im,ax=axes,location='bottom',shrink=.65,pad=.04,label='CSI amplitude (same scale in every strip)')
        fig.savefig(dest/f'{dataset}_reconstruction.png',dpi=180); plt.close(fig)
        np.savez_compressed(dest/f'{dataset}_reconstruction.npz',original=x,observed=observed,fill_missing=filled,reconstruct_all=y,timestamp=t)
        assert np.isfinite(filled).all() and np.isfinite(y).all()
        assert np.array_equal(filled[observed],x[observed])
        del model
    (dest/'说明.md').write_text('这些是真实保存权重的输出，固定使用三个数据集的第 0 个测试窗口，没有按效果挑样本。第一行原始 CSI 的白条代表原始丢包。第二行只补缺失位置，保留全部观测值。第三行把全部位置替换为 CSI-BERT2 的重建输出，包含已观测位置。第二、三行是同一模型的两种输出方式，不是不同模型；全段重建展示没有额外遮挡已知包，因此不能用这张图衡量对未知包的预测准确性。三行使用完全相同的颜色范围。BERT 和 CSI-BERT 第一版没有完成独立训练，未伪造相应对比行。\n',encoding='utf-8')
    print(dest)

if __name__=='__main__': main()
