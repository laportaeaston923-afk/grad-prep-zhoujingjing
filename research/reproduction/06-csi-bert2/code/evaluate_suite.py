"""Shared-mask regression baselines and packet-loss robustness on held-out data."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from run import ROOT,data,backbone,seed,Token_Classifier,Sequence_Classifier,DATASETS,classify_eval

class Metrics:
    def __init__(self): self.se=self.energy=self.ape=self.sape=self.n=self.nz=0
    def add(self,pred,truth):
        diff=(pred-truth).abs(); nonzero=truth.abs()>1e-8
        self.se+=diff.square().sum().item(); self.energy+=truth.square().sum().item(); self.n+=truth.numel()
        self.ape+=(diff[nonzero]/truth[nonzero].abs()).sum().item(); self.nz+=nonzero.sum().item()
        self.sape+=(2*diff/(pred.abs()+truth.abs()+1e-8)).sum().item()
    def value(self): return dict(mse=self.se/self.n,nmse=self.se/self.energy,mape=self.ape/max(self.nz,1),smape=self.sape/self.n,values=self.n)

def interpolation(x,t,visible,query):
    linear=torch.zeros_like(x); idw=torch.zeros_like(x); mean=torch.zeros_like(x)
    for i in range(len(x)):
        obs=visible[i].nonzero().flatten(); target=query[i].nonzero().flatten()
        if not len(target): continue
        if not len(obs): raise ValueError('All context missing')
        ts=t[i,obs]; qs=t[i,target]; values=x[i,obs]
        insertion=torch.searchsorted(ts.contiguous(),qs.contiguous())
        left=(insertion-1).clamp(0,len(obs)-1); right=insertion.clamp(0,len(obs)-1)
        blend=((qs-ts[left])/(ts[right]-ts[left]).clamp_min(1e-8)).clamp(0,1)
        linear[i,target]=values[left]*(1-blend[:,None])+values[right]*blend[:,None]
        distance=(qs[:,None]-ts[None,:]).abs().clamp_min(1)
        weights=distance.pow(-2); weights=weights/weights.sum(-1,keepdim=True)
        idw[i,target]=weights@values
        mean[i,target]=values.mean(0)
    return {'Linear_boundary_hold':linear,'IDW_power2':idw,'Visible_mean':mean}

def evaluate_regression(model,loader,device,future,ratio):
    seed(20042)
    metrics={k:Metrics() for k in ['CSI-BERT2','Linear_boundary_hold','IDW_power2','Visible_mean']}
    model.eval()
    with torch.no_grad():
        for x,_,_,_,t in loader:
            x=x.to(device); t=t.to(device); valid=x[...,0]!=-1000
            mask=torch.zeros_like(valid); count=round(100*ratio)
            if future: mask[:,-count:]=True
            else:
                for row in range(len(x)): mask[row,torch.randperm(100,device=device)[:count]]=True
            query=mask&valid; visible=(~mask)&valid
            w=visible.unsqueeze(-1); n=w.sum(1,keepdim=True).clamp_min(1)
            avg=(x*w).sum(1,keepdim=True)/n; std=(((x-avg).square()*w).sum(1,keepdim=True)/n).sqrt()
            normalized=(x-avg)/(std+1e-8); noise=torch.randn_like(x); normalized[~visible]=noise[~visible]
            estimates=interpolation(x,t,visible,query); estimates['CSI-BERT2']=model(normalized,t)*std+avg
            for name,y in estimates.items(): metrics[name].add(y[query],x[query])
    return {k:v.value() for k,v in metrics.items()}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--runs',nargs='+',default=['wigesture_paper_v2','wifall_paper_v2','wicount_paper_v2']); a=p.parse_args()
    torch.set_num_threads(4); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    allrows=[]
    for name in a.runs:
        out=ROOT/'runs'/name; config=json.loads((out/'config.json').read_text()); args=config['arguments']; dataset=args.get('dataset','WiGesture')
        results=json.loads((out/'results.json').read_text()); test=data('test',None,False,dataset=dataset)
        loader=DataLoader(test,batch_size=64); extra=dict(dataset=dataset,run=name,regression={},classification={})
        for stage,task,ratios in [('pretrain','recovery',[.15,.3,.5,.7]),('prediction','prediction',[.15,.2,.3,.4])]:
            if stage not in results: continue
            model=Token_Classifier(backbone(device,args.get('ablation','none')),52).to(device)
            model.load_state_dict(torch.load(out/f'{stage}.pth',map_location=device,weights_only=True))
            for ratio in ratios:
                metrics=evaluate_regression(model,loader,device,stage=='prediction',ratio)
                extra['regression'][f'{task}_{ratio}']=metrics
                for method,values in metrics.items(): allrows.append(dict(dataset=dataset,run=name,task=task,ratio=ratio,method=method,**values))
            del model
        for task,classes in DATASETS[dataset].items():
            if task not in results: continue
            model=Sequence_Classifier(backbone(device,args.get('ablation','none')),classes).to(device)
            model.load_state_dict(torch.load(out/f'{task}.pth',map_location=device,weights_only=True)); metrics={}
            for ratio in [0,.15,.3,.5,.7]:
                seed(30042); tensors=list(test.tensors); x=tensors[0].clone()
                for row in range(len(x)):
                    if ratio: x[row,torch.randperm(100)[:round(100*ratio)]]=-1000
                tensors[0]=x
                metrics[str(ratio)]=classify_eval(model,DataLoader(TensorDataset(*tensors),batch_size=64),device,task)
                allrows.append(dict(dataset=dataset,run=name,task=task,ratio=ratio,method='CSI-BERT2',**{k:v for k,v in metrics[str(ratio)].items() if k!='confusion_matrix'}))
            extra['classification'][task]=metrics; del model
        (out/'extended_metrics.json').write_text(json.dumps(extra,indent=2),encoding='utf-8')
        print('Evaluated',name,flush=True)
    baseline_file=ROOT/'runs/recurrent_baselines.json'
    if baseline_file.exists():
        for name,v in json.loads(baseline_file.read_text()).items():
            allrows.append(dict(dataset=v['dataset'],run=name,task='prediction',ratio=.2,method=v['model'],**{k:v[k] for k in ['mse','nmse','mape','smape','values','parameters','epochs','device']}))
    columns=list(dict.fromkeys(k for row in allrows for k in row))
    target=ROOT/'runs/comparison.csv'
    with target.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns); writer.writeheader(); writer.writerows(allrows)
    print(target)

if __name__=='__main__': main()
