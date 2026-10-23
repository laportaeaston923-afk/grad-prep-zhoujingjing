"""Explicitly specified RNN/LSTM/GRU forecasting baselines (not author checkpoints)."""
import argparse,json,time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from run import ROOT,data,seed
from evaluate_suite import Metrics

class Forecaster(nn.Module):
    def __init__(self,kind,hidden=128):
        super().__init__(); self.rnn=getattr(nn,kind)(52,hidden,batch_first=True,num_layers=1)
        self.head=nn.Linear(hidden,20*52)
    def forward(self,x):
        encoded,_=self.rnn(x); return self.head(encoded[:,-1]).reshape(-1,20,52)

def iteration(model,loader,device,optimizer=None):
    model.train(optimizer is not None); metric=Metrics()
    with torch.set_grad_enabled(optimizer is not None):
        for x,_,_,_,_ in loader:
            x=x.to(device); context=x[:,:80]; observed=context!=-1000
            count=observed.sum(1,keepdim=True).clamp_min(1)
            mean=(context*observed).sum(1,keepdim=True)/count
            std=(((context-mean).square()*observed).sum(1,keepdim=True)/count).sqrt()
            context=(context-mean)/(std+1e-8); context[~observed]=0
            pred=model(context)*std+mean; truth=x[:,80:]; valid=truth!=-1000
            loss=(pred[valid]-truth[valid]).square().mean()
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite baseline loss')
            if optimizer is not None:
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),3); optimizer.step()
            metric.add(pred[valid].detach(),truth[valid])
    return metric.value()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=30); p.add_argument('--datasets',nargs='+',default=['WiGesture','WiFall','WiCount']); p.add_argument('--models',nargs='+',choices=['RNN','LSTM','GRU'],default=['RNN','LSTM','GRU']); p.add_argument('--cpu',action='store_true'); a=p.parse_args()
    torch.set_num_threads(4); device=torch.device('cuda:0' if torch.cuda.is_available() and not a.cpu else 'cpu'); summary={}
    for dataset in a.datasets:
        seed(42); datasets=[data(s,2000 if s=='train' else None,s=='train',dataset=dataset) for s in ['train','valid','test']]
        loaders=[DataLoader(ds,batch_size=64,shuffle=i==0) for i,ds in enumerate(datasets)]
        for kind in a.models:
            out=ROOT/'runs'/f'{dataset.lower()}_{kind.lower()}'; out.mkdir(parents=True,exist_ok=True)
            if (out/'results.json').exists():
                summary[out.name]=json.loads((out/'results.json').read_text()); print('Already complete',out.name,flush=True); continue
            seed(42); model=Forecaster(kind).to(device); optim=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01,eps=1e-6)
            best=float('inf'); history=[]; start=time.time()
            for epoch in range(1,a.epochs+1):
                seed(42+epoch); train=iteration(model,loaders[0],device,optim); valid=iteration(model,loaders[1],device)
                history.append(dict(epoch=epoch,train=train,validation=valid))
                if valid['mse']<best: best=valid['mse']; torch.save(model.state_dict(),out/'model.pth')
            model.load_state_dict(torch.load(out/'model.pth',map_location=device,weights_only=True)); result=iteration(model,loaders[2],device)
            result.update(dataset=dataset,model=kind,epochs=a.epochs,parameters=sum(p.numel() for p in model.parameters()),seconds=time.time()-start,forecast_ratio=.2,device=str(device))
            (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            (out/'config.json').write_text(json.dumps(dict(model=kind,hidden_size=128,layers=1,input_dim=52,input_length=80,output_length=20,decoder='linear hidden state to all 20 future packets',learning_rate=.0005,optimizer='AdamW',loss='masked original-scale forecast MSE',batch_size=64,seed=42,split_window_counts=[len(ds) for ds in datasets],note='Independently specified baselines; paper did not publish these exact baseline architectures'),indent=2),encoding='utf-8')
            summary[out.name]=result; print(json.dumps(result),flush=True)
    (ROOT/'runs/recurrent_baselines.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

if __name__=='__main__': main()
