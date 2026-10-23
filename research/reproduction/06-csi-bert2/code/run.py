"""Bounded, seeded real-data reproduction using the author's architecture and losses."""
import argparse, json, os, pickle, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from transformers import BertConfig
from model import CSIBERT, Token_Classifier, Sequence_Classifier
import pretrain, prediction, finetune

def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)

DATASETS={'WiGesture':{'action':6,'people':8},'WiFall':{'action':5,'fall':2,'people':10},'WiCount':{'people':4}}

def data(split, count, random_windows, gap=1,dataset='WiGesture'):
    path=ROOT/'data/processed/data_sequence.pkl' if dataset=='WiGesture' else ROOT/'data'/dataset/'processed/data_sequence.pkl'
    with path.open('rb') as f: records=pickle.load(f)
    samples=[]
    for rec in records:
        n=len(rec['time']); train=int(n*.81); valid=int(n*.90)
        if 'split' in rec:
            if rec['split']!=split: continue
            lo,hi=0,n
        else:
            lo,hi={'train':(0,train),'valid':(train,valid),'test':(valid,n)}[split]
        # Split original recording before any crop or downsampling, preventing overlap.
        for start in range(lo,hi-100*gap+1,100*gap):
            idx=np.arange(start,start+100*gap,gap)
            x=rec['magnitude'][idx]; t=rec['time'][idx]
            if np.mean(x[:,0]!=-1000)<.5 or t[-1]<=t[0]: continue
            samples.append((x,t-t[0],rec['action'],rec['people']))
    if not samples: raise ValueError(f'No valid complete windows in {split}')
    if random_windows and count and len(samples)>count:
        samples=[samples[i] for i in np.random.choice(len(samples),count,replace=False)]
    x=np.stack([s[0] for s in samples]); t=np.stack([s[1] for s in samples]).astype(np.float32)
    a=np.array([s[2] for s in samples]); p=np.array([s[3] for s in samples])
    return TensorDataset(torch.from_numpy(x),torch.zeros(len(x)),torch.from_numpy(a),torch.from_numpy(p),torch.from_numpy(t))

def backbone(device,ablation='none'):
    config=BertConfig(max_position_embeddings=100,hidden_size=128,num_hidden_layers=6,num_attention_heads=8,intermediate_size=512,hidden_dropout_prob=.1,attention_probs_dropout_prob=.1)
    model=CSIBERT(config,52).to(device)
    if ablation=='no_arl': model.attention=lambda x:x
    if ablation=='no_time': model.time_embedding=lambda timestamp:torch.zeros((*timestamp.shape,128),device=timestamp.device)
    return model

def classify_eval(model,loader,device,task):
    classes=model.classifier[-1].out_features
    correct=total=0; confusion=torch.zeros((classes,classes),dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for x,_,a,p,t in loader:
            x=x.to(device); t=t.to(device); target=a.clone() if task in ['action','fall'] else p
            if task=='fall': target=(target!=0).long()
            valid=x!=-1000; count=valid.sum(1,keepdim=True).clamp_min(1)
            mean=(x*valid).sum(1,keepdim=True)/count
            std=(((x-mean).square()*valid).sum(1,keepdim=True)/count).sqrt()
            x=(x-mean)/(std+1e-8); noise=torch.randn_like(x); x[~valid]=noise[~valid]
            pred=model(x,t).argmax(-1).cpu()
            correct+=(pred==target).sum().item(); total+=len(target)
            for truth,guess in zip(target,pred): confusion[truth,guess]+=1
    matrix=confusion.float(); tp=matrix.diag(); recall=tp/matrix.sum(1).clamp_min(1); precision=tp/matrix.sum(0).clamp_min(1)
    f1=2*precision*recall/(precision+recall).clamp_min(1e-8)
    result={'accuracy':correct/total,'balanced_accuracy':recall.mean().item(),'macro_f1':f1.mean().item(),'samples':total,'confusion_matrix':confusion.tolist()}
    if task=='fall': result['fall_precision']=precision[0].item(); result['fall_recall']=recall[0].item()
    return result

def regression_eval(model,loader,device,future=False,ratio=.15):
    model.eval(); se=energy=absolute_pct=symmetric_pct=denom=pctdenom=0
    with torch.no_grad():
        for x,_,_,_,t in loader:
            x=x.to(device); t=t.to(device); mask=torch.zeros(x.shape[:2],device=device,dtype=torch.bool)
            count=max(1,round(x.shape[1]*ratio))
            if future: mask[:,-count:]=True
            else:
                for row in range(len(x)): mask[row,torch.randperm(100,device=device)[:count]]=True
            valid=x!=-1000; mask &= valid[...,0]
            visible=((~mask).unsqueeze(-1)&valid).float()
            mean=(x*visible).sum(1,keepdim=True)/visible.sum(1,keepdim=True).clamp_min(1)
            std=(((x-mean).square()*visible).sum(1,keepdim=True)/visible.sum(1,keepdim=True).clamp_min(1)).sqrt()
            z=(x-mean)/(std+1e-8); noise=torch.randn_like(z); z[~valid]=noise[~valid]; z[mask]=noise[mask]
            y=model(z,t)*std+mean
            truth=x[mask]; pred=y[mask]; diff=truth-pred
            se+=diff.square().sum().item(); energy+=truth.square().sum().item(); denom+=truth.numel()
            nz=truth.abs()>1e-8
            absolute_pct+=(diff[nz].abs()/truth[nz].abs()).sum().item(); pctdenom+=nz.sum().item()
            symmetric_pct+=(2*diff.abs()/(truth.abs()+pred.abs()+1e-8)).sum().item()
    return dict(mse=se/denom,nmse=se/energy,mape=absolute_pct/max(1,pctdenom),smape=symmetric_pct/denom,masked_values=denom,ratio=ratio)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['pretrain','prediction','action','fall','people','all'],default='all')
    parser.add_argument('--dataset',choices=list(DATASETS),default='WiGesture')
    parser.add_argument('--paper-mask',action='store_true',help='MLM 15-70%%, evaluation forecast 20%%')
    parser.add_argument('--ablation',choices=['none','no_arl','no_time'],default='none')
    parser.add_argument('--no-pretrain',action='store_true',help='Train classification model from scratch')
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--train-samples',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--run-name',default='paper_run')
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--no-gan',action='store_true')
    args=parser.parse_args()
    if args.epochs<1: parser.error('--epochs must be positive')
    if args.no_pretrain and args.stage not in DATASETS[args.dataset]: parser.error('--no-pretrain requires a classification stage')
    if args.stage in ['action','fall','people'] and args.stage not in DATASETS[args.dataset]: parser.error('Task unavailable for selected dataset')
    torch.set_num_threads(4); seed(args.seed)
    device=torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda:0')
    out=ROOT/'runs'/args.run_name; out.mkdir(parents=True,exist_ok=True)
    meta=dict(arguments=vars(args),torch=torch.__version__,cuda=torch.version.cuda,device=str(device),gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,split='WiFall: recording-disjoint 81/9/10 within person/action groups; other datasets: within each recording 81/9/10',note='Test set is never used for checkpoint selection',prediction_ratio=.20 if args.paper_mask else .15)
    (out/'config.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    train=data('train',args.train_samples,True,dataset=args.dataset); valid=data('valid',None,False,dataset=args.dataset); test=data('test',None,False,dataset=args.dataset)
    meta['window_counts']={k:len(v) for k,v in zip(['train','valid','test'],[train,valid,test])}
    (out/'config.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    loaders=[DataLoader(ds,batch_size=args.batch_size,shuffle=i==0,num_workers=0) for i,ds in enumerate([train,valid,test])]
    print(f'Device {device}; windows train/valid/test {len(train)}/{len(valid)}/{len(test)}',flush=True)
    stages=['pretrain','prediction']+list(DATASETS[args.dataset]) if args.stage=='all' else [args.stage]
    results=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else {}
    for stage in stages:
        seed(args.seed); bert=backbone(device,args.ablation)
        classification=stage in ['action','fall','people']
        model=Sequence_Classifier(bert,DATASETS[args.dataset][stage]).to(device) if classification else Token_Classifier(bert,52).to(device)
        discriminator=Sequence_Classifier(backbone(device,args.ablation),2).to(device)
        if stage!='pretrain' and not args.no_pretrain:
            if not (out/'pretrain.pth').exists(): raise FileNotFoundError('Run pretrain first in the same --run-name')
            if classification:
                bert.load_state_dict(torch.load(out/'backbone.pth',map_location=device,weights_only=True))
                for p in bert.parameters(): p.requires_grad=False
            else:
                model.load_state_dict(torch.load(out/'pretrain.pth',map_location=device,weights_only=True))
                discriminator.load_state_dict(torch.load(out/'discriminator.pth',map_location=device,weights_only=True))
        optim=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.0005,weight_decay=.01,eps=1e-6)
        optim_dis=torch.optim.AdamW(discriminator.parameters(),lr=.0005,weight_decay=.01,eps=1e-6)
        best=float('inf'); start=time.time()
        for epoch in range(1,args.epochs+1):
            seed(args.seed+epoch); torch.set_grad_enabled(True)
            if classification:
                train_metrics=finetune.iteration(loaders[0],device,model,optim,stage,train=True)
                seed(args.seed+10000); metric=classify_eval(model,loaders[1],device,stage); score=1-metric['accuracy']
            else:
                module=pretrain if stage=='pretrain' else prediction
                kwargs={'mask_min':.15} if stage=='pretrain' and args.paper_mask else {}
                train_metrics=module.iteration(loaders[0],device,model,discriminator,optim,optim_dis,train=True,gan=not args.no_gan,**kwargs)
                seed(args.seed+10000); metric=regression_eval(model,loaders[1],device,future=stage=='prediction',ratio=meta['prediction_ratio'] if stage=='prediction' else .15); score=metric['mse']
            if not np.isfinite(score) or not np.isfinite(train_metrics).all(): raise RuntimeError('Nonfinite metrics')
            log=dict(stage=stage,epoch=epoch,train=list(map(float,train_metrics)),validation=metric,seconds=time.time()-start)
            with (out/'history.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(log)+'\n')
            print(json.dumps(log),flush=True)
            if score<best:
                best=score; torch.save(model.state_dict(),out/f'{stage}.pth')
                if stage=='pretrain':
                    torch.save(bert.state_dict(),out/'backbone.pth'); torch.save(discriminator.state_dict(),out/'discriminator.pth')
        model.load_state_dict(torch.load(out/f'{stage}.pth',map_location=device,weights_only=True)); seed(args.seed+20000)
        results[stage]=classify_eval(model,loaders[2],device,stage) if classification else regression_eval(model,loaders[2],device,future=stage=='prediction',ratio=meta['prediction_ratio'] if stage=='prediction' else .15)
        if classification:
            try:
                gap_loader=DataLoader(data('test',None,False,gap=2,dataset=args.dataset),batch_size=args.batch_size)
                results[stage]['50hz_test']=classify_eval(model,gap_loader,device,stage)
            except ValueError as e:
                results[stage]['50hz_test']={'unavailable':str(e)}
        results[stage]['epochs']=args.epochs
        (out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print('TEST '+stage+' '+json.dumps(results[stage]),flush=True)

if __name__=='__main__': main()
