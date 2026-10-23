"""Apply trained CSI-BERT2 weights to npz CSI arrays, with no network access."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from run import ROOT, DATASETS, backbone, Token_Classifier, Sequence_Classifier, seed

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True,help='npz containing magnitude [N,100,52] and timestamp [N,100]')
    p.add_argument('--output',required=True)
    p.add_argument('--task',choices=['recovery','prediction','action','fall','people'],default='prediction')
    p.add_argument('--run-name',default='wigesture_paper_v2')
    p.add_argument('--ratio',type=float,default=None)
    p.add_argument('--cpu',action='store_true')
    a=p.parse_args(); seed(42); torch.set_num_threads(4)
    config=json.loads((ROOT/'runs'/a.run_name/'config.json').read_text())
    dataset=config['arguments'].get('dataset','WiGesture')
    if a.ratio is None: a.ratio=config.get('prediction_ratio',.15)
    if not 0<a.ratio<1: p.error('--ratio must be between 0 and 1')
    z=np.load(a.input,allow_pickle=False); x=np.asarray(z['magnitude'],dtype=np.float32); t=np.asarray(z['timestamp'],dtype=np.float32)
    if x.ndim!=3 or x.shape[1:]!=(100,52) or t.shape!=x.shape[:2]: p.error('Expected magnitude [N,100,52] and timestamp [N,100]')
    if not np.isfinite(t).all() or np.any(np.diff(t,axis=1)<=0): p.error('Timestamps must be finite and strictly increasing')
    if not np.isfinite(x).all(): p.error('Use -1000 to mark missing CSI values')
    d=torch.device('cpu' if a.cpu or not torch.cuda.is_available() else 'cuda:0'); bert=backbone(d,config['arguments'].get('ablation','none'))
    classification=a.task in ['action','fall','people']
    if classification and a.task not in DATASETS[dataset]: p.error('Task unavailable for this dataset')
    model=Sequence_Classifier(bert,DATASETS[dataset][a.task]).to(d) if classification else Token_Classifier(bert,52).to(d)
    name='pretrain' if a.task=='recovery' else a.task
    model.load_state_dict(torch.load(ROOT/'runs'/a.run_name/f'{name}.pth',map_location=d,weights_only=True)); model.eval()
    outputs=[]
    with torch.no_grad():
        for start in range(0,len(x),64):
            value=torch.tensor(x[start:start+64],device=d); ts=torch.tensor(t[start:start+64]-t[start:start+64,:1],device=d)
            observed=value!=-1000
            if a.task=='prediction': observed[:,-max(1,round(100*a.ratio)):]=False
            if not observed.any(1).all(): raise ValueError('Every subcarrier needs at least one observed packet')
            count=observed.sum(1,keepdim=True).clamp_min(1)
            mean=(value*observed).sum(1,keepdim=True)/count
            std=(((value-mean).square()*observed).sum(1,keepdim=True)/count).sqrt()
            v=(value-mean)/(std+1e-8); noise=torch.randn_like(v); v[~observed]=noise[~observed]
            out=model(v,ts)
            if classification: out=out.softmax(-1)
            else:
                out=out*std+mean
                out=torch.where(observed,value,out)
            outputs.append(out.cpu().numpy())
    result=np.concatenate(outputs)
    target=Path(a.output); target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,**({'probabilities':result,'labels':result.argmax(-1)} if classification else {'magnitude':result,'timestamp':t}))
    print(target.resolve())

if __name__=='__main__': main()
