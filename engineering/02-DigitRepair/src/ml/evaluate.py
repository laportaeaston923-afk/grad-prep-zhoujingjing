import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from engine import Engine,scores
from model import repair,make_masks,harmonic_fill

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device');args=parser.parse_args()
    engine=Engine(device=args.device);data=np.load(ROOT/'data'/'mnist.npz');rng=np.random.default_rng(2026)
    ids=np.concatenate([rng.choice(np.flatnonzero(data['test_labels']==k),8,replace=False) for k in range(10)])
    raw=data['test'][ids];images=torch.tensor(raw[:,None],device=engine.device,dtype=torch.float32)/127.5-1
    records=[];cases={}
    for kind in ['center','brush','half']:
        masks=make_masks(len(ids),seed=2026,kind=kind);hole=torch.tensor(masks,device=engine.device)
        observed=images*(1-hole)
        baseline=harmonic_fill(observed,hole)
        def measure(tensor):
            pixels=((tensor.detach().cpu().clamp(-1,1)+1)*127.5).round().byte().numpy()[:,0]
            errors=[];ink_errors=[];maes=[]
            for p,ref,m in zip(pixels,raw,masks[:,0]):
                s=scores(p,ref,m);errors.append(s['masked_rmse']);maes.append(s['masked_mae'])
                if s['masked_ink_rmse'] is not None:ink_errors.append(s['masked_ink_rmse'])
            return {'mean_masked_rmse':float(np.mean(errors)),'mean_masked_mae':float(np.mean(maes)),
                    'mean_masked_ink_rmse':float(np.mean(ink_errors)) if ink_errors else None}
        zero=images*(1-hole)-hole
        records.append({'mask':kind,'method':'black_fill',**measure(zero)})
        records.append({'mask':kind,'method':'harmonic',**measure(baseline)})
        for sampler,steps in [('ddim',20),('ddim',50),('ddpm',100)]:
            output,_=repair(engine.model,images,hole,steps=steps,sampler=sampler,seeds=[2026+i for i in range(len(ids))],keep_frames=False)
            r={'mask':kind,'method':sampler,'steps':steps,**measure(output)};records.append(r);print(r,flush=True)
            if sampler=='ddim' and steps==50:
                show=np.arange(0,len(ids),8)[:8]
                cases[kind]={'indices':ids[show].tolist(),'clean':raw[show].reshape(8,-1).tolist(),'mask':masks[show,0].astype(int).reshape(8,-1).tolist(),
                             'baseline':((baseline[show].cpu()+1)*127.5).round().byte().reshape(8,-1).tolist(),
                             'recovered':((output[show].cpu()+1)*127.5).round().byte().reshape(8,-1).tolist()}
    metrics={'dataset':'MNIST official test split','test_count':len(ids),'test_indices':ids.tolist(),'mask_seed':2026,'sample_seeds':[2026+i for i in range(len(ids))],'records':records,
             'notes':'One training run, one sampling seed per image. Metrics are per-image masked errors averaged across 80 test images; no best-candidate selection.'}
    (ROOT/'results'/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (ROOT/'results'/'cases.json').write_text(json.dumps(cases,separators=(',',':')),encoding='utf-8')
    # Offline reference example: explicitly labeled precomputed, not a fallback inference.
    examples=json.loads((ROOT/'src'/'web'/'examples.js').read_text().split('=',1)[1].rstrip(';'))
    ex=examples[0];mask=make_masks(1,kind='center')[0,0].astype(int).reshape(-1).tolist()
    payload={'pixels':ex['pixels'],'mask':mask,'reference':ex['pixels'],'sampler':'ddim','steps':50,'candidates':2,'seed':2026}
    result=engine.run(payload)
    (ROOT/'src'/'web'/'demo.js').write_text('window.REPAIR_DEMO='+json.dumps({'example':ex,'mask':mask,'result':result,'precomputed':True},separators=(',',':'))+';',encoding='utf-8')
