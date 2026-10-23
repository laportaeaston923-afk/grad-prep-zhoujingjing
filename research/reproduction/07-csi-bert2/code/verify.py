"""Check saved training evidence and exercise all four inference paths."""
import json, sys
from pathlib import Path
import numpy as np
import torch
from run import ROOT,data
import infer

def main():
    name=sys.argv[1] if len(sys.argv)>1 else 'wigesture_paper_v2'
    out=ROOT/'runs'/name
    results=json.loads((out/'results.json').read_text())
    config=json.loads((out/'config.json').read_text())
    history=[json.loads(x) for x in (out/'history.jsonl').read_text().splitlines()]
    stages=list(results)
    checks={}
    for stage in stages:
        rows=[x for x in history if x['stage']==stage]
        assert [x['epoch'] for x in rows]==list(range(1,config['arguments']['epochs']+1)),stage
        assert all(np.isfinite(x['train']).all() for x in rows),stage
        state=torch.load(out/f'{stage}.pth',map_location='cpu',weights_only=True)
        assert all(torch.isfinite(x).all() for x in state.values()),stage
        assert np.isfinite(results[stage].get('mse',results[stage].get('accuracy'))),stage
        checks[stage]={'epochs':len(rows),'checkpoint_finite':True}
    demo=out/'inference_examples'; demo.mkdir(exist_ok=True)
    ds=data('test',None,False,dataset=config['arguments'].get('dataset','WiGesture'))
    original=ds.tensors[0][:1].numpy(); timestamps=ds.tensors[-1][:1].numpy()
    source=demo/'input.npz'; np.savez_compressed(source,magnitude=original,timestamp=timestamps)
    for task in ['recovery' if s=='pretrain' else s for s in stages]:
        target=demo/f'{task}.npz'
        sys.argv=['infer.py','--input',str(source),'--output',str(target),'--task',task,'--run-name',name]
        infer.main()
        z=np.load(target)
        if task in ['action','fall','people']:
            probabilities=z['probabilities']
            assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(-1),1,atol=1e-5)
        else:
            y=z['magnitude']; assert y.shape==original.shape and np.isfinite(y).all()
            observed=original!=-1000
            if task=='prediction': observed[:,-round(100*config.get('prediction_ratio',.15)):]=False
            assert np.array_equal(y[observed],original[observed]),'Observed input was changed'
        checks.setdefault(task,{})['inference_passed']=True
        if task=='prediction':
            ratio=config.get('prediction_ratio',.15); tail=round(100*ratio)
            perturbed=original.copy(); perturbed[:,-tail:]+=10000
            modified=demo/'input_modified_future.npz'; modified_out=demo/'prediction_modified_future.npz'
            np.savez_compressed(modified,magnitude=perturbed,timestamp=timestamps)
            sys.argv=['infer.py','--input',str(modified),'--output',str(modified_out),'--task','prediction','--run-name',name]
            infer.main()
            assert np.array_equal(np.load(modified_out)['magnitude'],z['magnitude']),'Forecast leaks future target values'
            checks[task]['forecast_target_leakage_check']=True
    report=dict(passed=True,run=name,checks=checks)
    (out/'verification_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
