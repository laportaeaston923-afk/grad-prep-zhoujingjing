"""Reload every recurrent baseline and reproduce its recorded test metrics."""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from run import ROOT,data
from train_baselines import Forecaster,iteration

def main():
    torch.set_num_threads(4)
    summary=json.loads((ROOT/'runs/recurrent_baselines.json').read_text()); cache={}; checks={}
    for name,recorded in summary.items():
        dataset=recorded['dataset']
        if dataset not in cache: cache[dataset]=DataLoader(data('test',None,False,dataset=dataset),batch_size=64)
        model=Forecaster(recorded['model']); state=torch.load(ROOT/'runs'/name/'model.pth',map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in state.values())
        model.load_state_dict(state); actual=iteration(model,cache[dataset],torch.device('cpu'))
        for metric in ['mse','nmse','mape','smape']:
            assert np.isclose(actual[metric],recorded[metric],rtol=1e-5,atol=1e-6),(name,metric,actual[metric],recorded[metric])
        history=json.loads((ROOT/'runs'/name/'history.json').read_text())
        assert [r['epoch'] for r in history]==list(range(1,recorded['epochs']+1))
        checks[name]=dict(checkpoint_finite=True,recorded_test_metrics_reproduced=True,epochs=recorded['epochs'])
    report=dict(passed=True,checks=checks)
    (ROOT/'runs/baseline_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
