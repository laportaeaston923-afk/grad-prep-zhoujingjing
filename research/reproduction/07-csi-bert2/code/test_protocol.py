"""Small analytic checks for split integrity and baseline causality."""
import pickle
import numpy as np
import torch
from run import ROOT,DATASETS,data
from evaluate_suite import interpolation

def main():
    for name,tasks in DATASETS.items():
        ds={split:data(split,None,False,dataset=name) for split in ['train','valid','test']}
        assert all(len(d)>0 for d in ds.values())
        for split,d in ds.items():
            assert d.tensors[0].shape[1:]==(100,52)
            assert torch.all(torch.diff(d.tensors[-1],dim=1)>0)
            for task,count in tasks.items():
                label=d.tensors[3] if task=='people' else d.tensors[2]
                if task=='fall': label=(label!=0).long()
                assert set(label.tolist())==set(range(count)),(name,split,task)
        if name=='WiFall':
            records=pickle.load(open(ROOT/'data/WiFall/processed/data_sequence.pkl','rb'))
            seen={split:{r['source'] for r in records if r['split']==split} for split in ds}
            assert not seen['train']&seen['valid'] and not seen['train']&seen['test'] and not seen['valid']&seen['test']
        print(name,'split, timestamp and label checks passed')
    x=torch.arange(5,dtype=torch.float32).reshape(1,5,1).repeat(1,1,52)*2
    t=torch.arange(5,dtype=torch.float32).reshape(1,5)*10000
    visible=torch.tensor([[True,False,True,False,True]]); query=~visible
    y=interpolation(x,t,visible,query)['Linear_boundary_hold']
    assert torch.equal(y[query],x[query])
    causal=torch.tensor([[True,True,True,False,False]]); target=~causal
    pred=interpolation(x,t,causal,target)
    altered=x.clone(); altered[target]=10000
    other=interpolation(altered,t,causal,target)
    assert all(torch.equal(pred[k],other[k]) for k in pred)
    assert torch.equal(pred['Linear_boundary_hold'][target],torch.full((2,52),4.))
    print('Analytic interpolation and future-value independence checks passed')

if __name__=='__main__': main()
