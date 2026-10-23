"""Inspect the public backbone without executing its downloaded model.py."""
import hashlib,json
import torch
from safetensors.torch import load_file
from run import ROOT,backbone,data,seed

def main():
    torch.set_num_threads(4); seed(42)
    path=ROOT/'author_checkpoint/model.safetensors'
    original=load_file(str(path),device='cpu')
    assert all(k.startswith('model.') for k in original)
    state={k[len('model.'):]:v for k,v in original.items()}
    assert all(torch.isfinite(v).all() for v in state.values())
    model=backbone(torch.device('cpu')); model.load_state_dict(state,strict=True); model.eval()
    sample=data('test',None,False)[0]; x=sample[0].unsqueeze(0); t=sample[-1].unsqueeze(0)
    observed=x!=-1000; count=observed.sum(1,keepdim=True).clamp_min(1)
    avg=(x*observed).sum(1,keepdim=True)/count
    std=(((x-avg).square()*observed).sum(1,keepdim=True)/count).sqrt()
    x=(x-avg)/(std+1e-8); x[~observed]=0
    with torch.no_grad(): y=model(x,t)
    assert y.shape==(1,100,128) and torch.isfinite(y).all()
    report=dict(source='https://huggingface.co/RS2002/CSI-BERT2',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),strict_backbone_state_dict_match=True,tensor_count=len(state),feature_shape=list(y.shape),finite_output=True,recovery_head_included=False,prediction_head_included=False,classification_heads_included=False,remote_code_executed=False,note='Published checkpoint contains the feature backbone only. It cannot directly reproduce task errors without fitting task heads. Its exact training/test split is not established here; it is not used as an independent held-out performance claim.')
    (ROOT/'author_checkpoint/inspection.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
