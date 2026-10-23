"""Audit completed experiments and record portable evidence of the delivered bundle."""
import hashlib,json,platform
import matplotlib,numpy as np,torch,transformers
from run import ROOT

EXPECTED={
    'wicount_paper_v2':['pretrain','prediction','people'],
    'wifall_paper_v2':['pretrain','prediction','action','fall','people'],
    'wigesture_paper_v2':['pretrain','prediction','action','people'],
    'wigesture_no_gan':['pretrain','prediction','action','people'],
    'wigesture_no_arl':['pretrain','prediction','action','people'],
    'wigesture_no_time':['pretrain','prediction','action','people'],
    'wigesture_scratch':['action']}

def main():
    rows={}; total=0; hashes={}
    for name,stages in EXPECTED.items():
        out=ROOT/'runs'/name
        result=json.loads((out/'results.json').read_text()); verify=json.loads((out/'verification_report.json').read_text())
        assert verify['passed'] and set(result)==set(stages)
        history=[json.loads(x) for x in (out/'history.jsonl').read_text().splitlines()]
        for stage in stages:
            h=[x for x in history if x['stage']==stage]
            assert [x['epoch'] for x in h]==list(range(1,31)) and result[stage]['epochs']==30
            total+=len(h); path=out/f'{stage}.pth'; hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        rows[name]=dict(stages=stages,epoch_records=len(history),verification_passed=True)
        assert (out/'extended_metrics.json').exists()
    baseline=json.loads((ROOT/'runs/recurrent_baselines.json').read_text())
    assert len(baseline)==9 and all(v['epochs']==30 for v in baseline.values())
    assert json.loads((ROOT/'runs/baseline_verification.json').read_text())['passed']
    for name in baseline:
        path=ROOT/'runs'/name/'model.pth'; hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ['training_curves','forecast_comparison','packet_loss_robustness','ablations']:
        assert (ROOT/'reports'/f'{name}.png').stat().st_size>1000
    assert (ROOT/'继续复现报告.md').exists()
    report=dict(passed=True,experiments=rows,core_and_ablation_epoch_records=total,baseline_epoch_records=sum(v['epochs'] for v in baseline.values()),trained_task_models=len(hashes),environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,transformers=transformers.__version__,matplotlib=matplotlib.__version__,cuda=torch.version.cuda),checkpoint_sha256=hashes,limitations=['Original CommPre scenes/data not located','Not all paper baselines reproduced','No multi-seed statistics','Split/cropping/budget differences are documented'])
    (ROOT/'runs/completion_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS:',total,'core/ablation epochs +',sum(v['epochs'] for v in baseline.values()),'baseline epochs;',len(hashes),'trained task models')

if __name__=='__main__': main()
