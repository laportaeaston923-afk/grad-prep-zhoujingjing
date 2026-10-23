"""Sequential GPU experiment suite; never runs competing training jobs."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=30); a=p.parse_args()
    jobs=[
        ('wicount_paper_v2',['--dataset','WiCount','--stage','all']),
        ('wifall_paper_v2',['--dataset','WiFall','--stage','all']),
        ('wigesture_paper_v2',['--dataset','WiGesture','--stage','all']),
        ('wigesture_no_gan',['--dataset','WiGesture','--stage','all','--no-gan']),
        ('wigesture_no_arl',['--dataset','WiGesture','--stage','all','--ablation','no_arl']),
        ('wigesture_no_time',['--dataset','WiGesture','--stage','all','--ablation','no_time']),
        ('wigesture_scratch',['--dataset','WiGesture','--stage','action','--no-pretrain']),
    ]
    statuses={}
    for name,extra in jobs:
        out=ROOT/'runs'/name; out.mkdir(parents=True,exist_ok=True)
        expected=['pretrain','prediction','people'] if name.startswith('wicount') else ['pretrain','prediction','action','fall','people'] if name.startswith('wifall') else ['action'] if name.endswith('scratch') else ['pretrain','prediction','action','people']
        results=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else {}
        if all(s in results and results[s]['epochs']==a.epochs for s in expected):
            print('Already complete:',name,flush=True); statuses[name]='completed'; continue
        if (out/'history.jsonl').exists(): raise RuntimeError(f'Incomplete existing run {name}: use a new run name rather than silently mixing logs')
        cmd=[sys.executable,str(ROOT/'run.py'),'--epochs',str(a.epochs),'--paper-mask','--run-name',name]+extra
        print('START',name,flush=True); start=time.time()
        with (out/'console.log').open('w',encoding='utf-8') as f:
            process=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        statuses[name]=dict(exit_code=process.returncode,seconds=time.time()-start,command=cmd)
        (ROOT/'runs/suite_status.json').write_text(json.dumps(statuses,indent=2),encoding='utf-8')
        if process.returncode: raise RuntimeError(f'{name} failed; inspect its console.log')
        print('FINISHED',name,flush=True)

if __name__=='__main__': main()
