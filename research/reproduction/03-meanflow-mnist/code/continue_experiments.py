"""Resumable staged experiments; preserves each evaluated checkpoint."""
import os,sys,subprocess,shutil,time,json,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'code';RUNS=ROOT/'runs_dit_stable'
OUT=ROOT/'results'/'new_run'
OUT.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('TORCH_HOME',str(ROOT/'work'/'torch_cache'))
os.environ['OMP_NUM_THREADS']='8'
os.environ['MKL_NUM_THREADS']='8'
os.environ['PYTHONUNBUFFERED']='1'
import torch

def status(**kwargs):
    (OUT/'run_status.json').write_text(json.dumps(dict(updated=time.strftime('%Y-%m-%d %H:%M:%S'),**kwargs),indent=2))
    print(kwargs,flush=True)

def run(script,*args):
    cmd=[sys.executable,str(CODE/script),*map(str,args)]
    print('RUN',' '.join(cmd),flush=True)
    subprocess.run(cmd,check=True,env=os.environ)

def train(mode,step):
    file=RUNS/mode/'last.pt'
    current=0
    if file.exists():
        state=torch.load(file,map_location='cpu',weights_only=False)
        current=state['step'];del state
    if current<step:
        status(state='training',mode=mode,target=step,current=current)
        args=['--mode',mode,'--steps',step,'--device','cuda']
        if current:args.append('--resume')
        run('train.py',*args)
    if current<=step:
        dest=RUNS/mode/f'step_{step:06d}.pt'
        if not dest.exists():shutil.copy2(file,dest)

def main():
    # The initial MF continuation was launched interactively. Wait without
    # creating a second writer for its checkpoint and log.
    import psutil
    while True:
        active=[]
        for proc in psutil.process_iter(['pid','name','cmdline']):
            cmd=proc.info['cmdline'] or []
            if proc.pid!=os.getpid() and any(str(CODE/'train.py').lower()==s.lower() for s in cmd):
                active.append(proc.pid)
        if not active:break
        status(state='waiting_for_initial_training',pids=active)
        time.sleep(15)
    for target in (4000,8000,12000,16000,20000):
        for mode in ('mf','imf','imf_cfg'):
            train(mode,target)
        status(state='evaluating',target=target)
        # Includes all preserved checkpoints; already completed rows are skipped.
        run('fid_curve.py','--runs-root',RUNS,'--data-root',ROOT/'data',
            '--results',OUT,'--samples',2000)
        run('plot_results.py','--runs-root',RUNS,'--results',OUT,'--device','cuda')
        for mode in ('mf','imf','imf_cfg'):
            shutil.copy2(RUNS/mode/'train.csv',OUT/f'{mode}_train.csv')
    run('evaluate.py','--train-classifier','--samples',10000,
        '--runs-root',RUNS,'--data-root',ROOT/'data','--results',OUT,'--device','cuda')
    status(state='complete',steps=20000)

if __name__=='__main__':
    try:main()
    except Exception:
        status(state='failed',traceback=traceback.format_exc())
        raise
