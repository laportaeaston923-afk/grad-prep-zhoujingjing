"""Train all three DiT experiments, then generate the requested four diagrams.

Caution: this is real training, not a pre-computed or simulated graph.
"""
import argparse,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def launch(script,*args):
    cmd=[sys.executable,str(HERE/script),*map(str,args)]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd,check=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--steps",type=int,default=20000)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--dim",type=int,default=256)
    p.add_argument("--depth",type=int,default=6)
    p.add_argument("--heads",type=int,default=8)
    p.add_argument("--patch-size",type=int,default=4)
    p.add_argument("--device",default="auto")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    for mode in ("mf","imf","imf_cfg"):
        arguments=["--mode",mode,"--steps",args.steps,"--batch-size",args.batch_size,
                   "--dim",args.dim,"--depth",args.depth,"--heads",args.heads,
                   "--patch-size",args.patch_size,"--device",args.device]
        if args.resume:arguments.append("--resume")
        launch("train.py",*arguments)
    launch("plot_results.py","--device",args.device)
    print("For feature metrics, run: python code/evaluate.py --train-classifier")

if __name__=="__main__":
    main()
