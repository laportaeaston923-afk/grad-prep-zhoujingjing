"""Train three fair MNIST experiments, then plot their real results."""
import argparse,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def run(*args):
    cmd=[sys.executable,str(HERE/args[0]),*map(str,args[1:])]
    print(">>>"," ".join(cmd),flush=True)
    subprocess.run(cmd,check=True)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--steps",type=int,default=20000)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--width",type=int,default=32)
    p.add_argument("--device",default="auto")
    args=p.parse_args()
    for mode in ("mf","imf","imf_cfg"):
        run("train.py","--mode",mode,"--steps",args.steps,"--batch-size",args.batch_size,
            "--width",args.width,"--device",args.device)
    run("plot_results.py","--device",args.device)
    print("Optional metrics: python code/evaluate.py --train-classifier")
if __name__=="__main__":main()
