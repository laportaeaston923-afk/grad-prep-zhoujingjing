"""Plot true One-shot U-Net training MSE; do not label it cDDIM noise loss."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--input',default='outputs/oneshot/loss.csv')
p.add_argument('--output',default='outputs/oneshot/loss.png')
a=p.parse_args()
with Path(a.input).open(encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
if not rows: raise RuntimeError('One-shot training loss CSV is empty')
x=[int(r['epoch']) for r in rows]
y=[float(r['mean_train_loss']) for r in rows]
fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
ax.plot(x,y)
ax.set(xlabel='Epoch',ylabel='Mean training MSE',title='One-shot U-Net: direct channel prediction loss')
ax.grid(alpha=.25)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,dpi=160)
plt.close(fig)
print('Saved',out.resolve())
