"""Prepare official MNIST splits, optionally from an existing local raw cache."""
import argparse
import gzip
import hashlib
import json
import struct
import urllib.request
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
FILES=['train-images-idx3-ubyte','train-labels-idx1-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte']

def read_idx(path):
    raw=gzip.open(path,'rb').read() if path.suffix=='.gz' else path.read_bytes()
    magic=struct.unpack('>I',raw[:4])[0]
    if magic==2051:
        n,h,w=struct.unpack('>III',raw[4:16]);return np.frombuffer(raw[16:],dtype=np.uint8).reshape(n,h,w)
    if magic==2049:
        n=struct.unpack('>I',raw[4:8])[0];return np.frombuffer(raw[8:],dtype=np.uint8).reshape(n)
    raise ValueError('Unsupported IDX magic')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path);args=parser.parse_args()
    target=ROOT/'data';target.mkdir(exist_ok=True);arrays=[];hashes={}
    for name in FILES:
        source=args.source or target/'raw';source.mkdir(exist_ok=True,parents=True)
        path=source/name
        if not path.exists():
            path=source/(name+'.gz')
            if not path.exists():
                if args.source:raise FileNotFoundError(path)
                urllib.request.urlretrieve('https://ossci-datasets.s3.amazonaws.com/mnist/'+name+'.gz',path)
        arrays.append(read_idx(path));hashes[name]=hashlib.sha256(arrays[-1].tobytes()).hexdigest()
    np.savez_compressed(target/'mnist.npz',train=arrays[0],train_labels=arrays[1],test=arrays[2],test_labels=arrays[3])
    manifest={'dataset':'MNIST official train/test split','train_size':len(arrays[0]),'test_size':len(arrays[2]),'shape':[28,28],'array_sha256':hashes}
    (ROOT/'results'/'data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    # Official test examples, two per class; no training images in the UI.
    ids=[int(np.flatnonzero(arrays[3]==k)[j]) for k in range(10) for j in range(2)]
    examples=[{'id':int(i),'label':int(arrays[3][i]),'pixels':arrays[2][i].reshape(-1).tolist()} for i in ids]
    (ROOT/'src'/'web'/'examples.js').write_text('window.DIGIT_EXAMPLES='+json.dumps(examples,separators=(',',':'))+';',encoding='utf-8')
    print(manifest,flush=True)
