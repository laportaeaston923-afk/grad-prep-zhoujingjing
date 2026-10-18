"""Offline data preparation in a separate process (avoids mixed OpenMP runtimes)."""
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data = load_digits()
    indices = np.arange(len(data.target))
    train, test = train_test_split(indices, test_size=.2, random_state=42, stratify=data.target)
    out = Path(__file__).resolve().parents[2]/'results'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'dataset.npz', train=(data.data[train]/8-1).astype('float32'),
                        test=(data.data[test]/8-1).astype('float32'),
                        train_labels=data.target[train], test_labels=data.target[test],
                        train_indices=train, test_indices=test)
    print(f'sklearn digits: {len(train)} train / {len(test)} test, 8x8, seed 42')
