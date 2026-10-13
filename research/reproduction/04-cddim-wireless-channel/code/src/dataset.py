"""Load official QuaDRiGa .mat, or small labelled demo channels for a CPU smoke test."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset
import torch


def beamspace(channels):
    """Matches official code's fft2 + fftshift, preserving Rx/Tx array axes."""
    return np.fft.fftshift(np.fft.fft2(channels,axes=(-2,-1)),axes=(-2,-1))


def to_model_array(channels):
    h = beamspace(np.asarray(channels))
    amplitude = np.max(np.abs(h),axis=(1,2),keepdims=True)
    h = h / np.maximum(amplitude,1e-8)
    return np.stack([h.real,h.imag],axis=1).astype('float32')


def load_pairs(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'信道数据不存在: {path}. 请下载作者提供的 QuaDRiGa 数据，或先运行 smoke_test.bat')
    if path.suffix.lower() == '.npz':
        with np.load(path) as d:
            h=d['H_set']; pos=d['rx_positions']
    else:
        try:
            d=loadmat(path)
            h=d['H_set']; pos=d['rx_positions']
        except NotImplementedError as e:
            raise RuntimeError('这是 MATLAB v7.3 文件，请用 h5py 读取后转换成 NPZ（键 H_set/rx_positions）。') from e
    if h.ndim != 3 or h.shape[1:] != (4,32):
        if h.ndim == 3 and h.shape[:2] == (4,32):
            h=np.moveaxis(h,-1,0)
        else:
            raise ValueError(f'H_set shape 应为 [N,4,32]，实际为 {h.shape}')
    if pos.ndim!=2 or pos.shape[-1]!=3:
        if pos.shape[0]==3: pos=pos.T
        else: raise ValueError(f'rx_positions shape 应为 [N,3]，实际为 {pos.shape}')
    if len(h)!=len(pos): raise ValueError('信道矩阵数量和位置数量不匹配')
    return h.astype('complex64'),pos.astype('float32')


class ChannelDataset(Dataset):
    def __init__(self,channels,positions,position_scale=100.0):
        if len(channels)!=len(positions):raise ValueError('channels and positions length mismatch')
        self.data=torch.from_numpy(to_model_array(channels))
        self.positions=torch.from_numpy(np.asarray(positions,dtype='float32')/position_scale)
    def __len__(self): return len(self.data)
    def __getitem__(self,index): return self.data[index], self.positions[index]
