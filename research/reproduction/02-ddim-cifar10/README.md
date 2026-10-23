# DDIM · CIFAR-10

复用 DDPM 的噪声预测网络，进行 50 步采样，无需单独训练。

[代码](code/) · [结果](results/) · [实验配置](results/EXPERIMENT_GUIDE.md)

## 运行

需要 Python、PyTorch、torchvision、matplotlib、tqdm。进入本项目的 `code/`：

```bat
python -m pip install -r requirements.txt
python sample.py --checkpoint ../../01-ddpm-cifar10/results/ddpm.pth
```

与 DDPM 比较时使用同一权重。新图保存为 `results/samples_latest.png`，可加 `--show` 显示。统一设备下的耗时与 FID 对照尚未记录。

[返回复现目录](../README.md)
