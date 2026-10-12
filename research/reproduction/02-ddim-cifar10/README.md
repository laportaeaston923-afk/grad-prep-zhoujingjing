# DDIM · CIFAR-10

复用 DDPM 的噪声预测网络权重，改为 50 步 DDIM 采样，不单独训练模型。

## 目录

- `code/`：模型与采样代码。
- `results/`：生成图片和详细实验记录。

## 环境与运行

需要 Python、PyTorch、torchvision、matplotlib、tqdm；可使用 CUDA GPU。

```bat
cd code
python -m pip install -r requirements.txt
python sample.py --checkpoint ../../01-ddpm-cifar10/results/ddpm.pth
```

`--checkpoint` 应指向与 DDPM 对照使用的同一份权重。新采样图保存到 `results/samples_latest.png`；加 `--show` 可同时显示图片。

## 实验结果

[查看 results](results/)：50 步采样的生成结果图。采样时间和质量对照需统一权重、设备与设置；当前展示图不单独证明实际加速幅度。
