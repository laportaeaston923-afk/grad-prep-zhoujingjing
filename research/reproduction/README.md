# 代码复现

每个实验单独一个目录，记录环境配置、运行方法和实验结果。

## 实验列表

| 实验 | 数据集 | 主要内容 |
|---|---|---|
| [01 · DDPM](01-ddpm-cifar10/) | CIFAR-10 | 120 Epoch AMP 训练、Loss 曲线、生成图片与去噪过程 |
| [02 · DDIM](02-ddim-cifar10/) | CIFAR-10 | 复用 DDPM 权重，进行 50 步采样 |
| [03 · MeanFlow](03-meanflow-mnist/) | MNIST | 50,000 step 训练记录、一步/五步生成、CFG 与 FID 对照 |

## 目录结构

```text
reproduction/
├── README.md                     # 复现总览
├── 01-ddpm-cifar10/
│   ├── README.md                 # 实验说明与结果图
│   ├── train_full_120_amp.py     # 训练代码
│   ├── sample.py                # 采样代码
│   └── ddpm.py、unet.py 等       # 模型代码
├── 02-ddim-cifar10/
│   ├── README.md                 # 实验说明与结果图
│   ├── sample.py                # 采样代码
│   └── ddpm.py、unet.py 等       # 模型代码
└── 03-meanflow-mnist/
    ├── README.md                 # 配置、运行说明与结果
    ├── configs/                  # 实验配置
    ├── models/                   # 模型代码
    ├── train_repro.py 等         # 训练、采样与评估代码
    └── results/                  # 日志、Loss、指标、图片与动图
```

详细配置、运行命令和实验结果见各项目 README。DDPM/DDIM 的结果图展示在 README 中，MeanFlow 的结果文件保存在 `results/`。

[← 返回科研记录](../README.md)
