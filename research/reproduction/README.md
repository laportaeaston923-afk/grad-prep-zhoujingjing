# 代码复现

每个实验统一分为 `README.md`、`code/` 和 `results/`。

| 实验 | 数据集 | 主要内容 |
|---|---|---|
| [01 · DDPM](01-ddpm-cifar10/) | CIFAR-10 | 训练、生成图片与去噪过程 |
| [02 · DDIM](02-ddim-cifar10/) | CIFAR-10 | 复用 DDPM 权重，进行 50 步采样 |
| [03 · MeanFlow](03-meanflow-mnist/) | MNIST | 一步/五步生成、Loss、CFG 与 FID 对照 |

```text
reproduction/
├── README.md
├── 01-ddpm-cifar10/
│   ├── README.md     # 环境配置与运行说明
│   ├── code/         # 代码
│   └── results/      # 实验结果
├── 02-ddim-cifar10/
│   ├── README.md
│   ├── code/
│   └── results/
└── 03-meanflow-mnist/
    ├── README.md
    ├── code/
    └── results/
```

进入各项目的 `code/` 目录运行。图片、曲线、日志和指标保存在 `results/`，模型权重需自行准备。

[← 返回科研记录](../README.md)
