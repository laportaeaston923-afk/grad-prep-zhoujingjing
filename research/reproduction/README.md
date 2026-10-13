# 代码复现

每个实验按 `README.md`、`code/`、`results/` 整理。

| 实验 | 数据集 | 主要内容 |
|---|---|---|
| [01 · DDPM](01-ddpm-cifar10/) | CIFAR-10 | 训练、生成图像与去噪过程 |
| [02 · DDIM](02-ddim-cifar10/) | CIFAR-10 | 复用 DDPM 权重，进行 50 步采样 |
| [03 · MeanFlow](03-meanflow-mnist/) | MNIST | 一步 / 五步生成、Loss、CFG 和 FID |
| [04 · cDDIM](04-cddim-wireless-channel/) | QuaDRiGa | 无线信道生成，Fig. 5/6/7，ID/OOD 与 One-shot U-Net 对照 |
| [05 · Widar3.0](05-widar3-csi-dfs-bvp/) | 官方 CSI 示例 | CSI → DFS → BVP，代码、实验结果与学习笔记 |

各项目的 `results/` 放结果图，`code/` 放对应代码。大体积训练数据和模型权重按各项目 README 说明自行准备。

[← 返回科研记录](../README.md)
