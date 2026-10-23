# 代码复现

每个实验按 `README.md`、`code/`、`results/` 整理。

| 实验 | 数据集 | 主要内容 |
|---|---|---|
| [01 · DDPM](01-ddpm-cifar10/) | CIFAR-10 | 训练、生成图像与去噪过程 |
| [02 · DDIM](02-ddim-cifar10/) | CIFAR-10 | 复用 DDPM 权重，进行 50 步采样 |
| [03 · MF vs iMF（原项目 DiT）](03-meanflow-mnist/) | MNIST | 原项目 MFDiT；已新增 Fig.3 数值稳定性排查、LR 预热、JVP 诊断；四图待 GPU 真实训练 |
| [04 · cDDIM](04-cddim-wireless-channel/) | QuaDRiGa | 无线信道生成，Fig. 5/6/7，ID/OOD 与 One-shot U-Net 对照 |
| [05 · Widar3.0](05-widar3-csi-dfs-bvp/) | 官方 CSI 示例 | CSI → DFS → BVP，代码、实验结果与学习笔记 |
| [06 · GRWM 世界模型](06-cloning-deterministic-worlds/) | Maze9 | CVPR 2026；官方 VAE/GRWM + DFOT 扩散预测；单卡运行脚本与笔记已提交，真实训练待完成 |

| [07 · CSI-BERT2](07-csi-bert2/) | WiGesture / WiFall / WiCount | CSI 恢复、预测、分类，基线与消融；核心实验已跑通，未完整达到论文指标 |

各项目的 `results/` 放结果图，`code/` 放对应代码。大体积训练数据和模型权重按各项目 README 说明自行准备。

[← 返回科研记录](../README.md)
