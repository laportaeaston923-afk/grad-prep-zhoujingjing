# 代码复现

| 项目 | 数据 | 当前记录 |
|---|---|---|
| [01 · DDPM](01-ddpm-cifar10/) | CIFAR-10 | 120 Epoch AMP 训练、生成图与去噪过程 |
| [02 · DDIM](02-ddim-cifar10/) | CIFAR-10 | 复用 DDPM 权重，50 步采样 |
| [03 · MeanFlow / iMF](03-meanflow-mnist/) | MNIST | MF、iMF、CFG 各 20,000 步；Loss、FID 与生成结果 |
| [04 · cDDIM](04-cddim-wireless-channel/) | QuaDRiGa | 与 One-shot U-Net 的 ID / OOD 对照 |
| [05 · CSI-BERT2](05-csi-bert2/) | WiGesture / WiFall / WiCount | 恢复、预测、分类、基线与消融；未完整达到论文指标 |

各项目的 `code/` 保存代码，`results/` 保存结果；数据和权重按项目说明准备。

[科研记录](../README.md) · [首页](../../README.md)
