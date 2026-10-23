# 科研记录

围绕扩散模型与无线感知，记录论文阅读、方法复现和研究想法。

[论文笔记](paper-notes/) · [论文原文](papers/) · [代码复现](reproduction/) · [研究想法](ideas/)

## 论文阅读

### 1. 扩散模型

主要学习如何生成数据，以及如何降低训练和采样开销。包括扩散模型与相关的 Flow Matching 方法。

| 论文 / 方法 | 发表出处 | 解决什么问题 |
|---|---|---|
| [LDM](paper-notes/02-latent%20diffusion%20model.md) | CVPR 2022 | 像素空间扩散计算量大，将生成过程移到潜在空间，降低高分辨率图像生成的开销。 |
| [MeanFlow](paper-notes/05-meanflow.md) | NeurIPS 2025（Oral） | 多步采样推理成本高，通过学习平均速度实现一步生成。 |
| [ELF](paper-notes/09-ELF.md) | arXiv 技术报告，2026 | 连续语言生成需要衔接向量与离散 Token，在连续嵌入空间生成，最后使用共享网络解码成文字。 |

DDPM（NeurIPS 2020）的噪声预测与 DDIM（ICLR 2021）的加速采样结合 [CIFAR-10 复现](reproduction/) 学习。

### 2. 扩散模型与无线感知的交叉论文

重点关注无线数据不足、观测稀疏、信号受扰动等问题，以及生成或修复的数据能否帮助最终任务。这里也收录相关的无线通信与 Flow Matching 工作。

| 论文 | 发表出处 | 解决什么问题 |
|---|---|---|
| [CLAR](paper-notes/01-CLAR.md) | IEEE Internet of Things Journal, 2024 | CSI 标注数据有限，利用扩散模型生成增强样本，辅助人体活动识别。 |
| [LatentCSI](paper-notes/03-LatentCSI.md) | IEEE GLOBECOM Workshops 2025 | WiFi CSI 难以直观呈现场景，利用预训练潜在扩散模型从 CSI 生成 RGB 图像。 |
| [WiFi-Diffusion](paper-notes/04-wifi%20diffusion.md) | IEEE Journal on Selected Areas in Communications, 2025 | 无线电地图密集采集成本高，从少量无线采样估计细粒度 Radio Map。 |
| [Echo-Conditioned DDPM](paper-notes/06-Echo-Conditioned%20Denoising%20Diffusion.md) | arXiv preprint, 2025 | 根据 RF 回波预测未来目标状态，为多目标跟踪和感知波束选择提供依据。 |
| [mmWaveFlow](paper-notes/07-mmWaveFlow.md) | CVPR 2026 | 毫米波人体点云稀疏、数据有限，用 Flow Matching 统一点云增强与生成。 |
| [cDDIM](paper-notes/08-cDDIM.md) | IEEE Transactions on Wireless Communications, 2026（online 2025） | 高维 MIMO 信道数据采集成本高，根据用户位置生成信道样本，扩充通信训练数据。 |
| [SRLoc](paper-notes/11-SRLoc.md) | IEEE Transactions on Mobile Computing，Early Access，2026 | RSSI 对抗扰动会增大定位误差，借鉴扩散条件去噪思想，单步修复信号后再定位。 |

这些工作分别用生成模型补充训练数据、重建无线环境、预测未来状态或修复受损信号，最终仍需通过识别、定位、跟踪或通信任务评价效果。SRLoc 属于受扩散启发的重建方法。

相关基础阅读：[CSI-BERT2](paper-notes/10-CSI-BERT2.md) 发表于 IEEE Transactions on Mobile Computing（2025 年在线发表），使用 BERT 掩码学习完成 CSI 恢复、预测与分类，属于无线感知方法，不使用扩散模型。

## 复现进展

| 项目 | 已有记录 |
|---|---|
| [DDPM / DDIM · CIFAR-10](reproduction/) | DDPM 120 Epoch AMP 训练；DDIM 复用权重进行 50 步采样 |
| [MeanFlow / iMF · MNIST](reproduction/03-meanflow-mnist/) | MF、iMF、CFG 各 20,000 步，已记录 Loss、FID 和生成图 |
| [cDDIM · 无线信道](reproduction/04-cddim-wireless-channel/) | 与 One-shot U-Net 比较 ID / OOD 主波束匹配 |
| [Widar3.0 · 信号处理](reproduction/05-widar3-csi-dfs-bvp/) | 跑通单样本 CSI → DFS → BVP，保存结果与学习笔记 |
| [CSI-BERT2](reproduction/06-csi-bert2/) | 三个实测数据集的恢复、预测、分类、基线与消融；未完整达到论文指标 |

具体配置、运行命令与结果见各项目。

## 待验证的问题

| 问题 | 对应想法 |
|---|---|
| 增强的数据能否保留感知任务需要的信息？ | [CSI 数据增强](ideas/02-task-aware-csi-augmentation.md) |
| 能否减少位置条件信道生成的采样开销？ | [快速信道生成](ideas/03-fast-position-conditioned-channel.md) |
| 能否用生成不确定性减少无线电地图的采样量？ | [主动采样](ideas/04-active-radio-map-sampling.md) |

其余想法涉及 [CSI 图像生成](ideas/05-csi-image-faithfulness.md)、[波束状态预测](ideas/06-efficient-beam-state-prediction.md) 和 [时序点云增强](ideas/07-temporal-point-cloud-enhancement.md)，均待实验验证。

[返回首页](../README.md)
