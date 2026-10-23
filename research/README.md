# 科研记录

围绕扩散模型与无线感知，记录论文阅读、方法复现和研究想法。

[论文笔记](paper-notes/) · [论文原文](papers/) · [代码复现](reproduction/) · [研究想法](ideas/)

## 研究内容

| 方向 | 主要关注 | 阅读记录 |
|---|---|---|
| 生成模型基础 | 潜在空间生成、采样效率与连续表示 | [LDM](paper-notes/02-latent%20diffusion%20model.md)、[MeanFlow](paper-notes/05-meanflow.md)、[ELF](paper-notes/09-ELF.md) |
| CSI 表示与感知 | 信号恢复、未来预测、数据增强与分类 | [CLAR](paper-notes/01-CLAR.md)、[CSI-BERT2](paper-notes/10-CSI-BERT2.md) |
| 无线信道与环境重建 | 位置条件信道生成、无线电地图与图像重建 | [cDDIM](paper-notes/08-cDDIM.md)、[WiFi-Diffusion](paper-notes/04-wifi%20diffusion.md)、[LatentCSI](paper-notes/03-LatentCSI.md) |
| 目标跟踪与点云 | 未来状态预测、波束选择与毫米波点云增强 | [Echo-Conditioned DDPM](paper-notes/06-Echo-Conditioned%20Denoising%20Diffusion.md)、[mmWaveFlow](paper-notes/07-mmWaveFlow.md) |
| 定位鲁棒性 | 修复对抗扰动，减小 WiFi 指纹定位误差 | [SRLoc](paper-notes/11-SRLoc.md) |

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
