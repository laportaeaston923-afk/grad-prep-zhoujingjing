# 个人简介

- 姓名：周晶晶
- 目前学习的方向：扩散模型 × 无线感知

> **研究生阶段科研记录**

[📚 论文笔记](research/paper-notes/) · [📄 论文索引](research/papers/README.md) · [💻 代码复现](research/reproduction/README.md) · [🛠 工程实践](engineering/README.md)

## 01 · 当前进度

| 模块 | 已完成的内容 | 入口 |
|---|---|---|
| 论文阅读 | 已整理 11 篇笔记，覆盖生成模型基础、无线感知与无线通信应用 | [科研总览](research/README.md) |
| 论文资料 | 已建立论文索引，保存 4 份 PDF | [论文出处](research/papers/README.md) |
| DDPM | CIFAR-10；120 Epoch AMP 训练代码、Loss 曲线图、生成样本和反向去噪过程图 | [DDPM 记录](research/reproduction/01-ddpm-cifar10/) |
| DDIM | 复用 DDPM 网络权重的 50 步采样代码与生成结果图 | [DDIM 记录](research/reproduction/02-ddim-cifar10/) |
| MeanFlow / iMF | 已改为原项目 DiT 的 MF/iMF/CFG 复现脚本，旧 U-Net 已归档；真实训练图待生成 | [iMF 复现](research/reproduction/03-meanflow-mnist/) |
| CSI-BERT2 | 三个真实数据集的恢复、预测、分类、基线与消融；未完整达到论文指标 | [CSI-BERT2 复现](research/reproduction/06-csi-bert2/) |
| 研究想法与工程实践 | 已建立目录，持续补充 | [Ideas](research/ideas/) / [Engineering](engineering/) |

## 02 · 论文阅读与研究方向

### A. 生成模型基础

| 论文 / 方法 | 关注的问题 | 学习记录 |
|---|---|---|
| **Latent Diffusion Models（LDM）** | 为什么将扩散过程从像素空间转移到潜在空间？ | [阅读笔记](research/paper-notes/02-latent%20diffusion%20model.md) |
| **MeanFlow** | 如何通过学习平均速度实现一步生成？ | [阅读笔记](research/paper-notes/05-meanflow.md) · [代码与实测](research/reproduction/03-meanflow-mnist/) |
| **DDPM / DDIM** | 如何训练噪声预测网络，以及如何替换采样方式？ | [复现总览](research/reproduction/README.md) |
| **ELF** | 如何在连续语言嵌入空间生成文本，并在最后解码？ | [阅读笔记](research/paper-notes/09-ELF.md) |

### B. 生成模型在无线感知与无线通信中的应用

| 论文 / 方法 | 输入与研究任务 | 生成模型的作用 | 阅读笔记 |
|---|---|---|---|
| **CLAR** | WiFi CSI → 人体活动识别 | 生成增强样本，辅助特征学习 | [01 · CLAR](research/paper-notes/01-CLAR.md) |
| **LatentCSI** | WiFi CSI → RGB 图像 | 将 CSI 映射到图像潜在表示 | [03 · LatentCSI](research/paper-notes/03-LatentCSI.md) |
| **WiFi-Diffusion** | 稀疏无线信号采样 → Radio Map | 生成并筛选候选无线电地图 | [04 · WiFi-Diffusion](research/paper-notes/04-wifi%20diffusion.md) |
| **Echo-Conditioned DDPM** | RF 回波 → 下一时间块目标状态 | 条件生成未来目标状态，指导感知波束选择 | [06 · Echo-Conditioned DDPM](research/paper-notes/06-Echo-Conditioned%20Denoising%20Diffusion.md) |
| **mmWaveFlow** | 稀疏毫米波点云 ↔ 稠密人体点云 | Flow Matching 统一点云增强与毫米波点云生成 | [07 · mmWaveFlow](research/paper-notes/07-mmWaveFlow.md) |
| **cDDIM** | 用户位置 + 随机噪声 → MIMO 信道 | 条件 DDIM 学习位置相关信道分布，扩充通信训练数据 | [08 · cDDIM](research/paper-notes/08-cDDIM.md) |
| **CSI-BERT2** | CSI 序列 → 恢复、预测与分类 | BERT 掩码学习无线信号规律 | [10 · CSI-BERT2](research/paper-notes/10-CSI-BERT2.md) |
| **SRLoc** | 受攻击的 RSSI → 修复指纹 → 位置 | 受扩散模型启发的单步条件重建 | [11 · SRLoc](research/paper-notes/11-SRLoc.md) |

**目前关注的问题：** 在不同无线任务里，生成模型究竟生成了什么？生成的数据或状态又怎样帮助最终任务？

完整的论文原文与来源见 [论文索引](research/papers/README.md)。

## 03 · 代码复现：生成模型与无线感知

图像生成实验用于理解训练目标、采样过程和评估方法，再逐步联系无线感知任务。各项目分别保存代码与结果。

| 实验 | 主线 | 当前结果 |
|---|---|---|
| [DDPM · CIFAR-10](research/reproduction/01-ddpm-cifar10/) | 训练 U-Net 预测噪声，再进行反向采样 | 采用 `train_full_120_amp.py` 对应的完整 120 Epoch AMP 实验；已展示真实 Loss、生成图和去噪过程图 |
| [DDIM · CIFAR-10](research/reproduction/02-ddim-cifar10/) | 复用 DDPM 权重，改变采样方法 | 已展示 50 步采样代码和结果；与 DDPM 的统一设备耗时、FID 对照仍待补齐 |
| [MF / iMF · MNIST](research/reproduction/03-meanflow-mnist/) | 基于原项目 DiT，对比 JVP 方向、CFG 和一步生成 | 原项目 MFDiT 代码已更新稳定性设置与 JVP 诊断；新版实际训练结果待执行 |

| [06 · CSI-BERT2](research/reproduction/06-csi-bert2/) | 掩码预训练、CSI 恢复、末尾预测与感知分类 | WiGesture / WiFall / WiCount；实测指标、消融、基线和效果图已保存，未完整达到论文指标 |

DDPM 基础实现参考 [Bilibili 教学视频](https://www.bilibili.com/video/BV1vZi6BaEgF/)。MeanFlow 使用 [haidog-yaqub/MeanFlow](https://github.com/haidog-yaqub/MeanFlow) 的第三方实现，固定版本与许可记录在项目目录；当前结果属于 MNIST 实验，不声明达到原论文 ImageNet 成绩。

## 04 · 仓库目录与快速导航

| 路径 | 内容 |
|---|---|
| [research/](research/README.md) | 论文学习、代码复现与研究任务总览 |
| [research/paper-notes/](research/paper-notes/) | 11 篇论文笔记与阅读索引 |
| [research/papers/](research/papers/README.md) | 论文索引、官方出处与 4 份 PDF |
| [research/reproduction/](research/reproduction/README.md) | DDPM、DDIM、MeanFlow/iMF 等独立实验的统一总结 |
| [01-ddpm-cifar10/](research/reproduction/01-ddpm-cifar10/) | DDPM 训练代码与结果展示 |
| [02-ddim-cifar10/](research/reproduction/02-ddim-cifar10/) | DDIM 采样代码与结果展示 |
| [03-meanflow-mnist/](research/reproduction/03-meanflow-mnist/) | MeanFlow/iMF 训练代码、Loss/CFG/一步生成可视化与 MNIST 特征距离评估（待正式运行） |
| [04-cddim-wireless-channel/](research/reproduction/04-cddim-wireless-channel/) | 无线信道生成复现 |
| [05-widar3-csi-dfs-bvp/](research/reproduction/05-widar3-csi-dfs-bvp/) | CSI → DFS → BVP 复现 |
| [06-csi-bert2/](research/reproduction/06-csi-bert2/) | CSI 恢复、预测、分类及实测结果 |
| [research/ideas/](research/ideas/) | 研究想法，持续补充 |
| [engineering/](engineering/) | 工程实践，持续补充 |
