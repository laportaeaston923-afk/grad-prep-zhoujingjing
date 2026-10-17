# 个人简介

- 姓名：周晶晶
- 目前学习的方向：生成模型 × 无线感知

> **研究生阶段科研记录**

[📚 论文笔记](research/paper-notes/) · [📄 论文索引](research/papers/README.md) · [💻 代码复现](research/reproduction/README.md) · [🛠 工程实践](engineering/README.md)

## 01 · 当前进度

| 模块 | 已完成的内容 | 入口 |
|---|---|---|
| 论文阅读 | 已整理 6 篇笔记，覆盖生成模型基础与无线感知应用 | [科研总览](research/README.md) |
| 论文资料 | 已建立论文索引，保存 4 份 PDF | [论文出处](research/papers/README.md) |
| DDPM | CIFAR-10；120 Epoch AMP 训练代码、Loss 曲线图、生成样本和反向去噪过程图 | [DDPM 记录](research/reproduction/01-ddpm-cifar10/) |
| DDIM | 复用 DDPM 网络权重的 50 步采样代码与生成结果图 | [DDIM 记录](research/reproduction/02-ddim-cifar10/) |
| MeanFlow / iMF | 已改为原项目 DiT 的 MF/iMF/CFG 复现脚本，旧 U-Net 已归档；真实训练图待生成 | [iMF 复现](research/reproduction/03-meanflow-mnist/) |
| 研究想法与工程实践 | 已建立目录，持续补充 | [Ideas](research/ideas/) / [Engineering](engineering/) |

## 02 · 论文阅读与研究方向

### A. 生成模型基础

| 论文 / 方法 | 关注的问题 | 学习记录 |
|---|---|---|
| **Latent Diffusion Models（LDM）** | 为什么将扩散过程从像素空间转移到潜在空间？ | [阅读笔记](research/paper-notes/02-latent%20diffusion%20model.md) |
| **MeanFlow** | 如何通过学习平均速度实现一步生成？ | [阅读笔记](research/paper-notes/05-meanflow.md) · [代码与实测](research/reproduction/03-meanflow-mnist/) |
| **DDPM / DDIM** | 如何训练噪声预测网络，以及如何替换采样方式？ | [复现总览](research/reproduction/README.md) |

### B. 扩散模型在无线感知中的应用

| 论文 / 方法 | 输入与研究任务 | 生成模型的作用 | 阅读笔记 |
|---|---|---|---|
| **CLAR** | WiFi CSI → 人体活动识别 | 生成增强样本，辅助特征学习 | [01 · CLAR](research/paper-notes/01-CLAR.md) |
| **LatentCSI** | WiFi CSI → RGB 图像 | 将 CSI 映射到图像潜在表示 | [03 · LatentCSI](research/paper-notes/03-LatentCSI.md) |
| **WiFi-Diffusion** | 稀疏无线信号采样 → Radio Map | 生成并筛选候选无线电地图 | [04 · WiFi-Diffusion](research/paper-notes/04-wifi%20diffusion.md) |
| **Echo-Conditioned DDPM** | RF 回波 → 下一时间块目标状态 | 条件生成未来目标状态，指导感知波束选择 | [06 · Echo-Conditioned DDPM](research/paper-notes/06-Echo-Conditioned%20Denoising%20Diffusion.md) |

**目前关注的问题：** 在不同无线感知任务里，生成模型究竟生成了什么？生成的数据或状态又怎样帮助最终任务？

完整的论文原文与来源见 [论文索引](research/papers/README.md)。

## 03 · 代码复现：DDPM、DDIM、MeanFlow 与世界模型

图像生成实验用于理解训练目标、采样过程和评估方法，再逐步联系无线感知任务。三个项目分别保存代码与结果。

| 实验 | 主线 | 当前结果 |
|---|---|---|
| [DDPM · CIFAR-10](research/reproduction/01-ddpm-cifar10/) | 训练 U-Net 预测噪声，再进行反向采样 | 采用 `train_full_120_amp.py` 对应的完整 120 Epoch AMP 实验；已展示真实 Loss、生成图和去噪过程图 |
| [DDIM · CIFAR-10](research/reproduction/02-ddim-cifar10/) | 复用 DDPM 权重，改变采样方法 | 已展示 50 步采样代码和结果；与 DDPM 的统一设备耗时、FID 对照仍待补齐 |
| [MF / iMF · MNIST](research/reproduction/03-meanflow-mnist/) | 基于原项目 DiT，对比 JVP 方向、CFG 和一步生成 | 原项目 MFDiT 代码已更新稳定性设置与 JVP 诊断；新版实际训练结果待执行 |
| [GRWM 世界模型 · Maze9（CVPR 2026）](research/reproduction/06-cloning-deterministic-worlds/) | 对比 VAE 与 GRWM 状态表示，用 DFOT 扩散预测未来帧 | 原作者固定版本拉取、单卡串行脚本、中文笔记与测试已提交；尚未完成真实 GPU 训练 |


DDPM 基础实现参考 [Bilibili 教学视频](https://www.bilibili.com/video/BV1vZi6BaEgF/)。MeanFlow 使用 [haidog-yaqub/MeanFlow](https://github.com/haidog-yaqub/MeanFlow) 的第三方实现，固定版本与许可记录在项目目录；当前结果属于 MNIST 实验，不声明达到原论文 ImageNet 成绩。

## 04 · 仓库目录与快速导航

| 路径 | 内容 |
|---|---|
| [research/](research/README.md) | 论文学习、代码复现与研究任务总览 |
| [research/paper-notes/](research/paper-notes/) | 6 篇论文笔记 |
| [research/papers/](research/papers/README.md) | 论文索引、官方出处与 4 份 PDF |
| [research/reproduction/](research/reproduction/README.md) | DDPM、DDIM、MeanFlow/iMF 等独立实验的统一总结 |
| [01-ddpm-cifar10/](research/reproduction/01-ddpm-cifar10/) | DDPM 训练代码与结果展示 |
| [02-ddim-cifar10/](research/reproduction/02-ddim-cifar10/) | DDIM 采样代码与结果展示 |
| [03-meanflow-mnist/](research/reproduction/03-meanflow-mnist/) | MeanFlow/iMF 训练代码、Loss/CFG/一步生成可视化与 MNIST 特征距离评估（待正式运行） |
| [06-cloning-deterministic-worlds/](research/reproduction/06-cloning-deterministic-worlds/) | CVPR 2026 世界模型：VAE vs GRWM 的官方 Maze9 复现流程和资料 |
| [research/ideas/](research/ideas/) | 研究想法，持续补充 |
| [engineering/](engineering/) | 工程实践，持续补充 |
