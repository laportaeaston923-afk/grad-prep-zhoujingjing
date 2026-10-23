# grad-prep-zhoujingjing

研究生科研记录：论文阅读、代码复现与工程实践。

## 个人简介

- 姓名：周晶晶
- 研究方向：扩散模型与无线感知

## 任务进度总览

### 科研任务

| 项目 | 状态 | 进度 |
|---|---|---|
| 论文阅读 | 持续更新 | 已整理 11 篇笔记 |
| 代码复现 | 持续更新 | 已整理 6 个项目，完成范围见下表 |
| 研究想法 | 待验证 | [查看研究问题](research/ideas/) |

### 工程实践

| 项目 | 状态 | 内容 |
|---|---|---|
| [DigitRepair](engineering/01-DigitRepair/) | 已有可运行版本 | 条件扩散手写数字修复 |

## 目录结构

```text
grad-prep-zhoujingjing/
├── README.md                          个人简介、任务进度与内容导航
├── research/                          科研记录
│   ├── README.md                      科研资料入口
│   ├── paper-notes/                   11 篇论文阅读笔记
│   ├── papers/                        论文原文、PDF 与出处索引
│   ├── reproduction/                  代码复现
│   │   ├── README.md                  复现项目与进度
│   │   ├── 01-ddpm-cifar10/            DDPM · CIFAR-10
│   │   │   ├── code/                  模型、训练与采样
│   │   │   └── results/               Loss、生成图与去噪过程
│   │   ├── 02-ddim-cifar10/            DDIM · CIFAR-10
│   │   │   ├── code/                  50 步采样
│   │   │   └── results/               生成图与实验说明
│   │   ├── 03-meanflow-mnist/          MF / iMF / CFG · MNIST
│   │   │   ├── code/models/           DiT 模型
│   │   │   ├── legacy_unet/           旧 U-Net 实现
│   │   │   └── results/dit_stable/    训练日志、指标与结果图
│   │   │       └── fid10000/          最终 FID-10000 评估
│   │   ├── 04-cddim-wireless-channel/ cDDIM 与 One-shot U-Net 对照
│   │   │   ├── code/src/              One-shot 模型、训练与评估
│   │   │   └── results/               ID / OOD 热力图与 CDF
│   │   ├── 05-widar3-csi-dfs-bvp/      CSI → DFS → BVP
│   │   │   ├── code/
│   │   │   │   ├── Data/              官方 CSI 输入数据说明
│   │   │   │   ├── BVP/               BVP 算法
│   │   │   │   ├── Doppler2VelocityMapping/  多普勒与速度映射
│   │   │   │   ├── csi_tool_box/      CSI 读取与校正
│   │   │   │   ├── tftb/              时频分析依赖
│   │   │   │   ├── NOTES.md           学习笔记
│   │   │   │   └── QUESTIONS.md       疑问整理
│   │   │   └── results/userA-1-1-1-1/ 单样本结果、日志与核验指标
│   │   └── 06-csi-bert2/              CSI 恢复、预测与分类
│   │       ├── code/                  数据准备、训练与评估入口
│   │       │   └── src/               作者代码与 WiCount 数据
│   │       └── results/
│   │           ├── REPORT.md          实验报告
│   │           ├── figures/           预测、重建与消融图
│   │           └── runs/              各组实验日志与指标
│   └── ideas/                         待验证的研究想法
└── engineering/                       工程实践
    ├── README.md                      工程项目入口
    └── 01-DigitRepair/                 条件扩散手写数字修复
        ├── README.md                  运行方法与主要结果
        ├── DETAILS.md                 模型、实验与功能说明
        ├── src/
        │   ├── web/                   网页界面
        │   └── ml/                    模型、训练与评估
        ├── tests/                     测试代码
        ├── results/                   权重、指标与结果图
        └── screenshots/               界面截图
```

## 已完成内容速览

### 论文笔记

| 论文 | 内容 |
|---|---|
| [CLAR](research/paper-notes/01-CLAR.md) | CSI 数据增强与活动识别 |
| [Latent Diffusion Models](research/paper-notes/02-latent%20diffusion%20model.md) | 潜在空间扩散 |
| [LatentCSI](research/paper-notes/03-LatentCSI.md) | CSI 到图像生成 |
| [WiFi-Diffusion](research/paper-notes/04-wifi%20diffusion.md) | 无线电地图估计 |
| [MeanFlow](research/paper-notes/05-meanflow.md) | 平均速度与一步生成 |
| [Echo-Conditioned DDPM](research/paper-notes/06-Echo-Conditioned%20Denoising%20Diffusion.md) | RF 多目标跟踪 |
| [mmWaveFlow](research/paper-notes/07-mmWaveFlow.md) | 毫米波点云增强与生成 |
| [cDDIM](research/paper-notes/08-cDDIM.md) | 位置条件无线信道生成 |
| [ELF](research/paper-notes/09-ELF.md) | 连续语言嵌入生成 |
| [CSI-BERT2](research/paper-notes/10-CSI-BERT2.md) | CSI 恢复、预测与分类 |
| [SRLoc](research/paper-notes/11-SRLoc.md) | 对抗 RSSI 修复与定位防御 |

[论文出处与原文](research/papers/)

### 代码复现

| 项目 | 当前结果 |
|---|---|
| [DDPM · CIFAR-10](research/reproduction/01-ddpm-cifar10/) | 120 Epoch AMP 训练、生成图与去噪过程 |
| [DDIM · CIFAR-10](research/reproduction/02-ddim-cifar10/) | 复用 DDPM 权重，50 步采样 |
| [MeanFlow / iMF · MNIST](research/reproduction/03-meanflow-mnist/) | MF、iMF、CFG 各 20,000 步；Loss、FID 与生成图 |
| [cDDIM · 无线信道](research/reproduction/04-cddim-wireless-channel/) | ID / OOD 与 One-shot U-Net 对照 |
| [Widar3.0](research/reproduction/05-widar3-csi-dfs-bvp/) | 单样本 CSI → DFS → BVP |
| [CSI-BERT2](research/reproduction/06-csi-bert2/) | 三个数据集的核心实验、基线与消融；未完整达到论文指标 |

## 提交记录

[查看全部提交](https://github.com/laportaeaston923-afk/grad-prep-zhoujingjing/commits/main/)
