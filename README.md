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
| 代码复现 | 持续更新 | 已整理 5 个项目，完成范围见下表 |
| 研究想法 | 待验证 | [4 个扩散模型与无线感知方向](research/ideas/readme.md) |

### 工程实践

| 项目 | 状态 | 内容 |
|---|---|---|
| [DigitRepair](engineering/01-DigitRepair/) | 已有可运行版本 | 条件扩散手写数字修复 |

## 目录结构

```text
grad-prep-zhoujingjing/
├── README.md                         个人简介 + 科研与工程两项任务的进度总览
│
├── research/                         科研任务
│   ├── README.md                     论文列表 + 复现进度 + 研究想法总览
│   ├── paper-notes/                  每篇论文的阅读笔记
│   │   ├── README.md                 论文笔记目录
│   │   ├── 01-CLAR.md
│   │   ├── 02-latent diffusion model.md
│   │   ├── 03-LatentCSI.md
│   │   ├── 04-wifi diffusion.md
│   │   ├── 05-meanflow.md
│   │   ├── 06-Echo-Conditioned Denoising Diffusion.md
│   │   ├── 07-mmWaveFlow.md
│   │   ├── 08-cDDIM.md
│   │   ├── 09-ELF.md
│   │   ├── 10-CSI-BERT2.md
│   │   └── 11-SRLoc.md
│   ├── papers/                       论文 PDF 与原文链接
│   │   ├── README.md                 论文出处与原文索引
│   │   └── *.pdf                     已收录的论文原文
│   ├── reproduction/                 论文复现，每个项目一个子目录
│   │   ├── README.md                 复现项目与进度总览
│   │   ├── 01-ddpm-cifar10/
│   │   │   ├── README.md             环境配置 + 运行说明
│   │   │   ├── code/                 DDPM 训练与采样代码
│   │   │   └── results/              权重、Loss 曲线、生成图与去噪过程
│   │   ├── 02-ddim-cifar10/
│   │   │   ├── README.md             环境配置 + 运行说明
│   │   │   ├── code/                 DDIM 采样代码
│   │   │   └── results/              采样结果与实验说明
│   │   ├── 03-meanflow-mnist/
│   │   │   ├── README.md             环境配置 + 运行说明
│   │   │   ├── requirements.txt      依赖配置
│   │   │   ├── code/                 MeanFlow / iMF 训练与评估代码
│   │   │   ├── legacy_unet/          旧版 U-Net 实现
│   │   │   └── results/              Loss、FID、生成图与实验记录
│   │   ├── 04-cddim-wireless-channel/
│   │   │   ├── README.md             环境配置 + 运行说明
│   │   │   ├── code/                 信道生成对照实验代码
│   │   │   └── results/              信道热力图、CDF 与训练曲线
│   │   └── 05-csi-bert2/
│   │       ├── README.md             环境配置 + 运行说明
│   │       ├── code/                 CSI 恢复、预测、分类与评估代码
│   │       └── results/              实验报告、指标与结果图
│   └── ideas/                        自己的研究想法与创新
│       ├── readme.md                 四个研究想法总览
│       ├── 01-diffusion-csi-augmentation.md
│       ├── 02-diffusion-csi-recovery.md
│       ├── 03-diffusion-csi-human-reconstruction.md
│       └── 04-diffusion-radio-map-localization.md
│
└── engineering/                      工程实践
    ├── README.md                     工程项目介绍与入口
    └── 01-DigitRepair/                条件扩散手写数字修复项目
        ├── README.md                 项目介绍 + 功能说明 + 运行入口
        ├── DETAILS.md                配置、功能、实验结果与运行截图
        ├── requirements.txt          依赖配置
        ├── src/                      项目源代码
        │   ├── server.py             本地模型服务
        │   ├── engine.py             推理引擎
        │   ├── ml/                   模型、训练与评估
        │   └── web/                  网页界面
        ├── results/                  模型权重与实验结果
        ├── screenshots/              运行截图
        └── tests/                    功能与接口检查
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
| [CSI-BERT2](research/reproduction/05-csi-bert2/) | 三个数据集的核心实验、基线与消融；未完整达到论文指标 |

## 提交记录

[查看全部提交](https://github.com/laportaeaston923-afk/grad-prep-zhoujingjing/commits/main/)
