# 个人简介

- 姓名：周晶晶
- 目前学习的方向：生成模型 × 无线感知

> **研究生阶段科研记录**  

[📚 论文笔记](research/paper-notes/) · [📄 论文索引](research/papers/README.md) · [💻 代码复现](research/reproduction/README.md) · [🛠 工程实践](engineering/README.md)

---


## 01 · 当前进度

| 模块 | 目前完成的内容 | 状态 | 入口 |
|---|---|---|---|
| 论文阅读 | 已整理 **6 篇**论文笔记 | 持续更新 | [阅读记录](research/README.md) |
| 论文资料 | 按「生成模型基础 / 无线感知应用」整理索引，已有 **4 份 PDF** | 已建立索引 | [论文出处](research/papers/README.md) |
| DDPM 训练 | CIFAR-10， | 代码和结果已上传 | [实验记录](research/reproduction/01-ddpm-cifar10/) |
| DDIM 采样 | 复用 DDPM 权重，采样脚本调用 `ddim_sample(steps=50)` | 代码和结果已上传 | [DDIM 记录](research/reproduction/02-ddim-cifar10/) |
| 研究想法与工程实践 | 已建立对应目录 | 待持续补充 | [Ideas](research/ideas/) / [Engineering](engineering/) |

## 02 · 论文阅读与研究方向

### A. 生成模型基础

| 论文 / 方法 | 关注的问题 | 学习记录 |
|---|---|---|
| **Latent Diffusion Models (LDM)** | 为什么将扩散过程从像素空间转移到潜在空间？ | [阅读笔记](research/paper-notes/02-latent%20diffusion%20model.md) |
| **MeanFlow** | 是否能通过学习平均速度实现单步生成？ | [阅读笔记](research/paper-notes/05-meanflow.md) |
| **DDPM / DDIM** | 如何训练噪声预测网络，以及为什么可以替换采样方式？ | [DDPM / DDIM 复现总览](research/reproduction/README.md) |

### B. 扩散模型在无线感知中的应用

| 论文 / 方法 | 输入与研究任务 | 扩散模型的作用 | 阅读笔记 |
|---|---|---|---|
| **CLAR** | WiFi CSI → 人体活动识别 | 生成增强样本，辅助特征学习 | [01 · CLAR](research/paper-notes/01-CLAR.md) |
| **LatentCSI** | WiFi CSI → RGB 图像 | 将 CSI 映射到图像潜在表示，辅助可视化生成 | [03 · LatentCSI](research/paper-notes/03-LatentCSI.md) |
| **WiFi-Diffusion** | 稀疏无线信号采样 → Radio Map | 生成并筛选候选无线电地图 | [04 · WiFi-Diffusion](research/paper-notes/04-wifi%20diffusion.md) |
| **Echo-Conditioned DDPM** | RF 回波 → 下一时间块目标状态 | 条件生成目标位置预测，指导感知波束选择 | [06 · Echo-Conditioned DDPM](research/paper-notes/06-Echo-Conditioned%20Denoising%20Diffusion.md) |

**我目前尤其关注的问题是：** 在不同无线感知任务里，扩散模型究竟生成了什么？生成的数据或状态又怎样帮助最终任务？

- **CLAR：** 生成的是用于训练的 CSI 增强数据。
- **LatentCSI：** 建立无线信息到视觉潜在表示的映射。
- **WiFi-Diffusion：** 生成候选 Radio Map，以补全稀疏观测。
- **Echo-Conditioned DDPM：** 生成未来目标状态的预测，用于后续跟踪与波束选择。

完整的论文原文及来源链接见 **[Papers · 论文出处与链接](research/papers/README.md)**。

## 03 · 代码复现：DDPM → DDIM

当前复现以 **CIFAR-10 图像生成**为基础实验，用于验证对扩散模型训练与采样机制的理解。

```text
CIFAR-10 图像
      ↓
DDPM 前向加噪 + U-Net 噪声预测训练
      ↓
保存模型权重 ddpm.pth
      ├── DDPM：原始反向采样
      └── DDIM：复用权重，尝试 50 步采样
```

| 实验设置 | 当前记录 |
|---|---|
| 数据集 | CIFAR-10（32 × 32 RGB） |
| 噪声预测网络 | U-Net |
| 扩散时间步 | 1000 |
| 最终训练方案 | `train_full_120_amp.py`：AMP 混合精度，从头训练 120 Epoch |
| Batch Size / Learning Rate | 256 / 2e-4 |
| 优化器 / 损失函数 | Adam / MSE |
| DDIM 尝试 | 复用 `ddpm.pth`，`steps=50` |
| 训练 Loss | 采用 `train_results_120_amp/` 中完整 120 Epoch 的真实 Loss（待上传） |

**实验记录：** [DDPM / DDIM 统一总结](research/reproduction/README.md) · [DDPM](research/reproduction/01-ddpm-cifar10/) · [DDIM](research/reproduction/02-ddim-cifar10/)

基础 DDPM 代码参考了 [Bilibili 教学视频](https://www.bilibili.com/video/BV1vZi6BaEgF/)。该项目以学习、理解和复现为目的；参考实现不声明为独立原创。DDIM 内部采样实现及采样加速效果仍需结合代码和实际结果验证。

目前 GitHub 上已有实验的 README，**最终训练脚本 `train_full_120_amp.py`、来自 `train_results_120_amp/` 的完整 Loss 记录、采样脚本及生成图像尚未公开上传**，后续将按实际实验进度补充。

## 04 · 仓库目录

以下是当前已存在的主要目录和文件（不包含尚未上传的本地代码和实验图片）：

```text
grad-prep-zhoujingjing/
├── README.md
├── research/
│   ├── README.md
│   ├── paper-notes/
│   │   ├── 01-CLAR.md
│   │   ├── 02-latent diffusion model.md
│   │   ├── 03-LatentCSI.md
│   │   ├── 04-wifi diffusion.md
│   │   ├── 05-meanflow.md
│   │   └── 06-Echo-Conditioned Denoising Diffusion.md
│   ├── papers/
│   │   ├── README.md
│   │   └── 4 份论文 PDF
│   ├── reproduction/
│   │   ├── README.md               # DDPM / DDIM 统一总结
│   │   ├── 01-ddpm-cifar10/
│   │   │   └── README.md           # DDPM 单独记录
│   │   └── 02-ddim-cifar10/
│   │       └── README.md           # DDIM 单独记录
│   └── ideas/
│       └── readme.md
└── engineering/
    └── README.md
```

### 快速导航

| 路径 | 内容 |
|---|---|
| [research/README.md](research/README.md) | 论文学习与研究任务总览 |
| [research/paper-notes/](research/paper-notes/) | 6 篇论文笔记 |
| [research/papers/README.md](research/papers/README.md) | 论文 PDF 与官方出处 |
| [research/reproduction/](research/reproduction/README.md) | DDPM / DDIM 统一总结及两个独立项目 |
| [research/ideas/](research/ideas/) | 研究想法（待补充） |
| [engineering/](engineering/) | 工程实践（待补充） |

---

