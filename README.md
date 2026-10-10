# Generative Models × Wireless Sensing

> **研究生阶段科研准备记录**  
> 从扩散模型基础出发，探索生成式模型在 WiFi CSI、无线环境重建和 RF 感知中的应用。

**关键词：** `DDPM` · `DDIM` · `Latent Diffusion` · `MeanFlow` · `WiFi CSI` · `RF Sensing`

[📚 论文笔记](research/paper-notes/) · [📄 论文索引](research/papers/README.md) · [💻 代码复现](research/reproduction/01-ddpm-cifar10/) · [🛠 工程实践](engineering/README.md)

---

## 01 · 关于这个仓库

你好，这里记录我的科研学习过程。目前主要沿着两条相互关联的路线推进：

- **生成模型基础：** 理解 DDPM 的加噪与去噪、DDIM 的采样方式、LDM 的潜在空间建模，以及 MeanFlow 的单步生成思想。
- **生成模型 × 无线感知：** 关注扩散模型如何处理 WiFi CSI 与 RF 回波，并应用于数据增强、场景可视化、无线电地图估计和动态目标跟踪。

我希望这个仓库不只是论文清单，而是一份可以回顾、可以验证的学习记录：

**读懂问题 → 理解方法 → 分析模型结构 → 运行代码 → 记录实验 → 提出问题**

目前以**论文阅读和基础模型复现**为主，尚未将这些阅读成果表述为独立的科研创新。

## 02 · 当前进度

| 模块 | 目前完成的内容 | 状态 | 入口 |
|---|---|---|---|
| 论文阅读 | 已整理 **6 篇**论文笔记 | 持续更新 | [阅读记录](research/README.md) |
| 论文资料 | 按「生成模型基础 / 无线感知应用」整理索引，已有 **4 份 PDF** | 已建立索引 | [论文出处](research/papers/README.md) |
| DDPM 训练 | CIFAR-10，初始训练 **100 Epoch**，已编写复现说明 | 本地实验已开展；代码和结果待上传 | [实验记录](research/reproduction/01-ddpm-cifar10/) |
| DDIM 采样 | 复用 DDPM 权重，采样脚本调用 `ddim_sample(steps=50)` | 采样实现与结果待进一步核验、整理 | [DDIM 记录](research/reproduction/01-ddpm-cifar10/#6-ddim-采样复现) |
| 继续训练 | 基于已有 DDPM 权重，增加 **20 Epoch** 的训练实验 | 日志与 Loss 曲线整理中 | [训练记录](research/reproduction/01-ddpm-cifar10/#52-继续训练) |
| 研究想法与工程实践 | 已建立对应目录 | 待持续补充 | [Ideas](research/ideas/) / [Engineering](engineering/) |

> **记录原则：** 区分「阅读过」「本地运行过」「已上传代码」「已有可验证结果」。没有上传的实验图片、指标和运行日志，不提前标记为完成。

## 03 · 论文阅读与研究方向

### A. 生成模型基础

| 论文 / 方法 | 关注的问题 | 学习记录 |
|---|---|---|
| **Latent Diffusion Models (LDM)** | 为什么将扩散过程从像素空间转移到潜在空间？ | [阅读笔记](research/paper-notes/02-latent%20diffusion%20model.md) |
| **MeanFlow** | 是否能通过学习平均速度实现单步生成？ | [阅读笔记](research/paper-notes/05-meanflow.md) |
| **DDPM / DDIM** | 如何训练噪声预测网络，以及为什么可以替换采样方式？ | [CIFAR-10 复现记录](research/reproduction/01-ddpm-cifar10/) |

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

## 04 · 代码复现：DDPM → DDIM

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
| 初始训练 | 100 Epoch |
| Batch Size / Learning Rate | 256 / 2e-4 |
| 优化器 / 损失函数 | Adam / MSE |
| DDIM 尝试 | 复用 `ddpm.pth`，`steps=50` |
| 扩展实验 | 加载已有权重继续训练 20 Epoch |

**实验记录：** [DDPM / DDIM · CIFAR-10](research/reproduction/01-ddpm-cifar10/)

基础 DDPM 代码参考了 [Bilibili 教学视频](https://www.bilibili.com/video/BV1vZi6BaEgF/)。该项目以学习、理解和复现为目的；参考实现不声明为独立原创。DDIM 内部采样实现及采样加速效果仍需结合代码和实际结果验证。

目前 GitHub 上已有实验的 README，**训练脚本、采样脚本、Loss 日志、生成图像与模型权重尚未公开上传**，后续将按实际实验进度补充。

## 05 · 我的学习与记录方式

阅读每篇论文时，尽量使用一致的框架：

1. **核心内容：** 论文到底解决什么问题？
2. **已有方法的局限：** 为什么现有方法还不够？
3. **核心思想：** 为什么选用扩散模型？它在这里负责生成什么？
4. **整体框架：** 输入、模块、训练目标、推理流程与实验验证如何连接？
5. **总结与我的理解：** 哪些地方真正理解了，哪些问题还需要验证？

代码复现则侧重记录**实际配置、可运行代码、真实日志、生成结果和遇到的问题**，不单纯追求“跑通”。

## 06 · 后续计划

- [x] 建立论文阅读笔记与原文索引
- [x] 整理 DDPM 训练和 DDIM 采样的实验说明
- [ ] 上传 DDPM / DDIM 的实际代码与依赖说明（需核对参考代码许可）
- [ ] 上传 Loss 曲线、训练日志与生成样本
- [ ] 检查 `ddim_sample()` 的采样公式，并比较相同权重下的采样耗时与效果
- [ ] 补充无线感知论文的实验条件、局限与横向对比
- [ ] 逐步积累可验证的研究想法和工程实践记录

## 07 · 仓库目录

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
│   │   └── 01-ddpm-cifar10/
│   │       └── README.md
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
| [research/reproduction/](research/reproduction/) | DDPM / DDIM 复现记录 |
| [research/ideas/](research/ideas/) | 研究想法（待补充） |
| [engineering/](engineering/) | 工程实践（待补充） |

---

> **持续学习，重视理解，也重视可验证的实验结果。**
