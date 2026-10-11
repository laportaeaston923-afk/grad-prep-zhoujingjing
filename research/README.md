## 论文阅读

目前已整理 6 篇论文阅读笔记。

| | 论文 / 方法 | 主要内容 | 阅读笔记 |
|---|---|---|---|
| 01 | CLAR | DDPM 数据增强与 WiFi 人体活动识别 | [查看笔记](paper-notes/01-CLAR.md) |
| 02 | Latent Diffusion Models | 潜在空间扩散模型 | [查看笔记](paper-notes/02-latent%20diffusion%20model.md) |
| 03 | LatentCSI | 利用 CSI 生成 RGB 图像 | [查看笔记](paper-notes/03-LatentCSI.md) |
| 04 | WiFi-Diffusion | 无线电地图估计与环境重建 | [查看笔记](paper-notes/04-wifi%20diffusion.md) |
| 05 | MeanFlow | 单步生成与高效采样 | [查看笔记](paper-notes/05-meanflow.md) |
| 06 | Echo-Conditioned DDPM | RF 多目标跟踪与波束选择 | [查看笔记](paper-notes/06-Echo-Conditioned%20Denoising%20Diffusion.md) |

---

## 论文研究内容对比

不同论文中，生成模型所解决的问题并不相同。

| 论文 | 生成或预测对象 | 核心作用 |
|---|---|---|
| CLAR | WiFi CSI 数据 | 数据增强 |
| LatentCSI | 图像潜在表示 | 无线场景可视化 |
| WiFi-Diffusion | Radio Map | 无线环境重建 |
| Echo-Conditioned DDPM | 未来目标状态 | 目标跟踪 |
| MeanFlow | 数据样本 | 提高生成效率 |

通过对比不同任务，理解扩散模型在无线感知领域的应用方式。

---

## 代码复现

| 项目 | 数据集 | 实验内容 | 状态 |
|---|---|---|---|
| [DDPM](reproduction/01-ddpm-cifar10/) | CIFAR-10 | 100 Epoch 训练、原始采样与继续训练 | 代码及结果待上传 |
| [DDIM](reproduction/02-ddim-cifar10/) | CIFAR-10 | 复用 DDPM 权重、尝试 50 步采样 | 代码及结果待上传 |

目前已开展 DDPM/DDIM 在 CIFAR-10 数据集上的训练。

后续逐步补充：

- PyTorch 模型代码
- 训练配置与超参数
- Loss 曲线
- 模型生成结果
- 实验分析

详见 [DDPM / DDIM 复现总览](reproduction/README.md)，分别进入两个独立的代码目录。

---

