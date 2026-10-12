# 科研记录 · 生成模型 × 无线感知

通过论文阅读与代码实验，理解生成模型的训练、采样及其在无线感知中的用途。

[论文笔记](paper-notes/) · [论文索引](papers/README.md) · [代码复现](reproduction/README.md) · [研究想法](ideas/) · [返回仓库首页](../README.md)

## 01 · 论文阅读

目前已整理 6 篇论文阅读笔记。原文与官方出处见 [论文索引](papers/README.md)。

| 编号 | 论文 / 方法 | 主要内容 | 阅读笔记 |
|---|---|---|---|
| 01 | CLAR | DDPM 数据增强与 WiFi 人体活动识别 | [查看笔记](paper-notes/01-CLAR.md) |
| 02 | Latent Diffusion Models | 潜在空间扩散模型 | [查看笔记](paper-notes/02-latent%20diffusion%20model.md) |
| 03 | LatentCSI | 利用 CSI 生成 RGB 图像 | [查看笔记](paper-notes/03-LatentCSI.md) |
| 04 | WiFi-Diffusion | 无线电地图估计与环境重建 | [查看笔记](paper-notes/04-wifi%20diffusion.md) |
| 05 | MeanFlow | 平均速度建模与一步生成 | [查看笔记](paper-notes/05-meanflow.md) |
| 06 | Echo-Conditioned DDPM | RF 多目标跟踪与波束选择 | [查看笔记](paper-notes/06-Echo-Conditioned%20Denoising%20Diffusion.md) |

## 02 · 研究任务对比

不同论文中，生成模型所解决的问题不同。

| 论文 | 生成或预测对象 | 核心作用 |
|---|---|---|
| CLAR | WiFi CSI 数据 | 数据增强，辅助活动识别 |
| LatentCSI | 图像潜在表示 | 无线场景可视化 |
| WiFi-Diffusion | Radio Map | 无线环境重建 |
| Echo-Conditioned DDPM | 未来目标状态 | 目标跟踪与后续波束选择 |
| MeanFlow | 数据样本 | 研究一步生成与采样效率 |

阅读时重点记录：输入是什么、生成什么、如何训练、怎样评估，以及生成结果如何帮助最终任务。

## 03 · 代码复现

| 项目 | 数据集 | 已保存的内容 | 入口 |
|---|---|---|---|
| DDPM | CIFAR-10 | 120 Epoch AMP 训练代码；Loss 曲线、生成图与去噪过程图 | [DDPM 记录](reproduction/01-ddpm-cifar10/) |
| DDIM | CIFAR-10 | 复用 DDPM 网络权重的 50 步采样代码与生成结果图 | [DDIM 记录](reproduction/02-ddim-cifar10/) |
| MeanFlow | MNIST | 50,000 step 原日志与配置；Loss CSV/曲线；一步/五步过程；FID 与 CFG 对照 | [MeanFlow 记录](reproduction/03-meanflow-mnist/) |

DDPM 最终实验采用 `train_full_120_amp.py` 对应的完整 120 Epoch AMP 训练，Loss 以该次 `train_results_120_amp/` 记录为准。DDIM 复用噪声预测网络，不另行训练；当前 checkpoint 与统一速度对照的细节见项目说明。

### MeanFlow：已完成的实测

训练日志与配置标注 `meanflow`、50,000 step、batch=2；每 500 step 记录一次，共 100 个真实记录点，总 Loss 1.985971 → 1.856935。

| 采样 | CFG=1，FID ↓ | CFG=4.25，FID ↓ |
|---|---:|---:|
| 一步 | 179.8574 | **150.9314** |
| 五步 | **17.9291** | 35.2649 |

每个设置 10,000 张生成图 vs 同一组 10,000 张 MNIST test，采用 CPU / FP32、Inception-v3-compat 2048 维特征。两组都有数字类别条件，区别是是否加强引导；本次一步 FID 改善、五步 FID 变差，不能概括为“有 CFG 一定更好”。

| 想查看的内容 | 入口 |
|---|---|
| 原训练日志、配置、Loss CSV 与曲线 | [真实训练记录](reproduction/03-meanflow-mnist/results/mnist-meanflow-50k-training/) |
| 一步 / 五步静态过程图与 GIF | [生成过程可视化](reproduction/03-meanflow-mnist/results/uploaded-step-0050000-process/) |
| CFG 并排图片、动图、FID 与复测命令 | [CFG 对照实验](reproduction/03-meanflow-mnist/results/uploaded-step-0050000-cfg-comparison/) |
| 项目设置、代码来源与所有结果 | [MeanFlow 复现页面](reproduction/03-meanflow-mnist/) |

这些图像实验用于学习生成模型机制，尚不是无线感知任务的实测结果。MNIST 与 CIFAR-10 的分数不能直接用来比较方法优劣。

## 04 · 后续研究记录

- DDPM / DDIM：核对同一 checkpoint，补齐相同设备、相同设置下的采样时间与质量对照。
- MeanFlow：继续分析一步生成质量，区分 FID、类别一致性与多样性；现有 CFG 对照比较推理强度，未分别训练两个模型。
- 无线感知：结合论文的数据、任务与评估方式，选择具体的复现问题，并记录在 [Ideas](ideas/)。

[查看三个实验的统一总结](reproduction/README.md) · [返回仓库首页](../README.md)
