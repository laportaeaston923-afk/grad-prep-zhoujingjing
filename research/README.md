# 科研记录 · 生成模型 × 无线感知与通信

通过论文阅读与代码实验，理解生成模型的训练、采样及其在无线感知与无线通信中的用途。

[论文笔记](paper-notes/) · [论文索引](papers/README.md) · [代码复现](reproduction/README.md) · [研究想法](ideas/) · [返回仓库首页](../README.md)

## 01 · 论文阅读

目前已整理 8 篇论文阅读笔记。原文与官方出处见 [论文索引](papers/README.md)。

| 编号 | 论文 / 方法 | 主要内容 | 阅读笔记 |
|---|---|---|---|
| 01 | CLAR | DDPM 数据增强与 WiFi 人体活动识别 | [查看笔记](paper-notes/01-CLAR.md) |
| 02 | Latent Diffusion Models | 潜在空间扩散模型 | [查看笔记](paper-notes/02-latent%20diffusion%20model.md) |
| 03 | LatentCSI | 利用 CSI 生成 RGB 图像 | [查看笔记](paper-notes/03-LatentCSI.md) |
| 04 | WiFi-Diffusion | 无线电地图估计与环境重建 | [查看笔记](paper-notes/04-wifi%20diffusion.md) |
| 05 | MeanFlow | 平均速度建模与一步生成 | [查看笔记](paper-notes/05-meanflow.md) |
| 06 | Echo-Conditioned DDPM | RF 多目标跟踪与波束选择 | [查看笔记](paper-notes/06-Echo-Conditioned%20Denoising%20Diffusion.md) |
| 07 | mmWaveFlow | Flow Matching 统一稀疏 / 稠密人体点云的双向转换 | [查看笔记](paper-notes/07-mmWaveFlow.md) |
| 08 | cDDIM | 位置条件 MIMO 信道生成与数据增强 | [查看笔记](paper-notes/08-cDDIM.md) |

## 02 · 研究任务对比

不同论文中，生成模型所解决的问题不同。

| 论文 | 生成或预测对象 | 核心作用 |
|---|---|---|
| CLAR | WiFi CSI 数据 | 数据增强，辅助活动识别 |
| LatentCSI | 图像潜在表示 | 无线场景可视化 |
| WiFi-Diffusion | Radio Map | 无线环境重建 |
| Echo-Conditioned DDPM | 未来目标状态 | 目标跟踪与后续波束选择 |
| cDDIM | 用户位置条件下的 MIMO 信道 | 数据增强，辅助下游通信任务 |
| mmWaveFlow | 稀疏 / 稠密人体点云 | 点云增强与毫米波点云生成 |
| MeanFlow | 数据样本 | 研究一步生成与采样效率 |

阅读时重点记录：输入是什么、生成什么、如何训练、怎样评估，以及生成结果如何帮助最终任务。

## 03 · 代码复现

| 项目 | 数据集 | 已保存的内容 | 入口 |
|---|---|---|---|
| DDPM | CIFAR-10 | 120 Epoch AMP 训练代码；Loss 曲线、生成图与去噪过程图 | [DDPM 记录](reproduction/01-ddpm-cifar10/) |
| DDIM | CIFAR-10 | 复用 DDPM 网络权重的 50 步采样代码与生成结果图 | [DDIM 记录](reproduction/02-ddim-cifar10/) |
| MF vs iMF | MNIST | 相同条件下对比 JVP 方向、训练 Loss 与一步 FID（待正式训练）；历史日志保留 | [对比复现](reproduction/03-meanflow-mnist/) |

DDPM 最终实验采用 `train_full_120_amp.py` 对应的完整 120 Epoch AMP 训练，Loss 以该次 `train_results_120_amp/` 记录为准。DDIM 复用噪声预测网络，不另行训练；当前 checkpoint 与统一速度对照的细节见项目说明。

### MF vs iMF：新的对照实验

新实验只比较 **MF / iMF 的 JVP 输入方向**，在 MNIST 上独立训练两组模型，用相同数据、相同网络、相同种子，评估 **一步生成（1 NFE）** 的 Loss 曲线、FID 与耗时。正式结果尚未运行，不引用旧上传权重的指标填表。

原 50,000 step 训练日志、FID 与 CFG 评估作为历史记录保存在 [results](reproduction/03-meanflow-mnist/results/) 中，不用作本次 MF/iMF 对照的证据。配置及启动命令见 [新的 MF vs iMF 复现 README](reproduction/03-meanflow-mnist/)。

## 04 · 后续研究记录

- DDPM / DDIM：核对同一 checkpoint，补齐相同设备、相同设置下的采样时间与质量对照。
- MF vs iMF：按新对照配置分别训练两组网络，分析 Loss 稳定性、一步 FID 和生成效果。
- 无线感知：结合论文的数据、任务与评估方式，选择具体的复现问题，并记录在 [Ideas](ideas/)。

[查看三个实验的统一总结](reproduction/README.md) · [返回仓库首页](../README.md)
