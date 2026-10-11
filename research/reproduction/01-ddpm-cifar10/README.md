# DDPM · CIFAR-10 训练与采样复现

> 本目录记录 DDPM 在 CIFAR-10 上通过 AMP 混合精度从头训练 120 Epoch、模型权重保存、原始反向采样与完整 Loss 曲线。DDIM 单独记录在 [02-ddim-cifar10](../02-ddim-cifar10/)；两个实验的共同总结见 [复现总览](../README.md)。

## 1. 实验介绍

本项目使用 PyTorch 在 CIFAR-10 数据集上学习与复现 DDPM，并将训练好的模型权重用于后续 DDIM 采样实验。

本次实验的主线：

1. 使用 `train_full_120_amp.py` 以 AMP 混合精度从头训练 DDPM 的 U-Net 噪声预测模型，共 120 Epoch。
2. 保存训练所得模型权重；最终 AMP 实验实际生成的 checkpoint 文件名与旧 `ddpm.pth` 的关系待核对。
3. 使用 DDPM 原有采样函数生成图像。
4. 将相同的训练权重用于单独的 [DDIM 复现实验](../02-ddim-cifar10/)。
5. 以 `train_results_120_amp/` 中这次完整 120 Epoch 运行的真实 Loss 为准，整理训练曲线和生成效果。

参考论文：

- **DDPM：** Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020 — [论文](https://arxiv.org/abs/2006.11239)
- **DDIM：** Song et al., *Denoising Diffusion Implicit Models*, ICLR 2021 — [论文](https://arxiv.org/abs/2010.02502)

## 2. 代码来源与复现说明

本项目是用于学习的代码复现，**DDPM 基础实现参考 B 站教学视频**：

- [Bilibili：DDPM 代码讲解与复现](https://www.bilibili.com/video/BV1vZi6BaEgF/)

在此基础上，自己进行 CIFAR-10 训练、保存权重，并将采样脚本中的 DDPM 调用替换为 DDIM 调用。本次 DDPM 结果以 `train_full_120_amp.py` 和 `train_results_120_amp/` 所对应的完整 120 Epoch AMP 实验为准。

**说明：** 基础实现并非从零独立原创；若后续上传参考教程中的代码，需保留来源说明并核对原作者的代码许可。DDIM 采样公式的具体实现需要结合 `ddpm.py` 中的 `ddim_sample()` 函数进一步检查。

## 3. DDPM 核心实现

- **前向扩散：** 逐步向真实图像添加高斯噪声。
- **U-Net 噪声预测：** 学习从加噪图像 `x_t` 预测添加的噪声。
- **反向采样：** 从随机噪声出发逐步生成图像。
- **完整训练：** 从头运行 120 Epoch，记录这一轮实验的真实训练 Loss。

核心噪声预测目标：

```math
\mathcal{L}=\mathbb{E}_{x_0,t,\epsilon}
\left[\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|_2^2\right]
```

## 4. 实验配置

最终采用 `train_full_120_amp.py` 作为训练脚本，`train_results_120_amp/` 作为 Loss 记录来源。下列网络结构及部分训练参数来自此前展示的脚本，尚需与最终 AMP 脚本核对：

| 参数 | 设置 |
|---|---|
| 数据集 | CIFAR-10 |
| 图片尺寸 | 3 × 32 × 32 |
| 噪声预测网络 | U-Net |
| U-Net 基础通道数 | 64 |
| U-Net 通道倍率 | (1, 2, 4) |
| U-Net 注意力配置 | (False, False, False) |
| 每级残差块数量 | 2 |
| 扩散时间步 | 1000 |
| 训练 Batch Size | 256 |
| Learning Rate | 2e-4 |
| 优化器（此前展示的训练脚本） | Adam（最终 AMP 配置待核对） |
| 损失函数 | MSELoss |
| 图像归一化 | 每通道使用 mean=0.5、std=0.5，映射至 [-1,1] |
| DataLoader | shuffle=True、num_workers=4 |
| 最终实验 | `train_full_120_amp.py` 从头训练 120 Epoch（AMP） |
| Loss 来源 | `train_results_120_amp/` |
| 单次采样图片 | 16 张 |

## 5. DDPM 训练与结果

### 5.1 AMP 混合精度从头训练 120 Epoch

本次最终采用的是 `train_full_120_amp.py` **以 AMP 混合精度重新从头训练 120 Epoch** 的实验，而不是“先训练 100 Epoch，再从旧 checkpoint 继续训练 20 Epoch”。

训练的基本过程：

```text
CIFAR-10 图像 x₀（归一化至 [-1,1]）
     ↓
随机采样时间步 t 和高斯噪声 ε
     ↓
ddpm.q_sample(x₀, t, ε) → 得到 xₜ
     ↓
ddpm.eps_model(xₜ, t) → 预测噪声 εθ
     ↓
计算 MSE(εθ, ε)
     ↓
反向传播与优化器更新
```

此前展示的 `train.py` 使用 Adam、MSELoss 和 100 Epoch 配置；但**本次最终结果以实际运行的 `train_full_120_amp.py` 为准**，具体 AMP、学习率和批大小设置仍需与该脚本核对，不将旧脚本参数或日志冒充为本次记录。

### 5.2 完整 120 Epoch Loss 曲线

只展示 `train_results_120_amp/` 中这次 AMP 从头训练 **Epoch 1–120** 的真实 Loss，不拼接旧的 100 Epoch 和继续训练日志。

这次从头训练的 Loss 图片来源于本地 `train_results_120_amp/training_loss.png`。图中展示了原始每轮平均 Loss 及 5 Epoch 移动平均曲线：前期快速下降，随后逐步趋于平稳，末段在大约 0.03 附近波动。**Loss 下降仅说明噪声预测训练目标的优化趋势，并不单独证明图像生成质量。**

**待上传原图：** [training_loss.png](results/training_loss.png)（上传后可查看）。

### 5.3 DDPM 生成图像

本地生成结果为 **4×4 图像网格，共 16 张 32×32 RGB 图片**。图中可以看到动物、车辆等具有一定结构的样本，但部分样本仍存在明显模糊与细节不足。

**待上传原图：** [ddpm_samples.png](results/ddpm_samples.png)（上传后可查看）。

**权重来源提醒：** 此前分享的 `sample.py` 从项目根目录加载 `ddpm.pth`，而最终 AMP 训练目录中另有 `checkpoint_amp.pth` 和其他权重文件。因此，暂时不能认定这张采样图来自最终 AMP 120 Epoch checkpoint；需要在代码中确认实际加载的模型。

### 5.4 反向去噪过程可视化

本地 `denoising_trajectory.png` 展示了同一个样本从随机噪声经过多次反向去噪，逐渐形成可辨认图像的过程。它可以直观说明 DDPM 的迭代式生成特点。

**待上传原图：** [denoising_trajectory.png](results/denoising_trajectory.png)（上传后可查看）。

**说明：** 此图的采样时间步及权重来源尚未核对，暂不填写未经验证的具体设置。

## 6. 实验分析与个人理解

DDPM 的学习重点是理解：如何在真实图片上添加不同程度的高斯噪声，并训练 U-Net 预测所加噪声；采样时再从随机噪声逐步生成图像。

本次最终采用 `train_full_120_amp.py` 的 **AMP 从头训练 120 Epoch** 实验，Loss 以 `train_results_120_amp/training_loss.png` 为准，不采用旧的继续训练日志。

结合目前本地三张实验图片，可分别说明**训练收敛趋势、生成图像的视觉质量、反向去噪过程**。但不同图片的 checkpoint 来源需要分别核对，不将不同训练运行误写为同一次实验的结果。

预计上传的三张原始图片详见 [results/](results/)。

进一步的采样研究单独放在 [DDIM · CIFAR-10](../02-ddim-cifar10/)，通过复用训练好的 U-Net 权重，比较不同的反向采样过程。

**代码来源：** DDPM 基础实现参考了 [Bilibili 教学视频](https://www.bilibili.com/video/BV1vZi6BaEgF/)；上传源码前需要确认其再分发许可。
---

[← 返回代码复现目录](../) · [← 返回科研主页](../../README.md)
