# DDPM / DDIM · CIFAR-10 代码复现

> 基于同一个训练好的 U-Net，分别尝试 DDPM 与 DDIM 采样，理解**扩散模型的训练与采样可以分开设计**。

## 1. 实验介绍

本项目使用 PyTorch 在 CIFAR-10 数据集上学习与复现 DDPM，并在 DDPM 的基础上尝试 DDIM 采样。

本次实验的主线：

1. 通过 DDPM 训练 U-Net 噪声预测模型（初始训练 100 Epoch）。
2. 保存并复用模型权重 `ddpm.pth`。
3. 使用 DDPM 原有采样函数生成图像。
4. 在相同模型结构、相同权重下，将采样调用切换为 DDIM（`steps=50`）。
5. 在已有权重基础上继续训练 20 Epoch，并整理 Loss 曲线及生成效果。

参考论文：

- **DDPM：** Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020 — [论文](https://arxiv.org/abs/2006.11239)
- **DDIM：** Song et al., *Denoising Diffusion Implicit Models*, ICLR 2021 — [论文](https://arxiv.org/abs/2010.02502)

## 2. 代码来源与复现说明

本项目是用于学习的代码复现，**DDPM 基础实现参考 B 站教学视频**：

- [Bilibili：DDPM 代码讲解与复现](https://www.bilibili.com/video/BV1vZi6BaEgF/)

在此基础上，自己进行 CIFAR-10 训练、保存权重、尝试继续训练，并将采样脚本中的 DDPM 调用替换为 DDIM 调用。

**说明：** 基础实现并非从零独立原创；若后续上传参考教程中的代码，需保留来源说明并核对原作者的代码许可。DDIM 采样公式的具体实现需要结合 `ddpm.py` 中的 `ddim_sample()` 函数进一步检查。

## 3. DDPM 核心实现

- **前向扩散：** 逐步向真实图像添加高斯噪声。
- **U-Net 噪声预测：** 学习从加噪图像 `x_t` 预测添加的噪声。
- **反向采样：** 从随机噪声出发逐步生成图像。
- **继续训练：** 加载保存的 `ddpm.pth`，在原模型基础上训练更多轮次。

核心噪声预测目标：

```math
\mathcal{L}=\mathbb{E}_{x_0,t,\epsilon}
\left[\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|_2^2\right]
```

## 4. 实验配置

以下为本次训练及所展示采样脚本中的配置：

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
| 优化器（初始训练） | Adam |
| 损失函数 | MSELoss |
| 图像归一化 | 每通道使用 mean=0.5、std=0.5，映射至 [-1,1] |
| DataLoader | shuffle=True、num_workers=4 |
| 初始训练 | 100 Epoch |
| 继续训练 | 额外 20 Epoch（结果整理中） |
| 单次采样图片 | 16 张 |
| DDIM 采样步数 | 50 |

## 5. DDPM 训练与结果

### 5.1 初始训练（`train.py`）

在 CIFAR-10 训练集上训练 DDPM，共 100 Epoch，并在训练结束后将模型参数保存为 `ddpm.pth`。

**实际训练流程：**

```text
CIFAR-10 图像 x₀（归一化至 [-1,1]）
     ↓
随机采样时间步 t ∈ {0, …, 999} 与高斯噪声 ε
     ↓
ddpm.q_sample(x₀, t, ε) → 得到 xₜ
     ↓
ddpm.eps_model(xₜ, t) → 预测噪声 εθ
     ↓
MSE(εθ, ε)
     ↓
optimizer.zero_grad() → loss.backward() → optimizer.step()
```

`train.py` 中用 `Adam(lr=2e-4)` 优化模型；`t` 作为浮点数传给 U-Net，作为整数索引传给 `q_sample()`。每轮通过 `tqdm` 显示当前 Batch Loss 和按样本数加权的平均 Loss。

**日志与权重说明：** 这份原始 `train.py` 只在终端显示、打印 Loss，没有自行写入 CSV、绘制或保存 Loss 曲线；`ddpm.pth` 也是在 100 Epoch 全部结束后才保存。后续 20 Epoch 的继续训练属于另一个脚本，不应与这份初始训练代码混淆。

### 5.2 继续训练

在 `ddpm.pth` 的基础上，开展 Epoch 101–120 的继续训练实验；训练日志、Loss 曲线和新权重整理完成后再补充。

**Loss 曲线：待上传真实结果。**

### 5.3 图像生成

使用已保存的模型权重尝试 DDPM 采样，生成 16 张 32 × 32 的 RGB 图片。

**DDPM 生成样本：待上传。**

## 6. DDIM 采样复现

### 6.1 为什么加入 DDIM？

DDPM 的反向采样通常涉及较多迭代步骤。DDIM 的目的之一是**通过改变采样过程，以较少的步骤生成图像**。

本次复现没有重新训练一个 DDIM 专用网络，而是复用已经训练好的 DDPM 噪声预测 U-Net，重点观察**更换采样方法**所带来的变化。

### 6.2 实际修改了什么？

在提供的两个采样脚本中，主要改动是采样函数的调用。

**原 DDPM 采样：**

```python
imgs = ddpm.sample(
    shape=(16, 3, 32, 32),
    device=device
)
```

**修改为 DDIM 采样：**

```python
imgs = ddpm.ddim_sample(
    shape=(16, 3, 32, 32),
    device=device,
    steps=50
)
```

其余主要部分保持一致：

- 使用同样的 U-Net 结构；
- 加载同一份训练权重 `ddpm.pth`；
- 都生成 16 张 32 × 32 的 RGB 图片；
- 都将图片从 `[-1, 1]` 转换至 `[0, 1]` 后显示为 4 × 4 图片网格。

**需要区分的是：** `sample.py` 中看到的改动主要是更换函数调用，而真正决定 DDIM 是否正确执行 50 步采样的逻辑位于 `ddpm.py` 中的 `ddim_sample()`。因此不能仅凭调用方式，就认定内部已经正确实现论文的全部 DDIM 采样公式。

### 6.3 DDPM 与 DDIM 的区别

| 对比项 | DDPM | DDIM |
|---|---|---|
| 噪声预测网络 | U-Net | 可复用相同 U-Net |
| 本次使用的模型权重 | `ddpm.pth` | `ddpm.pth` |
| 采样调用 | `ddpm.sample()` | `ddpm.ddim_sample()` |
| 本次设置的采样步数 | 原始采样函数决定 | 50 |
| 是否为 DDIM 重新训练 | — | 否 |
| 主要区别 | 原始 DDPM 反向采样 | 更换反向采样过程 |

### 6.4 DDIM 实验结果

使用 `steps=50` 尝试生成 16 张图片。

**DDIM 50 步生成样本：待上传。**

后续可以在相同模型权重下记录两种采样方式的运行时间，并对比生成结果；目前不预先声称实际加速倍数或图像质量提升。

## 7. 实验分析与个人理解

**本次复现最重要的认识是：DDPM 的训练和采样并不是完全绑定的。**

DDIM 的学习重点不是重新设计 U-Net，而是研究如何利用同一个噪声预测网络，通过不同的采样更新方式生成图像。

从目前给出的采样脚本可以确认，两种方法使用相同的模型结构和权重，主要差别在于调用的采样函数以及 DDIM 指定了 `steps=50`。下一步需要结合 `ddpm.py` 检查 DDIM 采样公式，并通过实际图像、耗时和训练日志验证效果。

---

[← 返回代码复现目录](../) · [← 返回科研主页](../../README.md)
