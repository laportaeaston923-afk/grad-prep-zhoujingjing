# DDPM · CIFAR-10 代码复现

## 1. 实验介绍

本实验基于 PyTorch 实现去噪扩散概率模型（DDPM），使用 CIFAR-10 数据集进行图像生成实验。

**参考论文：**

Denoising Diffusion Probabilistic Models  
Ho et al., NeurIPS 2020

[论文地址](https://arxiv.org/abs/2006.11239)

## 2. 代码来源与复现说明

本项目为 DDPM 的学习与复现实验，基础代码参考
B站上的 DDPM 教学视频。

**参考资料：**

- 原始论文：[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- 视频教程：[Bilibili · DDPM 代码讲解与复现](https://www.bilibili.com/video/BV1vZi6BaEgF/)

**复现说明：**

本项目基于上述视频教程中的实现进行学习与复现，
重点理解 DDPM 的前向扩散、反向去噪、
U-Net 噪声预测以及模型训练过程。

在参考代码的基础上，我进行了以下实验：

1. 使用 CIFAR-10 数据集训练 DDPM，共 100 Epoch。
2. 保存训练后的模型权重 `ddpm.pth`。


本项目主要用于扩散模型的原理学习和实践验证，
不将参考实现声明为独立原创代码。

## 3. 核心实现

- DDPM 前向加噪过程
- U-Net 噪声预测网络
- DDPM 反向去噪采样
- CIFAR-10 数据集训练
- 加载已有权重继续训练
- Loss 曲线与生成结果可视化

## 4. 实验配置

| 参数 | 设置 |
|---|---|
| 数据集 | CIFAR-10 |
| 模型 | DDPM + U-Net |
| 图像大小 | 32 × 32 |
| Batch Size | 256 |
| Learning Rate | 2e-4 |
| 初始训练 | 100 Epoch |
| 总训练轮数 | 120 Epoch |

## 5. 实验结果

### 5.1 训练 Loss 曲线


**实验曲线：**

### 5.2 生成图像



## 6. 实验分析




