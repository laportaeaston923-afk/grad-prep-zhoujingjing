# 论文名称

**Mean Flows for One-step Generative Modeling（MeanFlow）**

## 1. 核心内容

MeanFlow 是一种单步生成（One-step Generation）方法。它学习一段时间区间内的**平均速度**，而不是某一时刻的瞬时速度，希望直接从噪声生成数据。

## 2. 已有方法有什么问题？

已有方法大多需要多次采样或积分步骤：

- **DDPM：** 从噪声出发，通过多步去噪生成图像。
- **DDIM：** 跳过部分时间步，减少采样步骤，但仍需多次网络推理。
- **Flow Matching：** 学习瞬时速度，通常需要通过 ODE 多步积分生成数据。

## 3. 核心思想

**MeanFlow 学习一个时间区间内的平均速度，从而实现单步生成。**

相比 DDPM、DDIM 的逐步去噪和 Flow Matching 的多步积分，它希望减少生成时所需的网络调用次数。

## 4. 整体框架

### （1）不同生成方法的采样流程

```text
DDPM 噪声 → 去噪 → 去噪 → …… → 图像
DDIM 噪声 → 少量去噪步骤 → …… → 图像
Flow Matching 噪声 → 多次沿瞬时速度移动 → 图像
MeanFlow 噪声 ─────────────→ 图像
```

### （2）核心公式

<img width="247" height="66" alt="image" src="https://github.com/user-attachments/assets/fbfa3637-54bd-4c2c-861e-39db19e2ece7" />

原笔记中的公式图片保留在此，后续可以继续添加公式推导和实验分析。

## 5. 总结与我的理解

MeanFlow 关注**生成效率**：通过学习平均速度，尝试直接从噪声到达数据，从而避免常规多步采样带来的推理开销。

我的理解是：与学习每一时刻的运动速度不同，MeanFlow 更强调一个时间区间的整体变化。
