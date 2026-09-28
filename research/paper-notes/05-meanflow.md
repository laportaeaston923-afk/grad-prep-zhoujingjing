# 论文名称
Mean Flows for One-step Generative Modeling
## 1. 核心内容
MeanFlow 是一种单步生成（One-step Generation）方法。
与 DDPM 和 DDIM 的多步去噪不同，MeanFlow 学习的是一段时间区间内的平均速度，而不是瞬时速度。

- `DDPM`：多步去噪，从噪声逐步生成图像。
- `DDIM`：通过跳过部分时间步减少采样步骤，但仍需要多次网络推理。
- `Flow Matching`：学习瞬时速度，生成时需要通过 ODE 进行多步积分。
- `MeanFlow`：学习平均速度，可以直接从噪声跳到数据，实现单步生成。

```TEXT
DDPM 噪声 → 去噪 → 去噪 → …… → 图像
DDIM 噪声 → 少量去噪步骤 → …… → 图像
Flow Matching 噪声 → 多次沿瞬时速度移动 → 图像
MeanFlow 噪声 ─────────────→ 图像
```
## 2. 核心公式

<img width="247" height="66" alt="image" src="https://github.com/user-attachments/assets/fbfa3637-54bd-4c2c-861e-39db19e2ece7" />
