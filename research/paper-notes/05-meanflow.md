# 论文名称
Mean Flows for One-step Generative Modeling
## 1. 核心内容
MeanFlow 是一种单步生成（One-step Generation）方法。
与 DDPM 和 DDIM 的多步去噪不同，MeanFlow 学习的是一段时间区间内的平均速度，而不是瞬时速度。

- DDPM：多步去噪，从噪声逐步生成图像。
- DDIM：通过跳过部分时间步减少采样步骤，但仍需要多次网络推理。
- Flow Matching：学习瞬时速度，生成时需要通过 ODE 进行多步积分。
- MeanFlow：学习平均速度，可以直接从噪声跳到数据，实现单步生成。

