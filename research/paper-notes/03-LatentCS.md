# 论文名称
High-resolution efficient image generation from WiFi CSI using a pretrained latent diffusion model
## 1. 核心内容

利用 WiFi CSI 重建一个高维的、可视化的、可控制的图像表示

## 2. 已有方法有什么问题？
- 直接在像素空间生成图像：
```text
CSI-->Generator-->RGB Image
```
也就是 CSI 直接对应到 RGB 像素。

因此直接在像素空间进行图像生成的计算成本比较高。
- GAN

<img width="261" height="103" alt="image" src="https://github.com/user-attachments/assets/fc3fa4cf-380c-4f34-8a25-0f5b5eed7218" />


- `Generator` 和 `Discriminator` 不断进行对抗训练
因此存在的问题包括：
- 训练过程更加复杂；
- Generator 和 Discriminator 需要相互博弈；
- 训练稳定性比较困难；
- 需要对抗训练相关的参数调整；
- 高分辨率像素空间生成的计算成本较高。

## 3. 核心思想
不让 CSI 直接生成大量 RGB 像素，而是先把 CSI 映射到 Stable Diffusion 的 latent 空间，再利用预训练的 Latent Diffusion Model 完成图像生成。
## 4. 整体框架

**`推理 / 采样过程`**

<img width="391" height="236" alt="image" src="https://github.com/user-attachments/assets/aabc3904-64c7-48bc-a59c-9bc76bea661d" />

- `Text` 的作用更接近：提供额外的语义控制

**`训练过程`**

<img width="381" height="250" alt="image" src="https://github.com/user-attachments/assets/1c31520b-a997-4ff8-9a17-e9e2a3c2ac54" />

- z：真实图片经过预训练 VAE Encoder 得到的 latent；
- ẑ：CSI Encoder 根据 CSI 预测得到的 latent。
计算 latent MSE：L = ||ẑ - z||²
训练目标：让 CSI 生成的 latent ẑ 尽可能接近真实图片经过 VAE 得到的 latent z。

## 5. 我的理解
它不是要把 CSI 当成摄像机，精确恢复原始照片
先利用一个轻量 CSI Encoder，把 CSI 中包含的环境信息映射到 Stable Diffusion 的 latent 空间，再利用预训练的 Latent Diffusion Model 将这个 latent 转换成高分辨率 RGB 图像，从而把无线信号转换成一种高维、可视化的环境表示。
