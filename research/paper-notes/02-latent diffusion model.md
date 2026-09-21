# High-Resolution Image Synthesis with Latent Diffusion Models

## 研究方向

图像生成 + Diffusion Model + VAE + Latent Space

---

## 相对于 DDPM 的好处
DDPM 是直接在原始图片的像素空间进行扩散和去噪：
原始图片->不断加噪->噪声->逐步去噪->生成图片
但是图片的像素空间维度很高，尤其是高分辨率图片，直接在像素空间进行 Diffusion 会产生很大的计算量。
LDM 不在原始图片空间做 Diffusion，而是先把图片压缩到一个更小的隐空间（Latent Space），再在隐空间中进行 Diffusion。

## 优势
1. **计算量更小**
原始图片的像素数量很多，直接做 Diffusion 计算量比较大。
LDM 先把图片压缩到 Latent Space，在更小的空间中进行 Diffusion，因此可以降低计算量。
2. **更适合高分辨率图片生成**
图片分辨率越高，像素空间就越大。
DDPM 直接在像素空间进行计算会越来越困难，而 LDM 可以先压缩图片，再在 Latent Space 中进行 Diffusion，因此更适合高分辨率图像生成。
3. **训练和生成效率更高**
因为 Diffusion 不再直接处理大量像素，而是在压缩后的 Latent Space 中工作，所以能够减少计算和显存开销。

---

## 整个模型怎么运行
LDM 可以理解成：
**先压缩 → 在隐空间做 Diffusion → 再还原**

### 第一步：Encoder 压缩图片
输入一张真实图片：
Image->Encoder->Latent
Encoder的作用就是把原始图片压缩成一个维度更低的 Latent 表示。

---

### 第二步：在 Latent Space 中进行 Diffusion

得到 Latent 后，就和 DDPM 一样进行扩散过程。

#### 前向过程

不断向 Latent 中加入噪声
z₀ → z₁ → z₂ → …… → zₜ

---

### 第三步：训练模型学习反向去噪
zₜ → zₜ₋₁ → zₜ₋₂ → …… → z₀
和 DDPM 最大的区别就在于：
DDPM 是在 Image Space 里面做这个过程，而 LDM 是在 Latent Space 里面做这个过程。

---

### 第四步：Decoder 把 Latent 还原成图片

---

## LDM 的理解

DDPM 的问题主要是：

**直接在像素空间做 Diffusion，计算量太大。**

LDM 的核心思想：

**先用 Encoder 把图片压缩到 Latent Space，再在 Latent Space 中进行 Diffusion，最后用 Decoder 把生成的 Latent 还原成图片。**

所以可以简单理解为：

**DDPM：在图片上直接生成图片**

**LDM：先把图片压缩成 Latent，在 Latent 上生成，再把 Latent 还原成图片。**

因此 LDM 相对于 DDPM 最大的优势就是：

**降低 Diffusion 的计算成本，让高分辨率图像生成更加可行。**
