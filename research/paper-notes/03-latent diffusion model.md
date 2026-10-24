# 论文名称

**High-Resolution Image Synthesis with Latent Diffusion Models（LDM）**

- **作者：** Robin Rombach、Andreas Blattmann、Dominik Lorenz、Patrick Esser、Björn Ommer
- **出处：** *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 10684–10695
- **论文：** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html) · [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) · [DOI](https://doi.org/10.1109/CVPR52688.2022.01042)
- **官方代码：** [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)
- **方法关键词：** Latent Diffusion、VAE、潜在空间、Cross-Attention、条件图像生成、高分辨率合成

## 1. 核心内容

将图像生成、Diffusion Model、VAE 和 Latent Space 结合起来，在压缩后的潜在空间中进行扩散生成。

## 2. 已有方法有什么问题？

DDPM 直接在原始图片像素空间中加噪和去噪：

```text
原始图片 → 不断加噪 → 噪声 → 逐步去噪 → 生成图片
```

- **计算量大：** 高分辨率图片的像素维度很高，直接在像素空间进行 Diffusion 需要较多计算。
- **生成成本高：** 随分辨率提高，训练与采样开销增大。

## 3. 核心思想

**先压缩 → 在隐空间做 Diffusion → 再还原。**

LDM 不在原始 Image Space 中进行扩散，而是先通过 Encoder 将图片压缩到低维 Latent Space，再在 Latent Space 中执行扩散与去噪。

主要优势：

1. 在更小的空间中进行扩散，减少计算量。
2. 相比直接在像素空间计算，更适合高分辨率图片生成。
3. 降低模型训练与生成所需的计算和显存开销。

## 4. 整体框架

### （1）Encoder 压缩图片

```text
Image → Encoder → Latent
```

Encoder 将原始图像变成维度更低的 Latent 表示。

### （2）在 Latent Space 中进行 Diffusion

前向加噪：

```text
z₀ → z₁ → z₂ → …… → zₜ
```

模型学习反向去噪：

```text
zₜ → zₜ₋₁ → zₜ₋₂ → …… → z₀
```

**DDPM 与 LDM 的主要区别：** 前者直接在 Image Space 中执行扩散，后者在压缩的 Latent Space 中执行扩散。

### （3）Decoder 将 Latent 还原成图片

```text
生成的 Latent → Decoder → Image
```

## 5. 总结与我的理解

LDM 的核心思想是：**使用 Encoder 将图片压缩到 Latent Space，在潜在空间进行 Diffusion，最后用 Decoder 将生成的 Latent 还原成图片。**

我的理解是：LDM 并没有取消扩散模型的加噪与去噪过程，而是改变了扩散发生的空间，以降低高分辨率图像生成的成本。
