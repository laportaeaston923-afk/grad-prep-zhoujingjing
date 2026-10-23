# Paper Notes · 论文阅读笔记


## 01 · 论文列表

| 编号 | 论文 / 方法 | 正式出处 | 主要问题 | 论文原文 | 阅读笔记 |
|---|---|---|---|---|---|
| 01 | **CLAR** · Diffusion Model-Based Contrastive Learning for Human Activity Recognition | IEEE Internet of Things Journal, 2024 | WiFi CSI 数据增强与人体活动识别 | [IEEE Xplore](https://doi.org/10.1109/JIOT.2024.3429245) · [arXiv](https://arxiv.org/abs/2408.05567) | [01-CLAR](01-CLAR.md) |
| 02 | **Latent Diffusion Models (LDM)** · High-Resolution Image Synthesis with Latent Diffusion Models | CVPR 2022 | 为什么在潜在空间进行扩散 | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html) · [arXiv](https://arxiv.org/abs/2112.10752) | [02-LDM](02-latent%20diffusion%20model.md) |
| 03 | **LatentCSI** · High-resolution efficient image generation from WiFi CSI using a pretrained latent diffusion model | IEEE GLOBECOM Workshops 2025 | 从 WiFi CSI 生成可控 RGB 环境图像 | [IEEE Xplore](https://doi.org/10.1109/GCWkshps68340.2025.11591056) · [arXiv](https://arxiv.org/abs/2506.10605) | [03-LatentCSI](03-LatentCSI.md) |
| 04 | **WiFi-Diffusion** · Achieving Fine-Grained WiFi Radio Map Estimation With Ultra-Low Sampling Rate | IEEE Journal on Selected Areas in Communications, 2025 | 从极少无线采样估计细粒度 Radio Map | [IEEE Xplore](https://doi.org/10.1109/JSAC.2025.3584562) · [arXiv](https://arxiv.org/abs/2503.12004) | [04-WiFi-Diffusion](04-wifi%20diffusion.md) |
| 05 | **MeanFlow** · Mean Flows for One-step Generative Modeling | NeurIPS 2025（Oral） | 学习平均速度，实现一步生成 | [NeurIPS](https://papers.neurips.cc/paper_files/paper/2025/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2505.13447) | [05-MeanFlow](05-meanflow.md) |
| 06 | **Echo-Conditioned DDPM** · Echo-Conditioned Denoising Diffusion Probabilistic Models for Multi-Target Tracking in RF Sensing | arXiv preprint, 2025 | 从 RF 回波预测未来目标状态并选择感知波束 | [arXiv](https://arxiv.org/abs/2510.25464) | [06-Echo-Conditioned DDPM](06-Echo-Conditioned%20Denoising%20Diffusion.md) |
| 07 | **mmWaveFlow** · Unified Enhancement and Generation of mmWave Human Point Clouds | CVPR 2026 | 用 Flow Matching 统一稀疏 / 稠密点云的双向转换 | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Su_mmWaveFlow_Unified_Enhancement_and_Generation_of_mmWave_Human_Point_Clouds_CVPR_2026_paper.html) | [07-mmWaveFlow](07-mmWaveFlow.md) |
| 08 | **cDDIM** · Generating High Dimensional User-Specific Wireless Channels using Diffusion Models | IEEE Transactions on Wireless Communications, 2026（online 2025） | 根据用户位置生成 MIMO 信道样本 | [IEEE Xplore](https://ieeexplore.ieee.org/document/11142589/) · [arXiv](https://arxiv.org/abs/2409.03924) | [08-cDDIM](08-cDDIM.md) |
| 09 | **ELF** | 上传笔记未记录完整出处 | 连续语言向量生成与最终 Token 解码 | — | [09-ELF](09-ELF.md) |
| 10 | **CSI-BERT2** | 上传笔记未记录完整出处 | CSI 恢复、未来预测与感知分类 | — | [10-CSI-BERT2](10-CSI-BERT2.md) |
| 11 | **SRLoc** | 上传笔记未记录完整出处 | 对抗 RSSI 修复与鲁棒 WiFi 指纹定位 | — | [11-SRLoc](11-SRLoc.md) |

## 02 · 核心内容

| 论文 / 方法 | 生成或预测对象 | 生成模型的作用 | 最终要验证什么 |
|---|---|---|---|
| CLAR | WiFi CSI 样本 | 数据增强，辅助对比学习 | 活动识别性能是否提升 |
| WiFi-Diffusion | Radio Map | 从稀疏采样恢复细粒度无线环境地图 | 地图估计质量与物理合理性 |
| Echo-Conditioned DDPM | 下一时间块的目标状态 | 概率预测未来角度与距离 | 跟踪误差与波束选择效果 |
| cDDIM | 位置条件下的 MIMO 信道 | 扩充信道训练数据 | 信道结构、样本多样性与下游通信性能 |
| mmWaveFlow | 稀疏或稠密人体点云 | Flow Matching 学习双向传输 | 点云几何质量与下游人体感知效果 |
| MeanFlow | 数据样本 | 减少生成所需的网络调用次数 | 生成质量与推理效率的权衡 |
| ELF | 连续语言嵌入与文本 Token | Flow Matching 生成语言向量，最后解码 | 文本质量、多样性与采样效率 |
| CSI-BERT2 | 缺失或未来 CSI、感知类别 | BERT 掩码恢复与预测、分类 | 恢复与预测误差、分类性能 |
| SRLoc | 受攻击的 RSSI 指纹 | 受扩散启发的单步条件重建 | 修复效果与最终定位误差 |

共 11 篇阅读笔记。新增三篇保留上传笔记的图片、表格与讲解；完整出版信息待核对补充。

[返回科研主页](../README.md) · [返回仓库首页](../../README.md)
