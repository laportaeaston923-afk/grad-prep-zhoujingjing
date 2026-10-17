# Paper Notes · 论文阅读笔记

这里记录论文的核心问题、方法结构、关键图、个人理解和复现边界。阅读时不只记“模型做了什么”，还要追问：**输入是什么、模型实际生成什么、生成结果如何验证、它怎样帮助最终任务？**

## 01 · 论文列表

| 编号 | 论文 / 方法 | 主要问题 | 阅读笔记 |
|---|---|---|---|
| 01 | CLAR | WiFi CSI 数据增强与人体活动识别 | [01-CLAR](01-CLAR.md) |
| 02 | Latent Diffusion Models（LDM） | 为什么在潜在空间进行扩散 | [02-latent diffusion model](02-latent%20diffusion%20model.md) |
| 03 | LatentCSI | 从 WiFi CSI 生成 RGB 图像 | [03-LatentCSI](03-LatentCSI.md) |
| 04 | WiFi-Diffusion | 从极少无线采样估计细粒度 Radio Map | [04-wifi diffusion](04-wifi%20diffusion.md) |
| 05 | MeanFlow | 学习平均速度，实现一步生成 | [05-meanflow](05-meanflow.md) |
| 06 | Echo-Conditioned DDPM | 从 RF 回波预测未来目标状态并选择感知波束 | [06-Echo-Conditioned Denoising Diffusion](06-Echo-Conditioned%20Denoising%20Diffusion.md) |
| 07 | mmWaveFlow | 用 Flow Matching 统一稀疏 / 稠密点云的双向转换 | [07-mmWaveFlow](07-mmWaveFlow.md) |
| 08 | cDDIM | 根据用户位置生成 MIMO 信道样本 | [08-cDDIM](08-cDDIM.md) |

## 02 · 横向比较：模型究竟生成什么？

| 论文 / 方法 | 生成或预测对象 | 生成模型的作用 | 最终要验证什么 |
|---|---|---|---|
| CLAR | WiFi CSI 样本 | 数据增强，辅助对比学习 | 活动识别性能是否提升 |
| WiFi-Diffusion | Radio Map | 从稀疏采样恢复细粒度无线环境地图 | 地图估计质量与物理合理性 |
| Echo-Conditioned DDPM | 下一时间块的目标状态 | 概率预测未来角度与距离 | 跟踪误差与波束选择效果 |
| cDDIM | 位置条件下的 MIMO 信道 | 扩充信道训练数据 | 信道结构、样本多样性与下游通信性能 |
| mmWaveFlow | 稀疏或稠密人体点云 | Flow Matching 学习双向传输 | 点云几何质量与下游人体感知效果 |
| MeanFlow | 数据样本 | 减少生成所需的网络调用次数 | 生成质量与推理效率的权衡 |

## 03 · 阅读时的固定问题

1. **问题是什么？** 数据不足、观测稀疏、预测不确定，还是生成成本太高？
2. **输入与输出分别是什么？** 条件信息是什么，模型最终输出什么？
3. **训练目标是什么？** 预测噪声、速度场、特征、状态，还是直接回归结果？
4. **关键模块分别解决什么问题？** 不把所有组件笼统地说成“负责生成”。
5. **证据是否充分？** 区分论文结果、个人复现结果、定性可视化和下游任务验证。
6. **还存在哪些边界？** 例如数据划分、OOD 泛化、样本多样性、计算成本与复现难度。

## 04 · 我的研究观察

不同论文虽然都使用生成建模，但它们的任务并不相同：cDDIM 通过条件扩散生成通信信道，mmWaveFlow 则使用 Flow Matching 在两种点云分布之间进行双向传输。不能只因为两者都能生成数据，就把它们视为同一种模型或同一个研究问题。

对无线方向而言，我希望继续关注一条共同主线：**生成模型产生的数据或状态，是否真的改善了后续的通信或感知任务？** 这比单独展示一张“看起来合理”的生成结果图更有说服力。

[返回科研主页](../README.md) · [返回仓库首页](../../README.md)
