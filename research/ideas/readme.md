# 研究想法 · 扩散模型 × 无线感知

每条 idea 单独保存，点击题目查看完整方案。以下均为待验证设想，尚不代表已完成的实验或已确认的创新点。

| 编号 | 研究想法 | 关联阅读 |
|---|---|---|
| 01 | [面向接收机缺失的条件扩散 DFS 补全](01-dfs-completion.md) | Widar3.0 + DDPM/DDIM |
| 02 | [面向任务的信息保持型 CSI 数据增强](02-task-aware-csi-augmentation.md) | CLAR |
| 03 | [快速生成位置条件信道](03-fast-position-conditioned-channel.md) | cDDIM + MeanFlow |
| 04 | [根据生成不确定性主动采样 Radio Map](04-active-radio-map-sampling.md) | WiFi-Diffusion |
| 05 | [CSI-to-Image 对真实观测的依赖](05-csi-image-faithfulness.md) | LatentCSI + LDM |
| 06 | [面向波束选择的低成本概率状态预测](06-efficient-beam-state-prediction.md) | Echo-Conditioned DDPM + MeanFlow |
| 07 | [利用时间约束改善毫米波点云增强](07-temporal-point-cloud-enhancement.md) | mmWaveFlow |

## 建议推进顺序

1. **先做 Idea 03 的 DDIM 步数基线：** 接近现有 cDDIM 阅读与实验记录，容易先得到可讨论的对照结果；验证后再决定是否投入条件 MeanFlow。
2. **再从 Idea 01 或 02 选一个感知任务：** 前者接已有 Widar 处理链，后者接 CLAR 的增强与识别逻辑，均需先解决多样本数据。
3. **Idea 04 作为主动采样备选；05–07 先核对复现条件：** 分别依赖配对视觉数据、闭环仿真或连续点云数据。

上述顺序是基于当前仓库基础的实施建议，不是对论文发表潜力的保证。每个方向先用简单基线回答研究问题，再判断复杂生成模型是否值得引入。


[论文笔记](../paper-notes/README.md) · [科研总览](../README.md) · [仓库首页](../../README.md)
