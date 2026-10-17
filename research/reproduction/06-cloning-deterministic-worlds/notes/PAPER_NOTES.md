# Cloning Deterministic Worlds（CVPR 2026）论文学习笔记

## 一句话总结

世界模型不仅要生成“看起来像”的未来画面，还要根据当前环境及动作，在长时间滚动预测中保持状态一致。作者指出：**潜在空间几何表示质量**会直接影响预测稳定性。

## 研究任务

给模型前若干帧迷宫观测与智能体动作，先编码为潜在状态，再用扩散动力学模型预测未来状态，最后解码为像素画面；与真实未来画面比较逐步 MSE。

## 对比实验

- VAE baseline：主要学习重建图像。
- GRWM：在相同类型世界模型管线上，改进状态编码器和潜在空间的几何结构。
- 两者分别预训练编码器、提取潜在数据、训练扩散动力学模型、计算像素级 rollout MSE。

## 创新点（组会用语）

1. **诊断问题**：长期预测误差的一部分来自“表示不适合动力学建模”，并非仅仅是扩散模型不够强。
2. **几何正则化表示**：结合历史时序观测，并用 temporal slowness 与 latent uniformity 约束潜在空间，鼓励时间局部连续、整体不过度塌缩。
3. **跨动力学骨干验证**：通过原始表示和 GRWM 表示的受控对照，检验潜在几何对长期滚动预测的作用。

## 扩散模型具体是哪一种？

在官方 Maze9 开源实现中，扩散动力学实现基于 DFOT，主干为 3D U-Net；配置使用 **1000 个离散扩散训练时间步、cosine 噪声日程、`pred_v` 目标**，生成时配置 10 次 DDIM 采样、`eta=0`。这属于潜在空间上的条件时序扩散预测，不是 MeanFlow 或 Flow Matching；`pred_v` 也不等同于 MeanFlow 的平均速度。

## 核心图该怎么看？

- **多步误差曲线**：横轴未来预测步数，纵轴归一化像素均方误差（MSE）。比较 VAE 与 GRWM 的预测质量和随步数累积的误差。
- **潜在聚类可视化**：将潜在特征按 KMeans 聚类，投到真实迷宫平面坐标，检验不同状态的表示是否具有可解释空间结构。
- **生成画面对比**：在相同起点和动作序列下比较真实画面、VAE baseline、GRWM 生成结果。

## 本复现和原论文的关系

本目录只提供**官方代码固定版本的获取、单卡顺序执行适配、结果导出和笔记**。研究创新模块并未重新实现；核心神经网络、损失函数和扩散采样均来自官方 Git 仓库。`debug` 仅用于验证流程，不能视为严格论文复现。报告数值必须用自己真实训练结果。

## 来源

- [论文 PDF（CVPR）](https://openaccess.thecvf.com/content/CVPR2026/papers/Xia_Cloning_Deterministic_Worlds_The_Critical_Role_of_Latent_Geometry_in_CVPR_2026_paper.pdf)
- [官方代码](https://github.com/XiaFire/Clone_Deterministic_Environment)
- [Maze9 数据](https://huggingface.co/datasets/xiazaishuo/cloning-deterministic-worlds-maze9)
