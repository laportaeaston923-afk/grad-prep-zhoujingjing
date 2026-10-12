# cDDIM · 无线信道生成

复现论文 *Generating High Dimensional User-Specific Wireless Channels Using Diffusion Models*（IEEE TWC）。

作者代码：[taekyunl/cDDIM](https://github.com/taekyunl/cDDIM) · [论文](https://arxiv.org/abs/2409.03924)

## 目录

- `code/`：条件 U-Net、DDIM、训练与 Fig. 5/6/7 评估代码（本机完整项目见下方说明）。
- `results/`：本次 Windows + CUDA 实验的 Loss 与 Fig. 5/6/7 结果图。

## 复现设置

QuaDRiGa BERLIN_UMa_LOS、28 GHz；4×32 MIMO 信道矩阵，用户三维位置作为条件。训练使用 10,000 组信道，200 Epoch，约 62,578 参数的精简 U-Net，256 次确定性 DDIM 采样。此次实现依据论文和公开源码重写，不是原作者源码的逐行等价复现。

训练结束时 MSE Loss 约 **0.05043**。Fig. 5 风格图比较 5 组生成与参考信道；Fig. 6/7 分别使用 500 组 ID/OOD 测试位置，测量主发射波束索引误差的累积分布。

| 允许索引误差 D | ID（0–100 m） | OOD（100–200 m） |
|---|---:|---:|
| 0 | 约 27% | 约 25% |
| 1 | 约 66% | 约 54% |
| 2 | 约 80% | 约 69% |
| 3 | 约 86% | 约 76% |

**结果说明：** ID 整体优于 OOD，但分布外仍有一定主波束匹配能力。以上数值从实际运行截图近似读取，精确值应以本机保存的 CSV 为准。这只是 cDDIM 单模型阶段性实验，未完成原论文全部对比基线及下游任务。

## 运行方法

本机 Windows + Conda `deepStudy`、CUDA 版 PyTorch。完整的本机源码、训练数据、checkpoint 与原始 CSV 在 `E:\all\cDDIM_reproduction_for_E_all\cDDIM_reproduction`；大型 `.mat` 数据和 checkpoint 不提交 GitHub。

项目从 `code/` 目录运行，正式数据需通过 QuaDRiGa / 作者提供的数据集准备。基本顺序：先运行 `run_paper.bat` 完成训练，再运行 `run_fig6_id.bat`，在 MATLAB 生成 100–200 m OOD 数据后运行 `run_fig7_ood.bat`。

[查看实验图](results/) · [返回复现总览](../README.md)
