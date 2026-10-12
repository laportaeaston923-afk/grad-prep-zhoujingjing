# 代码复现 · DDPM、DDIM 与 MeanFlow

三个独立实验目录：DDPM 训练噪声预测网络；DDIM 复用 DDPM 权重改变采样方式；MeanFlow 单独学习平均速度，研究一步生成。

[DDPM](01-ddpm-cifar10/) · [DDIM](02-ddim-cifar10/) · [MeanFlow](03-meanflow-mnist/) · [科研总览](../README.md) · [仓库首页](../../README.md)

## 01 · 项目总览

| 项目 | 数据集 | 实验目标 | 当前内容 |
|---|---|---|---|
| [DDPM](01-ddpm-cifar10/) | CIFAR-10 | 理解前向加噪、噪声预测与反向采样 | 120 Epoch AMP 训练代码；真实 Loss 曲线图、生成样本和去噪过程图 |
| [DDIM](02-ddim-cifar10/) | CIFAR-10 | 复用同一网络，研究不同采样方式 | 50 步采样代码与生成结果图；checkpoint 关联及统一耗时对照仍待核对 |
| [MeanFlow](03-meanflow-mnist/) | MNIST | 学习平均速度，比较一步/五步与 CFG 设置 | 50,000 step 原日志、配置、Loss CSV/曲线；过程图/GIF；四组 FID 与 CFG 并排对照 |

DDPM 的最终 Loss 展示采用 `train_full_120_amp.py` 对应的完整 120 Epoch AMP 实验，不合并旧的继续训练日志。DDPM/DDIM 的结果图已嵌入各自 README；完整原始 Loss CSV、评估协议和统一速度记录尚未在这两个目录中保存。

## 02 · 最新结果：MeanFlow

补充上传的日志和配置标注运行 `mnist_meanflow_50k`、模式 `meanflow`，记录到 50,000 step。训练 batch=2，配置精度为 BF16；原训练硬件未在日志中单独记录。FID 评估另用 CPU / FP32。

### 训练 Loss

每 500 step 记录一次区间平均指标，从 step 500 到 50,000，共 100 个记录点；按日志间隔没有缺失，总 Loss **1.985971 → 1.856935**。曲线未平滑，也未补造数据。

[Loss 曲线、原日志、配置与 CSV](03-meanflow-mnist/results/mnist-meanflow-50k-training/)。

### FID 与 CFG 对照

| 采样 | NFE | 无额外引导 CFG=1，FID ↓ | 有引导 CFG=4.25，FID ↓ |
|---|---:|---:|---:|
| 一步 | 1 | 179.8574 | **150.9314** |
| 五步 | 5 | **17.9291** | 35.2649 |

评估协议：同一个上传权重、相同噪声与数字类别，每个设置 10,000 张生成图，对比同一组 10,000 张 MNIST test；torch-fidelity 0.3.0、Inception-v3-compat 2048 维，CPU / FP32。

**本次 CFG=4.25 使一步 FID 降低 16.08%，却使五步 FID 升高 96.69%。** 两组都有数字类别条件；这是同一模型的推理引导强度对照，没有分别训练“有 CFG 蒸馏”和“无 CFG 蒸馏”两个模型。该实现把引导学入模型，因此两种强度仍为 1 / 5 NFE。

MeanFlow 的核心目标是一步生成，五步只作补充对照。较低的五步 FID 不能代替一步复现结果；图像轮廓更清晰也不等于整批图像的特征分布更接近真实数据。

### 结果导航

| 内容 | 入口 |
|---|---|
| 训练日志、原配置、Loss 曲线与 CSV | [50,000 step 真实训练记录](03-meanflow-mnist/results/mnist-meanflow-50k-training/) |
| CFG=1 的一步/五步 FID 与样本 | [基准评估](03-meanflow-mnist/results/uploaded-step-0050000-cfg1/) |
| 噪声 → 图片的静态图与 GIF | [一步 / 五步生成过程](03-meanflow-mnist/results/uploaded-step-0050000-process/) |
| CFG 并排图片、过程动图与完整 FID 协议 | [CFG 对照实验](03-meanflow-mnist/results/uploaded-step-0050000-cfg-comparison/) |
| 训练、采样、评估与复测说明 | [MeanFlow 项目首页](03-meanflow-mnist/) |

## 03 · DDPM 与 DDIM 的关系

DDPM 训练 U-Net 预测噪声并保存权重；DDIM 可以复用同一噪声预测网络，改变反向更新公式与采样时间步序列，因此不需要另行训练 U-Net。

| 项目 | DDPM | DDIM |
|---|---|---|
| 数据集 / 图片 | CIFAR-10，RGB 32×32 | 同左 |
| 网络 | U-Net | 复用 DDPM 网络 |
| 训练 | 最终采用 120 Epoch AMP 实验 | 不单独训练 |
| 采样调用 | `ddpm.sample()` | `ddpm.ddim_sample(..., steps=50)` |
| 当前展示 | Loss、生成图、去噪过程 | 50 步生成结果图 |
| 待补齐的对照 | 与 DDIM 使用同一 checkpoint、相同设备和精度，记录耗时与质量 | 同左 |

当前两者的 README 已展示结果图，代码也已保存；DDIM 使用的具体 checkpoint 版本、内部采样公式核验及实际加速幅度仍以项目记录为准。

## 04 · 已上传代码与文件

| 目录 | 主要文件 | 结果位置 |
|---|---|---|
| [01-ddpm-cifar10/](01-ddpm-cifar10/) | `ddpm.py`、`ddpm2.py`、`unet.py`、`train.py`、`train_full_120_amp.py`、`sample.py` | README 嵌入 Loss、生成样本与去噪过程图 |
| [02-ddim-cifar10/](02-ddim-cifar10/) | `ddpm.py`、`unet.py`、`train.py`、`sample.py` | README 嵌入采样结果图 |
| [03-meanflow-mnist/](03-meanflow-mnist/) | 参考模型与训练代码、`train_repro.py`、`sample_repro.py`、`eval_fid.py`、`plot_training_loss.py`、`visualize_sampling.py`、`compare_cfg_visuals.py`、配置与流程检查 | `results/` 保存原日志、配置、Loss、PNG/GIF 与评估 JSON |

三个目录分别保存代码。大型权重和数据集不随结果图上传；复测时使用与记录中哈希对应的权重。

## 05 · 复现范围与比较原则

- DDPM/DDIM 的 CIFAR-10 与 MeanFlow 的 MNIST 属于不同数据集，不能直接比较现有 FID 或耗时来证明哪种方法更优。
- 比较采样速度时，应统一设备、图片尺寸、batch、精度、计时范围和 checkpoint；GIF 播放时间不代表计算耗时。
- MeanFlow 采用第三方 PyTorch 实现，本次实测日志标注为 `meanflow`。参考代码也提供 iMF 分支，但当前结果不构成原 MeanFlow 与 iMF 的严格两模型消融实验，也不声明达到原论文 ImageNet 成绩。
- 已有结果以原日志、配置与评估 JSON 为准；Loss 下降、单张图片清晰或更强引导均不能单独证明整体生成质量。

## 06 · 参考资料

- [DDPM 原始论文（NeurIPS 2020）](https://arxiv.org/abs/2006.11239)
- [DDIM 原始论文（ICLR 2021）](https://arxiv.org/abs/2010.02502)
- [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447)
- [MeanFlow 第三方参考代码](https://github.com/haidog-yaqub/MeanFlow)
- [DDPM 学习参考：Bilibili 教学视频](https://www.bilibili.com/video/BV1vZi6BaEgF/)

[← 返回科研总览](../README.md) · [← 返回仓库首页](../../README.md)
