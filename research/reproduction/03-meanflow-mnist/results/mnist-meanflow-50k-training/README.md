# MNIST MeanFlow · 50,000 step 训练记录

来源：用户于 2026-10-11 上传的 `train.log` 和 `config.py`。运行名为 `mnist_meanflow_50k`，日志头与配置均标注 `meanflow`。本页从真实日志提取指标，没有重新训练、平滑或补造 Loss。

## Loss 曲线

![真实训练 Loss 曲线](loss_curve.png)

| 指标 | Step 500（首个记录点） | Step 50,000（末个记录点） |
|---|---:|---:|
| 总 Loss | 1.985971 | 1.856935 |
| FM Loss | 0.992869 | 0.895081 |
| MF Loss | 0.993101 | 0.961854 |
| 原始 MF 速度 MSE | 1.514608 | 3.794731 |

记录点为 500、1,000、…、50,000，共 100 个，按配置 `log_step=500` 检查没有缺失。参考训练实现每次写入前 500 step 的区间平均值，因此这些不是每个 batch 的瞬时 Loss；横轴也不从虚构的 step 0 起算。

上图三部分分别是总 Loss、FM/MF 分项、原始速度 MSE。总 Loss 整体下降、后段仍有波动；原始 MSE 并未随总 Loss 单调下降，记录范围为 0.574472–8.941065。不同指标含义和尺度不同，不能把总 Loss 当作 FID，也不能据此认定一步生成已经达到高质量。

## 训练配置

| 项目 | 上传配置 |
|---|---|
| 数据集 | MNIST，灰度 32×32 |
| 模式 | `meanflow` |
| 训练 step / batch | 50,000 / 2 |
| 模型 | patch=2，dim=256，depth=6，heads=4，10 类 |
| 学习率 / warmup | 1e-4 / 2,500 step |
| 配置精度 | BF16 |
| CFG | max_cfg=7.5；cfg_ratio=0.10；cfg_scale=[1.0,7.5]；cfg_scale_ratio=0.75 |
| 自适应损失参数 | adaptive_l2_gamma=0.0，adaptive_l2_c=1e-2 |

BF16 是训练配置值；日志未单独记录训练 GPU、CUDA 版本和实际设备。已测 FID 使用 CPU / FP32，是另一阶段的评估设置。日志时间戳跨度包含潜在暂停与其他操作，不作为纯训练耗时。

用户作为已评估权重的配套证据上传这些文件；配置中的模型结构与该权重一致。纯 state_dict 内没有运行 ID，文件关联依据用户提供的来源，而非权重内部元数据。

## 保存的文件

- [train.log](train.log)：原始日志，逐字节保留。
- [config.py](config.py)：原始训练配置，逐字节保留。
- [loss.csv](loss.csv)：100 个真实记录点，可用于独立检查或绘图。
- [loss_curve.png](loss_curve.png)：三部分指标曲线。
- [loss_source.json](loss_source.json)：原日志与配置 SHA256、记录范围和来源说明。

从项目根目录复绘：

```bat
python plot_training_loss.py --input results/mnist-meanflow-50k-training/train.log --out results/loss-replot
```

[FID 与生成图片](../uploaded-step-0050000-cfg1/)：CFG=1，10,000 张生成图对 10,000 张 MNIST test；一步 FID **179.8574**，五步 **17.9291**。五步为补充对照，一步才是 MeanFlow 的核心生成目标。

[← 返回 MeanFlow 复现页面](../../README.md)
