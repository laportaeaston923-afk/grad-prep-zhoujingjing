# MeanFlow / iMF 实验记录

## 当前状态

代码流程验证完成。2026-10-11 对用户上传的 `step_0050000.pt` 完成 MNIST FID 实测，没有重新训练模型。[完整评估、协议 JSON 与预览图](results/uploaded-step-0050000-cfg1/)。

## 上传权重评估

| 项目 | 实测记录 |
|---|---|
| 日期 | 2026-10-11 |
| 数据集 / 划分 | MNIST test，10,000 张真实图；每种采样 10,000 张生成图 |
| 设备 / 精度 | Linux CPU，8 线程；生成与 Inception 特征均 FP32 |
| 权重 | `step_0050000.pt`；SHA256 见结果 JSON；不将原始权重上传到仓库 |
| 模型结构 | 32×32 灰度图，patch=2、dim=256、depth=6、heads=4、10 类；7,831,304 参数 |
| 评估配置 | [configs/mnist_uploaded_50k.py](configs/mnist_uploaded_50k.py)；显式适配原始 state_dict |
| 方法 / CFG 设置 | 平均速度采样；补充日志和配置标注 `meanflow`。推理 CFG=1；max_cfg=7.5 已由上传训练配置确认 |
| 训练总 step / Batch size | 补充日志记录到 50,000 step；训练 batch=2。权重本身不含训练元数据 |
| 总训练时间 | 未单独计量；日志时间跨度不等于纯训练耗时 |\n| 真实 Loss | [100 个记录点与曲线](results/mnist-meanflow-50k-training/)；总 Loss 1.985971 → 1.856935 |\n| 训练配置精度 | 上传配置为 BF16；FID 评估另用 CPU / FP32，不代表原训练设备 |
| 一步 / 五步耗时 | 本次评估未做独立采样速度基准，不把特征提取时间当作采样耗时 |
| 一步 / 五步 FID | **179.8574 / 17.9291** |
| FID 特征 | torch-fidelity 0.3.0，Inception-v3-compat 2048 维 |
| 随机种子 | 参考样本 42，噪声 43；两种采样共用类别和初始噪声 |
| Batch size | 生成 32；特征提取 64 |
| 实际依赖、图像哈希与索引 | [fid.json](results/uploaded-step-0050000-cfg1/fid.json) |
| 样本 | [一步](results/uploaded-step-0050000-cfg1/preview_1step.png) / [五步](results/uploaded-step-0050000-cfg1/preview_5step.png)，各展示最先生成的 100 张 |

## CFG 推理设置对照

同一个上传权重，仅改变推理 CFG scale，噪声和类别相同；每个设置均为 10,000 张生成图 vs 同一 10,000 张 MNIST test。没有重新训练。

| 采样 | 无额外引导 CFG=1，FID ↓ | 有引导 CFG=4.25，FID ↓ |
|---|---:|---:|
| 1 步 | 179.8574 | 150.9314 |
| 5 步 | 17.9291 | 35.2649 |

一步 FID 改善 16.08%，五步 FID 变差 96.69%；此结果不支持“有 CFG 在所有采样步数下都更好”。两种引导设置仍各为 1 / 5 NFE。

[完整对照记录、样本和动图](results/uploaded-step-0050000-cfg-comparison/) · [数值汇总 JSON](results/uploaded-step-0050000-cfg-comparison/comparison.json)。

## 观察与范围

本协议下，五步 FID 明显低于一步；预览图中五步数字轮廓也更清晰。该辅助指标不能单独证明类别正确或覆盖充分，也不能作为原论文 ImageNet 结果的复现。

这个权重与本目录 `mnist_laptop.py` 的结构不同；本次评估不证明它由本目录的训练脚本训练得到。本次补充日志与配置标注 MeanFlow 模式，按每 500 step 的间隔完整记录到 50,000 step；原训练硬件和 GPU 速度仍未单独核验。

与 DDPM/DDIM 的比较需要相同数据集、划分和评估设置；此 MNIST 分数不可直接对照仓库中的 CIFAR-10 实验。
