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
| 方法 / CFG 设置 | 平均速度采样；训练模式未保存在权重中。CFG=1；max_cfg=7.5 依据用户已有实验记录 |
| 训练总 step / Batch size | 原始权重无元数据，无法核验；文件名提示 50,000 step |
| 总训练时间 / 真实 Loss | 本次未提供，不能由权重恢复 |
| 一步 / 五步耗时 | 本次评估未做独立采样速度基准，不把特征提取时间当作采样耗时 |
| 一步 / 五步 FID | **179.8574 / 17.9291** |
| FID 特征 | torch-fidelity 0.3.0，Inception-v3-compat 2048 维 |
| 随机种子 | 参考样本 42，噪声 43；两种采样共用类别和初始噪声 |
| Batch size | 生成 32；特征提取 64 |
| 实际依赖、图像哈希与索引 | [fid.json](results/uploaded-step-0050000-cfg1/fid.json) |
| 样本 | [一步](results/uploaded-step-0050000-cfg1/preview_1step.png) / [五步](results/uploaded-step-0050000-cfg1/preview_5step.png)，各展示最先生成的 100 张 |

## 观察与范围

本协议下，五步 FID 明显低于一步；预览图中五步数字轮廓也更清晰。该辅助指标不能单独证明类别正确或覆盖充分，也不能作为原论文 ImageNet 结果的复现。

这个权重与本目录 `mnist_laptop.py` 的结构不同；本次评估不证明它由本目录的训练脚本训练得到。原始训练模式、完整训练日志及 GPU 速度仍需另行记录。

与 DDPM/DDIM 的比较需要相同数据集、划分和评估设置；此 MNIST 分数不可直接对照仓库中的 CIFAR-10 实验。
