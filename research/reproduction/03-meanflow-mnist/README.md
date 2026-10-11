# MeanFlow / iMF · MNIST 

基于 [haidog-yaqub/MeanFlow](https://github.com/haidog-yaqub/MeanFlow) 的第三方 PyTorch 实现，复现「训练平均速度模型 → 保存权重 → 一步生成」的完整流程。默认采用该仓库当前的 `i-meanflow` 模式，以 MNIST 为入门数据集。

**状态：代码流程已验证；上传权重的 MNIST FID 已实测：一步 179.8574、五步 17.9291（CFG=1，各 10,000 张）。** [查看评估记录与样本](results/uploaded-step-0050000-cfg1/)。本目录训练脚本的完整训练、RTX 5060 Laptop GPU 的 BF16 验证仍待完成。

## 1. 来源与复现范围

- 原论文：[Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447)。
- 后续论文：[Improved Mean Flows](https://arxiv.org/abs/2512.02012)。
- 参考代码：上述第三方仓库，固定 commit `97cc5182abe9d69e5dcb350268f603fbe31a5897`。
- 保留参考实现的 `meanflow.py`、`models/dit.py`、`data.py`、`train.py` 和原始 MNIST/CIFAR-10 配置，保留其 [MIT 许可](LICENSE)。参考源码不是本项目独立原创。
- 新增 `train_repro.py`、`sample_repro.py`、`repro_utils.py`、笔记本配置和流程验证脚本。

本实验是**第三方实现的流程复现**，不声明达到论文的 ImageNet FID 或原论文严格复现。当前参考代码使用双头 DiT、额外 FM loss，以及其自身的 JVP 计算方案。特别地，当前 `meanflow` 与 `i-meanflow` 两个分支在 `dudt` 被 detach 的处理下，其损失表达式代数等价；仅切换模式字符串不能作为原 MeanFlow 与 iMF 的有效对照实验。

## 2. 先跑 MNIST

先进行 100 step 的真实 MNIST 检查，然后从头训练 10,000 step：

| 设置 | 配置 | 上游 MNIST 配置 |
|---|---|---|
| 图片 | MNIST，1 × 32 × 32 | 同左 |
| 模型 | DiT，dim=256，depth=6 | DiT，dim=512，depth=12 |
| Patch size | 4 | 2 |
| Batch size | 32 | 64 |
| 总训练 step | 10,000 | 200,000 |
| Learning rate | 1e-4 | 1e-4 |
| Warmup step | 500 | 2,500 |
| 精度 | BF16；可选 FP32 | BF16 |
| CFG 蒸馏 | 关闭；仍使用数字类别条件 | CFG 范围 [1.0, 7.5] |
| DataLoader workers | 0，适合先排查 Windows 流程 | 8 |


## 3. Loss保存和恢复

新增脚本默认每 100 step 记录一次真实区间平均 Loss，每 500 step 保存 checkpoint 和一步样本。输出内容：

- `config.json`：实际采用的配置，包括命令行覆盖。
- `environment.json`：PyTorch、依赖版本、设备、精度、随机种子和代码来源。
- `loss.csv`：真实训练区间的 step、区间长度、Loss、FM/MF loss、原始速度 MSE、学习率和梯度范数。
- `loss_curve.png`：真实区间平均 Loss 与原始 MF 速度 MSE 曲线。
- `images/step_0000500_1step.png`：固定噪声、固定类别的一步采样网格。
- `ckpts/step_0000500.pt`：模型、优化器、scheduler、step、Loss 历史及随机数状态。

## 4. 一步生成与耗时记录

```bat
python sample_repro.py --checkpoint runs/mnist_first/ckpts/step_0010000.pt --sample-steps 1 5 --out samples/mnist_compare
```

默认每类生成 10 张，共 100 张；一行依次为数字 0–9。输出 `samples_1step.png`、`samples_5step.png` 和 `timing.json`。两种采样使用相同类别与初始噪声，默认 CFG scale=1。


### FID：完整训练后直接评估

评估协议：

- 后端为 [torch-fidelity 0.3.0](https://torch-fidelity.readthedocs.io/en/v0.3.0/api.html)，`inception-v3-compat` 的 **2048 维**特征，特征提取采用 FP32。
- 默认真实参考集为 MNIST/CIFAR-10 **test** 划分；`--split train` 可改变划分，结果必须注明。真实样本不做随机翻转。
- MNIST 真实图片先按模型输入大小做 PIL 双线性缩放；灰度图复制为三个 RGB 通道。生成图先限制在 [0,1]，再按 `round(255*x)` 转成 uint8；两组均交由库内部缩放到 299 后提取特征。
- 固定随机种子，生成类别数量与选中参考集的类别分布相同；一步/五步使用相同初始噪声。默认 CFG scale=1、生成 FP32，可通过参数明确改变并记录。
- 结果写入 `fid_results/.../fid.json`：包含权重 SHA256、训练 step、参考样本索引、样本数、采样步数、精度、依赖版本与各次 FID；同时保存预览图。
- 显存不足可调小 `--batch-size 16 --fid-batch-size 16`。`--cpu` 可以运行，但完整评估会较慢。


[← 返回复现总览](../README.md)
