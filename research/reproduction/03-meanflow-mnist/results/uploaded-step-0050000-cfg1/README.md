# 上传权重的 MNIST FID 实测

评估日期：2026-10-11。使用用户上传的 `step_0050000.pt`，不重新训练。FID 越低，表示该特征空间中两组图像的分布越接近。

| 采样 | NFE | FID |
|---|---:|---:|
| 一步 | 1 | 179.8574 |
| 五步 | 5 | 17.9291 |

## 评估协议

- MNIST test，10,000 张真实图与每种采样 10,000 张生成图；参考样本种子 42，噪声种子 43。
- CFG scale=1；一步/五步使用相同类别与初始噪声，类别数量按测试集分布设置。
- 真实灰度图通过 PIL 双线性缩放到 32×32 后复制为 RGB；生成图 clip 到 [0,1]、round(255*x) 转为 uint8。
- `torch-fidelity==0.3.0`、`inception-v3-compat`、2048 维特征，库内部缩放到 299×299。全部使用 FP32。
- 设备为 CPU，8 线程；生成 batch=32，特征 batch=64。采用等价的 SDPA attention 和 Inception channels-last 布局；具体依赖版本见 [fid.json](fid.json)。
- 参考统计按图片内容、特征权重、设备与布局缓存。模型加载与保存原始权重不作修改。

## 权重与结构

上传权重的 SHA256：`88a7a517315ff5445136671f54c8c44520eea90f19bdd82bf4334789bf4452e1`。

模型通过严格 state_dict 加载：32×32 灰度图、patch=2、dim=256、depth=6、heads=4、10 类，7,831,304 个参数。后续上传的训练配置确认 `max_cfg=7.5` 和 minmax 归一化。评估适配配置见 [mnist_uploaded_50k.py](../../configs/mnist_uploaded_50k.py)。

评估时原文件仅含模型参数，未包含训练模式、累计 step 或优化器状态。随后用户补充的 `train.log` 与 `config.py` 标注 `meanflow`、记录到 50,000 step，模型结构与所评估权重一致；详见 [训练来源记录](../mnist-meanflow-50k-training/)。原 `fid.json` 保留评估当时的协议，后续训练证据独立保存，不改动已测 FID 数值。

该分数为本协议下 MNIST 的 Inception FID，不能直接与使用 MNIST 分类器特征、60,000 张训练参考图、不同预处理，或 ImageNet/CIFAR-10 的 FID 比较。

## 复测

把相同权重放进项目目录，从 `03-meanflow-mnist` 运行：

```bat
python eval_fid.py --checkpoint step_0050000.pt --config configs/mnist_uploaded_50k.py --num-samples 10000 --sample-steps 1 5 --precision no --cpu --threads 8 --attention sdpa --feature-memory-format channels_last --batch-size 32 --fid-batch-size 64 --out fid_results/recheck
```

在 GPU 上运行可去掉 `--cpu`；应保留其余评估设置并记录设备变化。

## 样本预览

每图展示该评估中最先生成的 100 张，类别顺序按真实参考集的固定随机顺序排列，不是每行 0–9。

一步：

![一步采样预览](preview_1step.png)

五步：

![五步采样预览](preview_5step.png)

## 生成过程可视化

同一权重、CFG=1、相同初始噪声，固定数字类别 0–9。[完整过程说明与静态图](../uploaded-step-0050000-process/)。

一步（2 种实际状态，1 次模型调用）：

![一步生成过程](../uploaded-step-0050000-process/process_1step.gif)

五步（6 种实际状态，5 次模型调用）：

![五步生成过程](../uploaded-step-0050000-process/process_5step.gif)

动图只切换真实采样状态，播放时间不代表计算耗时。这 10 个过程示例独立于 FID 的 10,000 张评估样本。

## 训练 Loss

已从补充上传的真实日志生成曲线：500–50,000 step，共 100 个记录点，总 Loss 1.985971 → 1.856935。\n\n![真实训练 Loss](../mnist-meanflow-50k-training/loss_curve.png)\n\n[训练配置、原日志、CSV 与解读](../mnist-meanflow-50k-training/)。

[← 返回实验说明](../../README.md)
