# MeanFlow / iMF · MNIST 入门复现

基于 [haidog-yaqub/MeanFlow](https://github.com/haidog-yaqub/MeanFlow) 的第三方 PyTorch 实现，复现「训练平均速度模型 → 保存权重 → 一步生成」的完整流程。默认采用该仓库当前的 `i-meanflow` 模式，以 MNIST 为入门数据集。

**状态：代码流程已验证；完整 MNIST 训练、RTX 5060 Laptop GPU 上的 BF16 验证、生成质量及 FID 尚待实测。** 验证用的少量训练 step 或随机合成图片不作为论文复现结果。

## 1. 来源与复现范围

- 原论文：[Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447)。
- 后续论文：[Improved Mean Flows](https://arxiv.org/abs/2512.02012)。
- 参考代码：上述第三方仓库，固定 commit `97cc5182abe9d69e5dcb350268f603fbe31a5897`。
- 保留参考实现的 `meanflow.py`、`models/dit.py`、`data.py`、`train.py` 和原始 MNIST/CIFAR-10 配置，保留其 [MIT 许可](LICENSE)。参考源码不是本项目独立原创。
- 新增 `train_repro.py`、`sample_repro.py`、`repro_utils.py`、笔记本配置和流程验证脚本。

本实验是**第三方实现的流程复现**，不声明达到论文的 ImageNet FID 或原论文严格复现。当前参考代码使用双头 DiT、额外 FM loss，以及其自身的 JVP 计算方案。特别地，当前 `meanflow` 与 `i-meanflow` 两个分支在 `dudt` 被 detach 的处理下，其损失表达式代数等价；仅切换模式字符串不能作为原 MeanFlow 与 iMF 的有效对照实验。

## 2. 安装环境（Windows + NVIDIA）

在 Anaconda Prompt 中创建独立环境，然后进入本目录：

```bat
conda create -n meanflow python=3.11 -y
conda activate meanflow
cd /d E:\all\grad-prep-zhoujingjing\research\reproduction\03-meanflow-mnist
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

把 `E:\all\...` 改成实际下载位置。CUDA 12.8 安装命令来自 [PyTorch 官方版本安装页](https://pytorch.org/get-started/previous-versions/)。这是指定版本的安装方案，GPU 上仍需实际验证；依赖安装后会由每次实验的 `environment.json` 记录具体版本。

`timm` 是 DiT 代码的直接依赖，已经补入 `requirements.txt`。新增的单卡脚本不用 Accelerate；保留的上游 `train.py` 需要它。`diffusers` 仅供上游 latent 训练使用，本目录的入门实验不需要 VAE。

检查 GPU 和一次实际 CUDA 运算：

```bat
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print((torch.ones(4, device='cuda') * 2).cpu())"
```

接着执行两步合成数据检查，验证本机 BF16、JVP、反向传播和保存是否兼容：

```bat
python train_repro.py --smoke --out runs/smoke_gpu --log-every 1 --save-every 1 --sample-every 2
```

`--smoke` 使用缩小的模型与合成图片，不下载 MNIST，不能评价生成质量。若 BF16 出现非有限值，可加 `--precision no` 使用 FP32。无 GPU 的流程检查可以加 `--cpu`。

## 3. 先跑 MNIST

先进行 100 step 的真实 MNIST 检查，确认下载、速度和显存占用：

```bat
python train_repro.py --steps 100 --out runs/mnist_check --log-every 10 --save-every 100 --sample-every 100
```

然后从头训练入门配置的 10,000 step：

```bat
python train_repro.py --out runs/mnist_first
```

| 设置 | 入门配置 | 上游 MNIST 配置 |
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

入门配置改变了模型与训练预算，**不保证 10,000 step 达到上游展示的图像质量**。显存不足时使用 `--batch-size 16`；这也会改变实验设置，应在结果中记录。训练使用 step，不是 epoch：每个 step 是一个 batch 的参数更新。

想使用原规模配置，可运行：

```bat
python train_repro.py --config configs/mnist.py --out runs/mnist_upstream
```

## 4. 真实 Loss、保存和恢复

新增脚本默认每 100 step 记录一次真实区间平均 Loss，每 500 step 保存 checkpoint 和一步样本。输出内容：

- `config.json`：实际采用的配置，包括命令行覆盖。
- `environment.json`：PyTorch、依赖版本、设备、精度、随机种子和代码来源。
- `loss.csv`：真实训练区间的 step、区间长度、Loss、FM/MF loss、原始速度 MSE、学习率和梯度范数。
- `loss_curve.png`：真实区间平均 Loss 与原始 MF 速度 MSE 曲线。
- `images/step_0000500_1step.png`：固定噪声、固定类别的一步采样网格。
- `ckpts/step_0000500.pt`：模型、优化器、scheduler、step、Loss 历史及随机数状态。

自适应加权 Loss 不一定像普通 MSE 那样持续明显下降，应结合原始 `mf_v_mse` 和生成图片判断学习情况。

在终端按 `Ctrl+C` 可以停止。恢复时使用 `ckpts/` 中最新的**完整 checkpoint**：

```bat
python train_repro.py --resume runs/mnist_first/ckpts/step_0000500.pt --steps 10000 --out runs/mnist_resumed
```

`--steps` 表示恢复后要达到的**累计总 step**，不是额外训练步数。恢复会读取 checkpoint 中的配置，保留已有真实 Loss 历史，并写入新目录；原文件不覆盖。若已有 10,000 step，想继续到 20,000，则传 `--steps 20000`。

暂停点晚于最新完整 checkpoint 的更新不会被恢复；默认最多丢失约 500 step，可调小 `--save-every`。恢复后的数据批次顺序可能变化，不承诺与不中断训练逐位一致。延长总训练步数时保持原学习率计划，计划结束后使用最低学习率。

指定的输出目录必须不存在，避免误覆盖；重复运行时换一个名字，或省略 `--out` 自动创建带时间戳的新目录。

## 5. 一步生成与耗时记录

```bat
python sample_repro.py --checkpoint runs/mnist_first/ckpts/step_0010000.pt --sample-steps 1 5 --out samples/mnist_compare
```

默认每类生成 10 张，共 100 张；一行依次为数字 0–9。输出 `samples_1step.png`、`samples_5step.png` 和 `timing.json`。两种采样使用相同类别与初始噪声，默认 CFG scale=1。

计时先预热 3 次，再重复 10 次。GPU 每次计时前后同步；不计模型加载、初始噪声生成与 PNG 保存时间。记录整批平均时间、标准差和按 batch 大小折算的每图耗时。**1-step 的 NFE=1，5-step 的 NFE=5**；每图耗时是批处理平均值，不等于单图请求延迟。

上游 `sample_each_class()` 的默认 `sample_steps=5`。本目录新增脚本明确指定一步生成，避免把默认五步结果标作一步。

此处是同一模型的一步/五步对照。若与已有 DDPM/DDIM 比较，需相同数据集、设备、图片尺寸、batch 大小、精度和计时范围；不能把 MNIST 的耗时或 FID 与 CIFAR-10 直接比较。

## 6. CIFAR-10 与结果记录

保留 `configs/cifar10.py`，可运行：

```bat
python train_repro.py --config configs/cifar10.py --out runs/cifar10_upstream
```

它采用上游的大模型与 200,000 step 配置，尚未在笔记本上验证；先完成 MNIST 流程更容易排查问题。本目录不填入未测得的正式 FID、训练时间或生成结果。

### FID：完整训练后直接评估

`eval_fid.py` 已补充并验证评估流程，支持本目录 `train_repro.py` 保存的 MNIST / CIFAR-10 checkpoint。先完整训练，再运行：

```bat
python -m pip install torch-fidelity==0.3.0
python eval_fid.py --checkpoint runs/mnist_first/ckpts/step_0010000.pt --sample-steps 1 5 --num-samples 10000 --out fid_results/mnist_first
```

脚本会生成 **10,000 张一步图片和 10,000 张五步图片**，分别与固定的 **10,000 张测试集真实图片**计算 FID。第一次运行会下载约 91 MB 的预训练 Inception 权重；若网络无法下载，可先下载官方兼容权重，再用 `--fid-weights` 指定本地路径。

评估协议：

- 后端为 [torch-fidelity 0.3.0](https://torch-fidelity.readthedocs.io/en/v0.3.0/api.html)，`inception-v3-compat` 的 **2048 维**特征，特征提取采用 FP32。
- 默认真实参考集为 MNIST/CIFAR-10 **test** 划分；`--split train` 可改变划分，结果必须注明。真实样本不做随机翻转。
- MNIST 真实图片先按模型输入大小做 PIL 双线性缩放；灰度图复制为三个 RGB 通道。生成图先限制在 [0,1]，再按 `round(255*x)` 转成 uint8；两组均交由库内部缩放到 299 后提取特征。
- 固定随机种子，生成类别数量与选中参考集的类别分布相同；一步/五步使用相同初始噪声。默认 CFG scale=1、生成 FP32，可通过参数明确改变并记录。
- 结果写入 `fid_results/.../fid.json`：包含权重 SHA256、训练 step、参考样本索引、样本数、采样步数、精度、依赖版本与各次 FID；同时保存预览图。
- 显存不足可调小 `--batch-size 16 --fid-batch-size 16`。`--cpu` 可以运行，但完整评估会较慢。

**MNIST 上的 Inception FID 作为辅助指标使用。** 它不等于 MNIST 分类网络特征上的 Fréchet 距离，也不能直接与 ImageNet/CIFAR-10 或不同样本数、预处理、划分的论文分数比较。较低的 FID 不单独证明类别正确或数字清晰，应同时检查生成图。

若只是检查评估能否运行，可减少样本并明确标为 debug：

```bat
python eval_fid.py --checkpoint runs/mnist_first/ckpts/step_0000500.pt --num-samples 256 --sample-steps 1 --debug --out fid_results/debug
```

少量样本、合成数据测试权重或非常早期的权重会被标作 `pipeline_check_or_preliminary`，不能作为完整复现成绩。当前只有短测权重，**正式 FID 仍待完整训练权重**。

记录真实实验时请在 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md) 填写配置、训练总 step、样本与指标，并上传对应真实曲线和生成图片；权重和数据默认由 `.gitignore` 排除。

## 7. 流程验证

```bat
python tests/check_repro.py
```

检查真实 MeanFlow loss 中的 JVP 与反向传播、checkpoint 状态、恢复后的历史保留、模型继续更新，以及采样调用次数为 1/5 NFE。它使用 CPU 小模型和合成数据，不衡量生成质量。验证状态和环境详见 [VALIDATION.md](VALIDATION.md)。

[← 返回复现总览](../README.md)
