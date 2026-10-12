# 流程验证记录

验证日期：2026-10-11。平台：Linux / CPU；GPU 与 Windows 兼容性需在目标设备实测。

实际检查环境：Python 3.12.14、PyTorch 2.14.1+cpu、torchvision 0.29.1+cpu、timm 1.0.30、einops 0.8.2。README 中的 PyTorch 2.7.1 / CUDA 12.8 安装方案尚未在该 CPU 检查环境或目标 GPU 上验证。

## 已完成

1. 所有 Python 文件通过 `python -m compileall -q`。
2. `python tests/check_repro.py` 通过：
   - 使用参考代码的实际 JVP 与 loss，完成反向传播和参数更新。
   - 训练两步后恢复到累计四步，保留原始真实 Loss 历史。
   - checkpoint 含非空优化器状态、scheduler、随机数状态及累计 step。
   - 自动保留合成数据测试标记，避免恢复测试时误当作正式 MNIST 实验。
   - forward hook 验证一步采样调用模型一次，五步采样调用五次。
   - 输出图像尺寸正确、数值有限且处于 [0, 1]。
   - PNG、Loss CSV、曲线及计时 JSON 可以写出。
3. 真实 MNIST 下载成功；使用 `mnist_laptop.py` 的 7.79M 参数模型，CPU / FP32 / batch=2 完成两步训练并生成一步采样图片。

两步训练仅验证数据与代码链路，不能说明模型收敛或图像质量。合成测试使用约 0.05M 参数的小模型。

## FID 评估与实测

- 上游纯 state_dict 权重可通过显式 `--config` 加载；本次上传权重通过严格结构检查，包含 7,831,304 个参数。
- FP32 小批次对照：原生 attention 与 SDPA 的生成图最大差异约 1.8e-6；连续布局与 channels-last 的 Inception 特征最大差异约 5.5e-6。
- 同一真实参考的缓存按图片内容、特征权重、设备和布局区分，避免引用不同预处理或权重的统计量。
- 修复 torch-fidelity 0.3 的 NumPy 统计缓存与新版 PyTorch 默认加载模式的兼容问题；缓存回读验证通过。

- 使用 `torch-fidelity==0.3.0` 的实际预训练 `inception-v3-compat` / 2048 维特征完成 CPU 短测。
- 真实 MNIST test 图片与两步训练模型生成图片可完成预处理、特征提取及 FID 计算，并写入带权重哈希和协议的 JSON。
- 少量样本短测只验证评估流程；不上传其分数作为正式复现结果。
- `python tests/check_fid.py` 通过：灰度转 RGB、uint8 一致性、参考样本选择可复现、NumPy 统计缓存回读；相同真实图片的 FID 为约 -0.00006617（浮点误差范围内接近 0）。
- 上传的 `step_0050000.pt` 完成 10,000 张生成图对 10,000 张 MNIST test 的正式规模评估：CFG=1，一步 FID=179.8574，五步 FID=17.9291，进程退出码 0。详见 [结果与协议](uploaded-step-0050000-cfg1/)。原始权重不含训练历史，评估结果不验证其训练过程。

## 未完成的正式实验

- RTX 5060 Laptop GPU 的 CUDA / BF16 验证与显存、速度测试。
- 完整 10,000 step MNIST 训练及生成质量检查。
- CIFAR-10 训练、FID 评估、与 DDPM/DDIM 的公平对照。
- 原论文 ImageNet 结果的严格复现。

不将流程检查用图片、权重或短测耗时作为正式实验结果上传。上传权重的实际评估已记录在 `EXPERIMENT_RECORD.md`。

## 目录整理验证（2026-10-12）

代码迁移至 `code/`，结果集中至 `results/`。全部 Python 文件语法检查通过；DDPM/DDIM 采样 CLI 可用；MeanFlow 的 JVP/反向传播、checkpoint 恢复、1/5 NFE 采样与输出检查通过。单步模拟检查确认默认训练输出位于 `results/runs/`。这些检查没有重新训练或替换已发布的正式实验结果。
