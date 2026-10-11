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

## FID 评估流程（新增）

- 使用 `torch-fidelity==0.3.0` 的实际预训练 `inception-v3-compat` / 2048 维特征完成 CPU 短测。
- 真实 MNIST test 图片与两步训练模型生成图片可完成预处理、特征提取及 FID 计算，并写入带权重哈希和协议的 JSON。
- 少量样本短测只验证评估流程；不上传其分数作为正式复现结果。
- `python tests/check_fid.py` 检查灰度转 RGB、uint8 一致性、参考样本选择可复现，以及同一组真实图片的 FID 接近 0。

## 未完成的正式实验

- RTX 5060 Laptop GPU 的 CUDA / BF16 验证与显存、速度测试。
- 完整 10,000 step MNIST 训练及生成质量检查。
- CIFAR-10 训练、FID 评估、与 DDPM/DDIM 的公平对照。
- 原论文 ImageNet 结果的严格复现。

不将检查用图片、权重或短测耗时作为正式实验结果上传。正式结果请填写 `EXPERIMENT_RECORD.md`。
