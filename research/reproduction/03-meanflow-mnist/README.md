# MeanFlow · MNIST

基于第三方 PyTorch 实现，学习平均速度并研究一步生成。现有日志标注为 MeanFlow、50,000 step；五步用于补充对照。

## 目录

- `code/`：训练、采样、评估代码及 `configs/`、`models/`。
- `results/`：原日志、配置、Loss、FID、生成图片与过程动图。

## 环境与运行

需要已安装 PyTorch 和 torchvision；其余依赖见 `code/requirements.txt`。GPU 运行需要可用的 CUDA，CPU 运行加 `--cpu`。

```bat
cd code
python -m pip install -r requirements.txt
```

训练示例（新运行，默认输出到 `results/runs/`）：

```bat
python train_repro.py --config configs/mnist_laptop.py --steps 100 --precision no --cpu
```

复测已有权重：将相同的 `step_0050000.pt` 放进 `results/weights/`，从 `code/` 运行：

```bat
python eval_fid.py --checkpoint ../results/weights/step_0050000.pt --config configs/mnist_uploaded_50k.py --cfg-scale 1 --sample-steps 1 5 --num-samples 10000 --cpu --threads 8 --attention sdpa --feature-memory-format channels_last --batch-size 32 --fid-batch-size 64
```

将 `--cfg-scale 1` 改为 `4.25` 可复测有引导设置。评估默认输出到 `results/fid_runs/`，过程可视化与完整命令见实验记录。

## 实测结果

| 采样 | CFG=1，FID ↓ | CFG=4.25，FID ↓ |
|---|---:|---:|
| 一步 | 179.8574 | 150.9314 |
| 五步 | 17.9291 | 35.2649 |

同一权重、相同噪声和类别，每个设置 10,000 张生成图 vs 10,000 张 MNIST test；CPU / FP32。两组都有类别条件，CFG=4.25 改善一步 FID，却使五步 FID 变差。

[查看 results](results/)：Loss 曲线、一步/五步生成过程与 CFG 对照。

代码来源：[haidog-yaqub/MeanFlow](https://github.com/haidog-yaqub/MeanFlow)，固定 commit `97cc5182abe9d69e5dcb350268f603fbe31a5897`；MIT 许可保留在 `code/LICENSE`。本次是第三方实现的 MNIST 实验，不声明达到原论文 ImageNet 成绩。
