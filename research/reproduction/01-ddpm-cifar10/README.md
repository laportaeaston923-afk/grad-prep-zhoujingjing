# DDPM · CIFAR-10

复现噪声预测训练与反向采样，记录 120 Epoch AMP 实验的 Loss、生成图片和去噪过程。

## 目录

- `code/`：模型、训练和采样代码。
- `results/`：Loss 曲线、生成图片、去噪过程与详细实验记录。

## 环境与运行

需要 Python、PyTorch、torchvision、matplotlib、tqdm；可使用 CUDA GPU。

```bat
cd code
python -m pip install -r requirements.txt
python sample.py --checkpoint ../results/ddpm.pth
python ddpm2.py --checkpoint ../results/ddpm.pth
```

将匹配网络结构的模型权重放到 `results/ddpm.pth`，也可用 `--checkpoint` 指定其他文件。新采样图保存为 `results/samples_latest.png`，过程图保存为 `results/denoising_latest.png`；加 `--show` 可同时显示图片。

`train_full_120_amp.py` 将训练目标设为 120 Epoch，读取 `results/train_results_120/checkpoint.pth` 或已有 AMP checkpoint，输出到 `results/train_results_120_amp/`。从 `code/` 运行：

```bat
python train_full_120_amp.py
```

## 实验结果

[查看 results](results/)：完整 120 Epoch Loss 展示图、生成样本和去噪过程图。此处展示的是已有实验，本次目录整理没有重新训练。

代码学习参考：Bilibili DDPM 教学视频（BV1vZi6BaEgF）；基础实现非独立原创，来源说明保留在详细实验记录中。
