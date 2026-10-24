# DDPM · CIFAR-10

120 Epoch AMP 训练，记录 Loss、生成样本和反向去噪过程。

[论文笔记](../../paper-notes/12-DDPM.md) · [代码](code/) · [结果](results/) · [实验配置](results/EXPERIMENT_GUIDE.md)

## 运行

需要 Python、PyTorch、torchvision、matplotlib、tqdm。进入本项目的 `code/`：

```bat
python -m pip install -r requirements.txt
python sample.py --checkpoint ../results/ddpm.pth
python ddpm2.py --checkpoint ../results/ddpm.pth
```

权重须与网络结构匹配；可通过 `--checkpoint` 指定路径。结果写入 `results/samples_latest.png` 和 `results/denoising_latest.png`，`--show` 可显示图片。

训练命令：`python train_full_120_amp.py`。脚本读取 `results/train_results_120/checkpoint.pth` 或已有 AMP checkpoint，目标为 120 Epoch，输出到 `results/train_results_120_amp/`。

基础实现参考 Bilibili 教学视频 BV1vZi6BaEgF；来源见实验配置。

[返回复现目录](../README.md)
