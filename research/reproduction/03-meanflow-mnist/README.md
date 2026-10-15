# 03 · Improved MeanFlow (iMF) — MNIST 复现

> **当前状态：已编写并验证核心代码；完整 MNIST 训练和真实图表等待在 GPU 上运行。** 未训练完成前，不上传伪造的结果曲线。

**论文**：[Improved Mean Flows: On the Challenges of Fastforward Generative Models](https://arxiv.org/abs/2512.02012)  
**代码参考**：[haidog-yaqub/MeanFlow](https://github.com/haidog-yaqub/MeanFlow)，第三方 PyTorch 实现（MIT）；[作者官方 JAX 代码](https://github.com/Lyy-iiis/imeanflow)。本项目按论文核心公式独立改写为轻量 MNIST 实验，并非原 ImageNet DiT 的完整复现。

## 要做的四组图

| 实验 | 结果图 | 核心观察 |
|---|---|---|
| 论文 Fig. 3 风格 Loss | [fig3_loss_mnist.png](results/fig3_loss_mnist.png) | MF/iMF 无 CFG 的训练 Loss；只统计 t ≠ r，普通 MSE |
| CFG 对比 | [cfg_scales.png](results/cfg_scales.png) | 同一 iMF 模型使用 ω=1、1.5、2、3、4 一步生成 |
| 一步生成可视化 | [one_step_process.png](results/one_step_process.png) | 随机噪声 → 平均速度 u → 一次前向得到手写数字 |
| iMF vs MF | [mf_vs_imf.png](results/mf_vs_imf.png) | 相同网络、类别、训练步数和噪声的一步生成对比 |

完成定量评估后，另有 [mf_vs_imf_metrics.png](results/mf_vs_imf_metrics.png) 和 metrics.csv。表中的图片文件需要在完成训练后才会生成。

## 核心原理与改动

- MF 论文 Eq.(9)：JVP 使用条件速度 e−x 作为方向；
- iMF 论文 Eq.(12)：JVP 使用预测的边界瞬时速度 vθ(z,t)=uθ(z,t,t) 作为方向；
- 两者都预测平均速度 uθ(z,r,t)，损失使用重新参数化的 v-loss；
- iMF CFG 将引导尺度 ω 作为可改变的输入条件，训练时随机采样引导尺度，对应论文 Algorithm 2；
- 生成只需要 1-NFE：z0 = z1 − uθ(z1,0,1,c,ω)，图中的中间速度可视化不是多步去噪。

**特别注意**：参考仓库的 mode 选项单独切换，不足以严格复现论文 Fig.3 中 JVP 方向的区别。本实验明确改变了 JVP tangent，不将同一数学目标的两种写法误作对照。

## Windows / VS Code 运行

打开 VS Code 的终端，进入 03-meanflow-mnist 文件夹。推荐安装带 NVIDIA CUDA 的 PyTorch，然后运行：

```powershell
pip install -r requirements.txt
python code/run_all.py --steps 20000 --batch-size 128 --device auto
```

命令会自动下载 MNIST、依次训练 mf、imf、imf_cfg 三个模型，保存真实训练日志及 checkpoint，最后生成上面四张图。要先用最短流程检查代码，可将 --steps 改为 2；**两步训练的随机输出不能作为论文结果**。建议 GPU 进行正式训练，20,000 步为 MNIST 练习起点，并非论文超参数或生成质量保证。

如要分开训练或断点续训：

```powershell
python code/train.py --mode mf --steps 20000
python code/train.py --mode imf --steps 20000
python code/train.py --mode imf_cfg --steps 20000
python code/train.py --mode imf --steps 25000 --resume
python code/plot_results.py
```

**可选定量对照**：

```powershell
python code/evaluate.py --train-classifier --classifier-epochs 4 --num-samples 2000
```

此脚本会另训练 MNIST 分类器，计算生成图在分类器特征空间中的 Fréchet 距离及目标数字类别一致率，并保存评价器在真实 MNIST test 上的正确率。该指标不是标准的 Inception FID，**绝对不能拿它和论文 ImageNet FID=1.72 比较**。需看实际记录决定 iMF 是否优于 MF，不预先写结论。

## 目录

```text
03-meanflow-mnist/
├── README.md
├── requirements.txt
├── code/
│   ├── model.py          # U-Net、JVP 两种方向、CFG、1-step
│   ├── train.py          # 分开训练+真实 CSV+模型保存
│   ├── plot_results.py   # 生成四组图片
│   ├── evaluate.py       # 可选 MNIST 特征距离
│   └── run_all.py        # 一键执行
├── results/              # 正式训练后放结果
├── runs/                 # 本地产生 checkpoint 和日志；不提交大权重
└── data/                 # MNIST 自动下载；不提交
```

**论文设定差异**：MNIST 28×28、小卷积 U-Net、无 VAE；论文实验则为 ImageNet 256×256、DiT。这里只验证核心目标、CFG 灵活性和一步生成，不声称复现原论文的 FID、模型规模、in-context token 结构及 CFG interval。Fig.3 是否出现相同趋势须根据实际日志客观报告。