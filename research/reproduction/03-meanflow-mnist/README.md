# 03 · Improved MeanFlow (iMF) — MNIST + 原项目 DiT

**状态：DiT 代码已上传；实际 MNIST 训练和真实结果图待在本机 GPU 上运行。**

本项目直接采用 [haidog-yaqub/MeanFlow](https://github.com/haidog-yaqub/MeanFlow) 的 **MFDiT（DiT Transformer）**，不是之前自行实现的 U-Net。之前 U-Net 版代码已归档到 [legacy_unet/](legacy_unet/)，不会读取或者覆盖之前的旧模型权重。

参考：[iMF 论文](https://arxiv.org/abs/2512.02012) · [原仓库](https://github.com/haidog-yaqub/MeanFlow)（第三方 PyTorch，MIT）· [论文作者 JAX 代码](https://github.com/Lyy-iiis/imeanflow)。

## 本次只做四组图

| 实验 | 训练完成后的图片 | 目标 |
|---|---|---|
| Fig.3 风格 Loss | results/dit/fig3_loss_dit.png | MF/iMF 不同 JVP 方向，无 CFG，plain MSE，仅 t≠r |
| CFG 尺度比较 | results/dit/cfg_scales_dit.png | 同一 iMF-CFG 模型测试 ω=1/1.5/2/3/4 |
| 一步生成过程 | results/dit/one_step_process_dit.png | 噪声 z1 → 平均速度 u → 图像 z0=z1-u（1-NFE） |
| MF 与 iMF | results/dit/mf_vs_imf_dit.png | 相同 DiT 规模、训练预算、随机噪声和类别条件 |

可选运行 code/evaluate.py 计算 MNIST 分类器特征 Fréchet 距离、类别一致率（results/dit/mf_vs_imf_metrics_dit.png 和 metrics.csv）。**此数值不是标准 Inception FID，不能与论文 ImageNet 的 FID 对比。**

## 源代码沿用哪些部分

直接从原项目 [commit 97cc518](https://github.com/haidog-yaqub/MeanFlow/commit/97cc5182abe9d69e5dcb350268f603fbe31a5897) 拷贝、未修改：

- [code/models/dit.py](code/models/dit.py)：原始双头 **MFDiT** 神经网络；
- [code/data.py](code/data.py)：原始 MNIST 预处理（28×28 变为 32×32）；
- [code/upstream_meanflow.py](code/upstream_meanflow.py)：原 meanflow.py，作为参考保留；
- [code/upstream_mnist_config.py](code/upstream_mnist_config.py)：原 MNIST 配置；
- [UPSTREAM_LICENSE](UPSTREAM_LICENSE)：原项目 MIT 许可。

为正确比较论文 Fig.3，新写的 [code/experiment_core.py](code/experiment_core.py) 实现：

- **MF / Eq.(9)：** JVP 方向采用真实条件速度 e−x；
- **iMF / Eq.(12)：** JVP 方向采用网络预测的瞬时速度 vθ，使用边界条件 vθ(z,t)=uθ(z,t,t)，不训练额外的 v 辅助头；
- 两者都预测平均速度 uθ；回归重新参数化的瞬时速度；无 CFG、无 adaptive loss，只记录 t≠r 的 v-loss；
- **iMF-CFG / Algorithm 2：** 显式条件 ω，使用原 MFDiT 的辅助 v-head 与引导目标；此模型和 Fig.3 两种无 CFG 消融分别训练。

原第三方代码中切换 meanflow.mode 主要改变目标的 u-loss/v-loss 表述，**不是** Fig.3 论文中的不同 JVP 方向。训练代码对此做了有依据的改动，**DiT 网络本身不改**。

## VS Code Windows：从头使用

推荐独立 Conda 环境，避免之前 base 环境 OpenMP 库冲突。

~~~powershell
conda create -n imfdit python=3.11 -y
conda activate imfdit
~~~

按设备在 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装匹配的 CUDA / PyTorch。然后进入本项目目录执行：

~~~powershell
pip install -r requirements.txt
python code\smoke_test.py
~~~

看到 SMOKE TEST PASSED 后再开始训练。**不要**依赖 KMP_DUPLICATE_LIB_OK=TRUE 的危险兼容模式来执行正式实验。

**训练三组 DiT 并自动生成四张结果图：**

~~~powershell
python code\run_all.py --steps 20000 --batch-size 64
~~~

新模型默认 dim=256, depth=6, heads=8, patch=4。仍是**原项目的 MFDiT**，但缩小训练成本；上游 MNIST 配置是 dim=512、depth=12、patch=2、200000 步，因此本项目为方法级复现，不能冒称原论文精确结果。GPU 显存有限可以尝试减小 batch size。

**中断后续训（相同参数）：**

~~~powershell
python code\run_all.py --steps 25000 --batch-size 64 --resume
~~~

会恢复模型与优化器，但数据 shuffle 顺序会重新开始，不保证与不中断训练逐位一致。无 --resume 时若检测到已有 checkpoint，会拒绝覆盖。

**若三个模型已训练完毕，只重新生成四张图：**

~~~powershell
python code\plot_results.py
~~~

**可选定量对比：**

~~~powershell
python code\evaluate.py --train-classifier --samples 2000
~~~

旧 U-Net checkpoint 不能用于新 DiT。新权重位置为 runs_dit/mf/last.pt、runs_dit/imf/last.pt、runs_dit/imf_cfg/last.pt；旧 runs/ 目录保持不变。四张图保存在 results/dit/，上传真实图前不制造示例论文曲线。

## 文件结构

~~~text
03-meanflow-mnist/
├── README.md
├── UPSTREAM_LICENSE
├── requirements.txt
├── code/
│   ├── models/dit.py           # 原项目 DiT 网络，未改
│   ├── data.py                 # 原项目 MNIST
│   ├── upstream_meanflow.py    # 上游训练目标留档
│   ├── upstream_mnist_config.py
│   ├── experiment_core.py      # Eq9/Eq12/Algorithm2
│   ├── train.py                # 真实 CSV + checkpoint
│   ├── plot_results.py         # 四种真实图
│   ├── evaluate.py             # 可选定量指标
│   ├── run_all.py
│   └── smoke_test.py
├── results/dit/                # 待真实运行输出
├── legacy_unet/                # 旧版代码备份
├── runs_dit/                   # 本地权重，不提交
└── data/                       # 本地 MNIST，不提交
~~~

**研究结论原则：** 不能预设 iMF 在 MNIST 上一定优于 MF；应该根据真实 Loss、生成图和指标客观分析差异。
