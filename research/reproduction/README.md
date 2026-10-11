# 代码复现 · DDPM、DDIM 与 MeanFlow

> 三个独立实验目录。DDPM 负责训练噪声预测网络；DDIM 复用其权重研究另一种采样方法；MeanFlow / iMF 单独训练平均速度模型，验证一步生成流程。

## 01 · 项目总览

| 项目 | 主要目标 | 实验内容 | 详细记录 |
|---|---|---|---|
| **DDPM** | 理解前向加噪、噪声预测与反向去噪 | CIFAR-10 使用 `train_full_120_amp.py` 从头进行 120 Epoch AMP 训练；以 `train_results_120_amp/` 的完整 Loss 为最终结果 | [01 · DDPM](01-ddpm-cifar10/) |
| **DDIM** | 理解如何在不重新训练 U-Net 的情况下改变采样方式 | 复用 DDPM 权重，将采样调用改为 `ddim_sample(steps=50)` | [02 · DDIM](02-ddim-cifar10/) |
| **MeanFlow / iMF** | 理解平均速度模型与一步生成 | MNIST 50k 真实 Loss、生成过程；FID 对照：一步 179.8574→150.9314、五步 17.9291→35.2649（CFG=1→4.25）；一步改善、五步变差 | [03 · MeanFlow / iMF](03-meanflow-mnist/) |

**当前状态：** DDPM 的最终训练记录采用 `train_full_120_amp.py` 对应的 **120 Epoch AMP** 实验，以 `train_results_120_amp/` 中的真实 Loss 为准，不合并旧的继续训练日志；完整 Loss 曲线、Python 源码及生成图片尚未上传。DDIM 内部采样公式与实际耗时仍需核验，使用的具体 checkpoint 待上传时核对。

## 02 · DDPM 与 DDIM 的关系

```text
                   CIFAR-10
                       │
                DDPM 训练 U-Net
                       │
                保存 ddpm.pth
                       │
           ┌───────────┴───────────┐
           │                       │
      DDPM 原始采样           DDIM 50 步采样
      ddpm.sample()          ddpm.ddim_sample()
           │                       │
        生成图像                生成图像
```

两个实验可以复用**相同的噪声预测网络结构及 checkpoint**，仅改变采样方式。DDIM 在本次实践中不单独训练模型；当前 DDIM 脚本实际加载的是哪份 checkpoint（旧权重或 120 Epoch AMP 新权重）尚需核对，不能直接认定已使用最终 AMP 权重。其采样函数也需结合 `ddpm.py` 检查。

## 03 · 实验设置对比

| 项目 | DDPM | DDIM |
|---|---|---|
| 数据集 | CIFAR-10 | CIFAR-10 |
| 图片大小 | 3 × 32 × 32 | 3 × 32 × 32 |
| 网络 | U-Net | 复用 DDPM U-Net |
| 扩散时间步 | 1000 | 1000（共用噪声调度） |
| 模型权重 | 120 Epoch AMP 实验实际保存的 checkpoint（待核对文件名） | 可复用同一 checkpoint（当前 DDIM 脚本所加载权重待核对） |
| 采样函数 | `sample()` | `ddim_sample()` |
| 采样步数 | 由原始 DDPM 采样函数决定 | 脚本指定 50 |
| 训练方式 | `train_full_120_amp.py`，从头训练 120 Epoch（AMP） | 无需单独训练，可复用 DDPM 权重 |

## 04 · 代码文件组织

建议分别上传到以下目录。**目前下面的 `.py` 文件为计划结构，并非已经上传：**

```text
research/reproduction/
├── README.md                 # 两个实验的统一总结
├── 01-ddpm-cifar10/
│   ├── README.md             # DDPM 训练、采样及结果
│   ├── ddpm.py
│   ├── unet.py
│   ├── train_full_120_amp.py # 最终采用的 120 Epoch AMP 脚本
│   ├── sample.py
│   └── results/              # 选自 train_results_120_amp/ 的真实 Loss、生成样本
└── 02-ddim-cifar10/
    ├── README.md             # DDIM 实现与结果
    ├── ddpm.py               # 含 ddim_sample 的版本
    ├── unet.py               # 相同网络结构
    └── sample_ddim.py
```

DDIM 的代码是在 DDPM 实现基础上修改的，若希望两个文件夹能够**独立运行**，需要各自具有正确的依赖模块；若后续改成共享模块结构，应同步调整 Python 导入路径。

## 05 · 参考资料

- [DDPM 原始论文（NeurIPS 2020）](https://arxiv.org/abs/2006.11239)
- [DDIM 原始论文（ICLR 2021）](https://arxiv.org/abs/2010.02502)
- [DDPM 学习参考：Bilibili 教学视频](https://www.bilibili.com/video/BV1vZi6BaEgF/)



[← 返回科研任务](../README.md)
