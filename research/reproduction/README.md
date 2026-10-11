# 代码复现 · DDPM 与 DDIM（CIFAR-10）

> **两个独立实验目录，一份统一总结。** DDPM 负责训练噪声预测网络；DDIM 在已有 DDPM 权重的基础上研究另一种采样方法。

## 01 · 项目总览

| 项目 | 主要目标 | 实验内容 | 详细记录 |
|---|---|---|---|
| **DDPM** | 理解前向加噪、噪声预测与反向去噪 | CIFAR-10 训练 100 Epoch；保存 `ddpm.pth`；尝试继续训练 20 Epoch | [01 · DDPM](01-ddpm-cifar10/) |
| **DDIM** | 理解如何在不重新训练 U-Net 的情况下改变采样方式 | 复用 DDPM 权重，将采样调用改为 `ddim_sample(steps=50)` | [02 · DDIM](02-ddim-cifar10/) |

**当前状态：** 已整理复现说明；Python 源代码、训练日志、生成图片尚未上传到这两个目录。表中的实验设置来自现有本地脚本，DDIM 内部采样公式与实际耗时仍需核验。

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

两个实验中的**噪声预测网络结构和训练权重相同**，采样方式不同。DDIM 在本次实践中不单独训练模型；其采样函数的实现仍需结合 `ddpm.py` 检查。

## 03 · 实验设置对比

| 项目 | DDPM | DDIM |
|---|---|---|
| 数据集 | CIFAR-10 | CIFAR-10 |
| 图片大小 | 3 × 32 × 32 | 3 × 32 × 32 |
| 网络 | U-Net | 复用 DDPM U-Net |
| 扩散时间步 | 1000 | 1000（共用噪声调度） |
| 模型权重 | `ddpm.pth` | 复用 `ddpm.pth` |
| 采样函数 | `sample()` | `ddim_sample()` |
| 采样步数 | 由原始 DDPM 采样函数决定 | 脚本指定 50 |
| 是否重新训练 | DDPM 初始训练 100 Epoch | 否 |

## 04 · 代码文件组织

建议分别上传到以下目录。**目前下面的 `.py` 文件为计划结构，并非已经上传：**

```text
research/reproduction/
├── README.md                 # 两个实验的统一总结
├── 01-ddpm-cifar10/
│   ├── README.md             # DDPM 训练、采样及结果
│   ├── ddpm.py
│   ├── unet.py
│   ├── train.py
│   ├── sample.py
│   └── train_continue.py     # 若已准备好
└── 02-ddim-cifar10/
    ├── README.md             # DDIM 实现与结果
    ├── ddpm.py               # 含 ddim_sample 的版本
    ├── unet.py               # 相同网络结构
    └── sample_ddim.py
```

DDIM 的代码是在 DDPM 实现基础上修改的，若希望两个文件夹能够**独立运行**，需要各自具有正确的依赖模块；若后续改成共享模块结构，应同步调整 Python 导入路径。

## 05 · 结果与后续计划

- [x] 建立 DDPM / DDIM 两份独立复现记录
- [x] 记录 DDPM 初始训练与 DDIM 50 步采样的配置
- [ ] 核对参考代码的公开使用与再分发许可后，上传源代码
- [ ] 上传真实 Loss 曲线、CSV 日志与生成样本
- [ ] 检查 `ddim_sample()` 内部的更新公式和时间步选择
- [ ] 在相同模型权重、硬件及图片数量下比较实际采样时间与生成质量

## 06 · 参考资料

- [DDPM 原始论文（NeurIPS 2020）](https://arxiv.org/abs/2006.11239)
- [DDIM 原始论文（ICLR 2021）](https://arxiv.org/abs/2010.02502)
- [DDPM 学习参考：Bilibili 教学视频](https://www.bilibili.com/video/BV1vZi6BaEgF/)

本项目主要用于扩散模型原理学习。DDPM 基础代码参考视频教程，不声明为从零独立原创；引用来源不等于自动获得代码再分发许可。

[← 返回科研任务](../README.md)
