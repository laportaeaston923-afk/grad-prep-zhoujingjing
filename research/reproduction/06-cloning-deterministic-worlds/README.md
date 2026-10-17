# 06 · Cloning Deterministic Worlds（CVPR 2026）复现

> **状态：复现流程与单卡运行入口已准备；尚未进行真实 GPU 训练、不能声称达到论文结果。**

论文：[Cloning Deterministic Worlds: The Critical Role of Latent Geometry in Long-Horizon World Models](https://openaccess.thecvf.com/content/CVPR2026/papers/Xia_Cloning_Deterministic_Worlds_The_Critical_Role_of_Latent_Geometry_in_CVPR_2026_paper.pdf)  
官方实现：[XiaFire/Clone_Deterministic_Environment](https://github.com/XiaFire/Clone_Deterministic_Environment) · [Maze9 数据集](https://huggingface.co/datasets/xiazaishuo/cloning-deterministic-worlds-maze9)

## 复现目标

在 **同一份 Maze9 数据集**下比较 **VAE baseline** 与 **GRWM**：

```text
Maze9 图像、动作
       │
       ├── VAE  → 潜在特征 → DFOT 扩散动力学 → 未来画面 ─┐
       └── GRWM → 潜在特征 → DFOT 扩散动力学 → 未来画面 ─┤
                                                       ↓
                                            每一步像素 MSE 曲线对比
```

GRWM 改进的是**状态编码与潜在空间几何**（时间平滑与均匀性）；并未修改 DFOT 扩散模型的核心算法。官方 Maze9 扩散配置为离散时间 + cosine 调度 + `pred_v` + DDIM 10 步采样，不是 Flow Matching/MeanFlow。

## 目录

```text
06-cloning-deterministic-worlds/
├── README.md                 # 中文使用说明
├── upstream.lock.json        # 固定原作者 Git commit
├── code/
│   ├── run.py                # 获取代码、检查数据、分阶段单卡运行
│   └── export_results.py     # 复制真实实验生成的指标/图片
├── notes/PAPER_NOTES.md      # 原理、创新点、图的理解
├── tests/test_pipeline.py    # 适配器单元测试（不训练模型）
└── results/README.md        # 实验状态和真实输出说明
```

官方项目没有随本次审查发现明确 `LICENSE` 文件，因此 **没有直接把上游大段研究代码重新发布到本仓库**。首次运行 `bootstrap` 时克隆固定的官方 commit 到被 `.gitignore` 排除的 `.upstream/official/`；实际训练调用原作者的 `pipeline.py`、`dyna_efficient.py`、`evaluate_maze9_rollout_mse.py` 等代码，不将自己的简化网络冒充 GRWM。

## 0. 运行条件

- **推荐 Linux / Windows WSL2（Ubuntu）**。官方入口为 Bash，不能从原生 Windows CMD/PowerShell 直接运行。
- Python、PyTorch CUDA（版本需与本机驱动匹配）、`git`、`bash`、可用 NVIDIA GPU。
- 官方 Maze9 训练集与验证集：各目录需要 `info.hdf5`、`obs/*.npy`。完整数据量级为数十 GB。低显存机器请先使用 debug 验证，**仍不能保证成功跑通**。
- 官方默认阶段会用多张卡；此项目只调整**两个 AE 训练任务串行调度**，统一设为指定单卡，保留模型/损失/采样代码；单卡训练耗时可能很长。

## 1. 下载官方实现、安装依赖

在本项目根目录打开 VS Code 的 **WSL Ubuntu 终端**：

```bash
python code/run.py bootstrap
python -m pip install -r .upstream/official/requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

`bootstrap` 会检出 `upstream.lock.json` 锁定的源码版本；如果目录已经存在但版本不同，会明确报错，以免混入不同算法版本。安装 PyTorch 建议先按照官方 PyTorch 网站为自己的 GPU 选择 CUDA 安装方法。

## 2. 下载 Maze9 数据

从上述 Hugging Face 数据集页下载：训练 `9x9_n0.8`、验证 `9x9_val`。下载后确保目录**直接**包含 `info.hdf5` 与 `obs/`，示例：

```text
/home/you/data/maze9/9x9_n0.8/info.hdf5
/home/you/data/maze9/9x9_n0.8/obs/...
/home/you/data/maze9/9x9_val/info.hdf5
/home/you/data/maze9/9x9_val/obs/...
```

检查（改成你实际的绝对路径）：

```bash
export TRAIN_DIR=/home/you/data/maze9/9x9_n0.8
export VAL_DIR=/home/you/data/maze9/9x9_val
python code/run.py check
```

## 3. 训练与评价（建议按阶段运行）

**A. debug 流程验证（不是论文指标）**：

```bash
python code/run.py ae         --profile debug --gpu 0
python code/run.py precompute --profile debug --gpu 0
python code/run.py dynamics   --profile debug --gpu 0
python code/run.py eval       --profile debug --gpu 0
python code/export_results.py
```

debug 仅把 AE 限制到部分样本（训练 128 / 验证 32）、1 Epoch，扩散训练 200 steps、评估 16 个样本、采样 4 步；**潜在提取仍会读取完整输入数据**，所以不能当成纯几分钟的小型实验，也不能拿 debug MSE 声称论文改进结论。建议先执行 `python code/run.py ae --dry-run` 检查参数。

**B. 更接近论文设置（单卡顺序训练，时间/显存开销大）**：

```bash
python code/run.py ae         --profile paper --gpu 0
python code/run.py precompute --profile paper --gpu 0
python code/run.py dynamics   --profile paper --gpu 0
python code/run.py eval       --profile paper --gpu 0
python code/export_results.py
```

**同一次实验四个阶段必须保持相同 profile**，否则扩散权重的目录名称会不同。若 GPU 不够，先报告报错与显卡型号，再有依据地调 Batch Size，不要在未验证时把参数调整说成论文原设置。

## 4. 结果与组会汇报

完整评价执行成功后，官方保存 `outputs/maze9_metrics/maze9_vae_vs_grwm_norm_mse_per_step.pdf/png/csv` 等输出；`export_results.py` 会将**真实存在**的指标和曲线复制到本目录 `results/`。训练日志/权重原样保存在 `.upstream/official/logs`、`checkpoints`、`ckpts`，不会上传超大模型文件。

建议组会只展示三个核心内容：**（1）VAE 与 GRWM 方法对照（2）潜在聚类可视化（3）多步 MSE 对比曲线**。官方提供 `visualization/maze9_latent_clustering_and_mse.ipynb`，需要完成训练并提供对应文件后才能展示真实结果。

[论文学习笔记](notes/PAPER_NOTES.md) · [结果状态](results/README.md) · [官方脚本](https://github.com/XiaFire/Clone_Deterministic_Environment/blob/main/run_maze9_vae_grwm.sh)

## 5. 本次实际验证边界

可用 `python -m unittest discover -s tests -v` 测试适配器参数、顺序调度和结果导出逻辑。**单元测试通过 ≠ 已训练出世界模型**。本次提交不包含原论文数字、假 Loss、预制的生成图或虚假的模型权重。
