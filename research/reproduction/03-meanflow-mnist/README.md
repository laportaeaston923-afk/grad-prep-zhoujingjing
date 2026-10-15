# MNIST · MF vs iMF：一步生成对比

本项目从头训练 **原始 MeanFlow (MF)** 与 **Improved MeanFlow (iMF)**，只比较 **1 NFE（一步）** 生成，不进行五步采样对照。本次对比重点是论文 *Improved Mean Flows* 第 4.1 节和图 3 所指出的 **JVP 方向不同**。

## 1. 要验证什么

两组使用 **同一 MNIST 数据集、同一 DiT 平均速度网络、相同随机种子、训练步数及优化参数**。均使用类别条件，但 **关闭 CFG**；均不用辅助 v-head / 辅助 FM Loss，以免引入额外变量。

| 方法 | JVP 使用的方向 | 训练目标 |
|---|---|---|
| MF | 训练样本的条件速度 `e-x` | 原始 MF 的平均速度损失，等价改写为瞬时速度损失 |
| iMF | 网络预测的 `u_theta(z_t,t,t)` | iMF 的瞬时速度回归 |

共同使用以下形式（停止 JVP 结果的梯度）：

```python
# 对照实验的唯一算法区别：tangent
tangent = (e - x) if variant == "mf" else model(z, t, t, return_v=False)
_, dudt = jvp(u_model, (z, t, r), (tangent, 1, 0))
V = u + (t - r) * stopgrad(dudt)
loss = mse(V, e - x)
```

对固定 JVP 方向，原始 MF 的 u-target 损失与上述 V-loss 形式有相同的停止梯度优化方向，因此这里通过 **切换 JVP 输入** 进行控制变量实验。论文图 3 使用基础 L2，所以新配置设 `adaptive_l2_gamma=1`（即权重恒为 1，**普通 MSE**），没有沿用旧代码的自适应权重 `gamma=0`。

**说明：** `models/dit.py` 仍保留第三方代码的 v-head 参数，便于加载历史权重；此对照训练的 `return_v=False` 不调用它，v-head 不参与 Loss 或优化。iMF 在 `r=t` 边界通过同一个 u-head 获得瞬时速度估计。训练时约 50% 样本设 `r=t`。

## 2. 对照设置

| 设置 | MF | iMF |
|---|---|---|
| 数据集 | MNIST 32×32 灰度图 | 相同 |
| 网络 | DiT, dim=256, depth=6, heads=8, patch=4 | 相同 |
| 训练总步数 | 10,000 | 10,000 |
| Batch / 学习率 | 32 / 1e-4 | 相同 |
| 随机种子 | 42 | 42 |
| CFG / 辅助 v-head | 关闭 / 不使用 | 相同 |
| 损失 | 普通 MSE | 普通 MSE |
| 生成 | **一步（1 NFE）** | **一步（1 NFE）** |

仅改变 JVP 方向，不改变模型大小或采样步数。本设置是 **MNIST 上的核心算法对照**，不是论文的 ImageNet 大规模完整复现。

## 3. 运行方法

在 `code/` 目录运行，先安装合适的 PyTorch / torchvision，再安装剩余依赖：

```bat
cd research\reproduction\03-meanflow-mnist\code
python -m pip install -r requirements.txt
python tests/check_mf_imf.py
```

一次运行两组独立训练、一步 FID 与一步采样计时（GPU 默认 **FP32**，不再默认 BF16）：

```bat
python run_mf_imf_comparison.py --steps 10000 --num-samples 10000 --batch-size 32
```

CPU 上只进行最小流程检查（**不代表模型收敛**）：

```bat
python run_mf_imf_comparison.py --steps 2 --batch-size 2 --cpu --precision no --skip-fid --skip-timing
```

脚本会依次执行 MF 和 iMF，结果保存在 `results/mf-vs-imf-one-step/<时间戳>/` 中；包含双方独立的 `train/loss.csv`、`loss_curve.png`、`loss_diagnostics.png`、`train/images/`、`train/ckpts/`，以及 `fid/`、`timing/`、`comparison.json` 和生成的对照 README。FID 使用两组各 **10,000 张生成图 vs 同一组 10,000 张 MNIST test**；两组采用相同标签与初始噪声。若为少量样本短测，不应作为正式 FID。

### 复现论文 Figure 3：数值稳定性修订 v2（2026-10-15）

你之前的 MNIST 对照中，iMF 在 step 5200/6700 附近出现超过 2 万的 Loss 尖峰。该尖峰是真实训练日志，不应删掉或在绘图时截断；不能直接当作论文结果。为了先排除**实现和数值精度**问题，已作如下修改：

1. **JVP 子图强制 FP32**：包括 iMF 的 `v_theta = u_theta(z,t,t)` 和真正求导的 JVP；MF 和 iMF 一致。BF16 自动混合精度下的 JVP 可能导致数值误差；该修改是稳定性排查措施，**并未证实 BF16 就是此前尖峰的根因**。
2. 新实验两组都默认使用 `--precision no`（GPU FP32）。此前 MF 为 BF16，不能与新 FP32 iMF 混用来声称公平对比。
3. 只画 `t != r` 的原始速度 MSE，记录每个区间的非边界样本数并**按样本数加权**。输出 `loss_comparison.png`；同时输出 `loss_diagnostics.png`，用于检查总 Loss、梯度范数、JVP 方向及结果的 RMS。
4. 普通 L2、无 CFG、同一个 u-head、同一时间采样和结构保持不变；梯度裁剪（5.0）继续沿用，记录的 `grad_norm` 是裁剪**前**的值。只比较一步生成。

**重要：** v2 引入 `comparison_revision=2`，新脚本会拒绝把旧 BF16 的 `step_0010000.pt` 用作新对照的任一组。老权重及 Loss 必须作为历史实验保留，不能通过直接加载旧权重获得修改后算法的曲线。新实验需要 MF、iMF **分别重新训练**。

先在你本地 `E:\\all\\03-meanflow-mnist\\code` 做短测：

```powershell
conda activate meanflow
python -m pip install matplotlib
python tests/check_mf_imf.py
python run_mf_imf_comparison.py --steps 300 --batch-size 32 --skip-fid --skip-timing --precision no
```

短测没问题后再开展正式训练：

```powershell
python run_mf_imf_comparison.py --steps 10000 --num-samples 10000 --batch-size 32 --precision no
```

如果 Windows 出现 `OMP: Error #15`，先检查具体依赖/conda 环境，不要通过 `KMP_DUPLICATE_LIB_OK=TRUE` 强行忽略。训练阶段避免在同一进程内导入 Matplotlib；每 500 step 自动保存 checkpoint。只对**本 v2 实验**使用 `--mf-resume` 或 `--imf-resume` 续训，别混用旧 BF16 权重。

**论文差异：** Figure 3 使用 ImageNet 潜变量、MeanFlow-B/2、约 50,000 训练步；这里仍使用 MNIST 和简化 DiT（10,000 step），因此不能保证出现论文中 MF 上升、iMF 平稳下降的相同趋势。此修订用于提高比较的可解释性，**不是美化曲线，也不是完整复现 Figure 3 的硬件/数据设置**。

## 4. 实验结果

**等待两组实际训练及评估后填写。** 不把过去上传的 50,000 step 权重结果当作本次 MF/iMF 对照，也不预先编造 FID、速度或 Loss。

| 指标 | MF | iMF |
|---|---|---|
| v2 训练 Loss 曲线 | 待重新运行 | 待重新运行 |
| 一步生成图片 | 待运行 | 待运行 |
| 一步 FID ↓ | 待运行 | 待运行 |
| 一步生成耗时 | 待运行 | 待运行 |

比较时主要看 **Loss 稳定性、一步 FID、相同类别/噪声下的数字清晰度**。两组都是 1 NFE，理论上推理调用次数相同；iMF 的差异重点在训练 JVP 输入，不应预先宣称采样速度更快。

## 5. 来源与旧结果

- 论文：[Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447)；[Improved Mean Flows](https://arxiv.org/abs/2512.02012)。
- 第三方 PyTorch 代码：[haidog-yaqub/MeanFlow](https://github.com/haidog-yaqub/MeanFlow)，原始 commit `97cc5182abe9d69e5dcb350268f603fbe31a5897`，MIT 许可见 [code/LICENSE](code/LICENSE)。
- 本次新增 [comparison_flow.py](code/comparison_flow.py)、两个 [configs](code/configs/) 及 [run_mf_imf_comparison.py](code/run_mf_imf_comparison.py)，保留旧代码方便溯源。
- 以前的上传权重、CFG、FID 和采样实验仍保留在 [results/](results/) 作为**历史记录**；它们不是本次控制变量实验的结果。

[返回代码复现目录](../README.md)
