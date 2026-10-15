# 03 · Improved MeanFlow（iMF）— MNIST / DiT

**状态：已修改数值不稳定问题的训练设置并补齐诊断；还没有得到新版本的真实训练曲线。** 不保证 MNIST 小 DiT 可以精确复制原论文的 ImageNet 曲线，更不会用处理图像的办法伪造下降趋势。

- 论文：Improved Mean Flows（Fig.3 训练稳定性对照）
- [参考 GitHub：haidog-yaqub/MeanFlow](https://github.com/haidog-yaqub/MeanFlow)（第三方 PyTorch 实现）
- 模型：**原仓库的 models/dit.py / MFDiT，未经修改**，而不是之前的 U-Net。
- 之前的 U-Net 代码放在 legacy_unet/，原 DiT 首轮不稳定实验权重保留在用户电脑的 runs_dit/。

## 结果（新实验运行后生成）

| 用户要求 | 新版结果路径 | 说明 |
|---|---|---|
| 论文 Fig.3 Loss 对比 | results/dit_stable/fig3_loss_dit.png | 同模型/配置的 MF 与 iMF；只画 t≠r 样本的普通 v-loss |
| 不同 CFG 尺度 | results/dit_stable/cfg_scales_dit.png | iMF-CFG，同噪声，ω=1/1.5/2/3/4 |
| 一步生成过程 | results/dit_stable/one_step_process_dit.png | z1 → 预测平均速度 u → z0=z1-u；仅 **1-NFE** |
| MF vs iMF 一步生成 | results/dit_stable/mf_vs_imf_dit.png | 相同模型规模、噪声、数字类别和训练步数 |

使用 code/evaluate.py 可额外得到 MNIST 分类器特征 Fréchet 距离及数字类别一致率。该特征距离**不是标准 Inception FID**，也不对应论文报告的 ImageNet FID。

**注意：现阶段以上 PNG 尚待你本机实际训练生成。** 旧图来自旧训练，不会被当作新版实验结果。

## 代码：使用原 DiT，具体改了什么？

以下源代码与上游相同（已校验 Git blob）：code/models/dit.py、code/data.py、code/upstream_meanflow.py。参考原配置保存在 code/upstream_mnist_config.py，MIT 许可见 UPSTREAM_LICENSE。

**核心 Fig.3 公式没有改变：**

- 原 MF：复合预测 V=u+(t-r)·JVP(u; e-x)，用条件速度 e-x 作 JVP 方向；
- iMF：V=u+(t-r)·JVP(u; vθ)，用边界速度 vθ=u(z,t,t) 作 JVP 方向；
- 两者都使用普通平方误差回归 e-x，保留 stop-gradient。**不加 CFG、不用 adaptive loss、不使用辅助 v-head 损失**。
- 两者各自从相同初始化种子训练；只比较 t≠r 样本，t=r 部分仍参加正常训练。
- iMF-CFG 单独使用原 MFDiT 双头和论文 Algorithm 2 引导目标，不能与 Fig.3 的无 CFG Loss 数值直接比较。

**新增数值稳定性设置（影响实验条件，不能冒充论文原数值）：**

| 项目 | 之前不稳定版本 | 本次默认 |
|---|---|---|
| 最高学习率 | 1e-4，固定 | **2e-5**，前 2,000 步线性 warmup 后余弦衰减 |
| Adam 第二动量 β₂ | 0.999 | **0.95** |
| 梯度裁剪阈值 | 5.0 | **1.0** |
| EMA 权重 | 无 | 0.999（只用在生成图/评估，不改变训练 Loss） |
| JVP 数值记录 | 无 | **JVP RMS / max、(t-r)×JVP RMS / max** |
| 其他记录 | Loss | 增加 t=r / t≠r Loss、速度幅值、梯度范数、学习率 |
| 异常保护 | 无 | 连续两个日志窗口 t≠r Loss 超过 250 则保存并停止 |

**为什么先这么改？** 原来上传的 CSV 表明：t=r 样本的 Loss 通常在 0.3～1.5 附近，而 t≠r 样本飙升至数万。问题集中在包含 (t-r)×JVP 的损失部分。新版不改训练数学目标，先用保守优化器配置和更完善的日志验证根因。改动能否真正改善曲线须由新日志检验。

## 在 Windows / VS Code 运行（先不要再做 60,000 步）

先用 **新的干净环境** 安装适配显卡的 PyTorch；旧 base 环境遇到过 OpenMP 运行时冲突。不要为了正式实验设置 KMP_DUPLICATE_LIB_OK=TRUE。

在 E:\all 下用一个全新的文件夹放这个仓库的 **最新** 03-meanflow-mnist；原来已训练出的 runs_dit/ 和旧 CSV 请备份，不要删。

~~~powershell
conda create -n imfdit python=3.11 -y
conda activate imfdit
~~~

PyTorch 需要先按 [官网安装页面](https://pytorch.org/get-started/locally/) 选择你的 CUDA 版本安装，然后：

~~~powershell
pip install -r requirements.txt
python code\smoke_test.py
~~~

出现 SMOKE TEST PASSED 后，**先只做 MF/iMF 各 4,000 步检查**，无需先跑 CFG：

~~~powershell
python code\run_all.py --fig3-only --steps 4000 --batch-size 64
~~~

两份真实新日志会出现在：

- runs_dit_stable/mf/train.csv
- runs_dit_stable/imf/train.csv

先看新列 weighted_jvp_rms、weighted_jvp_max、grad_norm_max、offdiag_v_loss 是否还有爆炸。**如果仍出现较大尖峰，应该先诊断，不要继续浪费 GPU 时间。**

如果 4,000 步稳定，再从原权重继续到 20,000 步：

~~~powershell
python code\run_all.py --fig3-only --steps 20000 --batch-size 64 --resume
~~~

此命令会恢复模型及优化器，并读取已保存的学习率计划；数据加载器的 shuffle 顺序重启，不保证逐位一致。

在 MF/iMF 曲线满意之后，再训练 CFG，生成全部四图：

~~~powershell
python code\run_all.py --steps 20000 --batch-size 64 --resume
~~~

已经完成的 MF/iMF 会跳过新的训练循环；没有旧 checkpoint 的 imf_cfg 会开始训练。也可以直接做全部三组训练，但首次建议先检查 Fig.3。

生成图和可选定量评估：

~~~powershell
python code\plot_results.py
python code\evaluate.py --train-classifier --samples 2000
~~~

结果文件存入 results/dit_stable/。新版权重存入 runs_dit_stable/，**不会覆盖原 runs_dit/ 和 runs/**。

**若中途触发异常保护：** 先看 CSV 中的 weighted_jvp_rms、weighted_jvp_max 和 grad_norm_max，将日志提供出来，不要删除异常样本、强行裁剪绘图 y 轴或把真实数据替换成论文曲线。

## 项目结构

~~~text
03-meanflow-mnist/
├── README.md
├── UPSTREAM_LICENSE
├── requirements.txt
├── code/
│   ├── models/dit.py         # 原第三方项目的 DiT，原样
│   ├── data.py               # 原 MNIST 预处理，原样
│   ├── upstream_meanflow.py  # 参考实现留档
│   ├── experiment_core.py    # MF Eq.9 / iMF Eq.12 / CFG Algorithm 2
│   ├── train.py              # 稳定训练 + 完整 CSV 诊断 + EMA
│   ├── run_all.py            # Fig3-only / 三组实验 / 续训
│   ├── plot_results.py       # 四组真实结果图
│   ├── evaluate.py           # 可选指标
│   └── smoke_test.py         # 正式训练前测试
├── legacy_unet/              # 旧 U-Net 备份
├── results/dit_stable/       # 本次真实新结果，运行后生成
└── runs_dit_stable/          # 本地 ckpt 和 CSV，不传大权重
~~~

**研究界限：** 原论文 Fig.3 的场景是 ImageNet 256×256 的 MF-B/2；本项目为 MNIST 32×32 + 缩小版原 DiT 的方法级复现。目标是检验 MF/iMF 的稳定性差异，而不是承诺数值与原论文一致。
