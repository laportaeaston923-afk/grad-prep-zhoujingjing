# DDPM 实验图片

这里计划放置 **3 张最终展示图片**（请直接上传原始 PNG 文件，不要上传 IDE 或文件管理器的截图）。

| 文件名 | 来源 | 内容 |
|---|---|---|
| `training_loss.png` | 本地 `train_results_120_amp/training_loss.png` | 从头训练 120 Epoch 的 AMP Loss |
| `ddpm_samples.png` | DDPM 采样程序产生的 4×4 图片网格 | 16 张生成图片 |
| `denoising_trajectory.png` | 本地项目根目录同名图片 | 从噪声到生成结果的逐步去噪过程 |

**实验追溯提醒：** Loss 明确对应 120 Epoch AMP 训练；此前 `sample.py` 显示加载的是项目根目录旧 `ddpm.pth`，不一定对应 AMP 模型。因此，请核对生成样本和去噪轨迹使用的权重，勿在未核实前将三张图写成同一训练 checkpoint 的结果。

上传前确认图片确实来自本人的实验输出，并确认参考代码的公开发布许可。
