# 实验结果目录

## 新版 DiT 数值稳定性实验（待用户 GPU 正式运行）

请参考 ../README.md，先用 --fig3-only --steps 4000 做短程诊断，然后确认稳定后续训至 20000。

新版真实结果将由 code/plot_results.py 生成，保存在 results/dit_stable/：

- fig3_loss_dit.png — 使用完整、真实、未经裁剪的 MF 与 iMF 非对角损失日志（原值和滑动平均同时保留）。
- one_step_process_dit.png — 噪声 → 平均速度（显示时归一化） → 一步生成的数字。
- mf_vs_imf_dit.png — 同类标签和同一初始噪声下对比。
- cfg_scales_dit.png — 同一个 iMF-CFG 模型在不同引导尺度下的一步生成。

可以再生成指标 metrics.csv 和 mf_vs_imf_metrics_dit.png，注意不是标准 Inception FID。

### 数值诊断

两种模型的真实训练日志在本地 runs_dit_stable/mf/train.csv 和 runs_dit_stable/imf/train.csv。新记录包含：

- offdiag_v_loss、diag_v_loss：t≠r 和 t=r 的损失；
- jvp_rms、jvp_max：JVP 自身幅值；
- weighted_jvp_rms、weighted_jvp_max：实际乘上 (t-r) 后的 JVP 修正项；
- grad_norm_mean、grad_norm_max、learning_rate：更新数值稳定性的诊断指标。

以前的训练图对应旧优化器设置，不与新实验混用。当前没有上传虚构训练结果。
