# DiT 真实实验结果

请运行 python code/run_all.py --steps 20000 --batch-size 64，再把自动生成的真实图上传此处。

四张目标图均在 results/dit/：

1. fig3_loss_dit.png — MF/iMF JVP 方向实验，plain MSE、无 CFG、仅 r≠t 的样本。浅色原始值，实线滑动平均值。
2. cfg_scales_dit.png — 同一 iMF-CFG DiT，ω=1/1.5/2/3/4。
3. one_step_process_dit.png — 噪声 z1 → 平均速度 u（灰度仅用于展示） → 最终图像 z0=z1-u，1 NFE。
4. mf_vs_imf_dit.png — 同样随机噪声、目标数字和模型规模下两种方法的一步生成结果。

可选：运行 python code/evaluate.py --train-classifier，得到 metrics.csv 和 mf_vs_imf_metrics_dit.png（MNIST feature Fréchet，而非 Inception FID）。

**目前仅提供完整生成代码，尚未得到用户 GPU 真实训练结果；没有放置伪造曲线或生成图。**

旧 U-Net 版脚本已经归档在 ../legacy_unet/，不可与 DiT 图混为一谈。
