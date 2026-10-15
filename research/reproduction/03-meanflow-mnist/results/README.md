# 结果图（完成真实训练后生成）

运行 python code/run_all.py --steps 20000 生成以下四张：

- fig3_loss_mnist.png — 无 CFG、只计算 t ≠ r、普通 MSE 的 MF / iMF Loss。
- cfg_scales.png — 同一 iMF CFG 模型、相同噪声下 ω=1/1.5/2/3/4。
- one_step_process.png — z1 随机噪声、u 平均速度、z0 一步结果（1-NFE）。
- mf_vs_imf.png — 同一初始噪声和相同类别条件的 MF / iMF 生成。

额外运行 python code/evaluate.py --train-classifier 可生成 mf_vs_imf_metrics.png 与 metrics.csv。

目前没有声称为 MNIST 正式训练所得的图片；请先完成训练再提交真正的实验结果。