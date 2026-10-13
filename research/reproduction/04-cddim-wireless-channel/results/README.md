# cDDIM / One-shot U-Net 结果

实验来自 Windows + CUDA 本机运行；训练信道 10,000，200 Epoch，ID/OOD 各 500 个测试位置。

| 文件 | 内容 |
|---|---|
| [loss.webp](loss.webp) | cDDIM 200 Epoch Loss |
| [fig5_style_comparison.webp](fig5_style_comparison.webp) | cDDIM 和参考信道的 Fig. 5 风格比较 |
| [fig6_id_cdf.webp](fig6_id_cdf.webp) |  cDDIM ID CDF |
| [fig7_ood_cdf.webp](fig7_ood_cdf.webp) |  cDDIM OOD CDF |
| [fig6_id_vs_fig7_ood.webp](fig6_id_vs_fig7_ood.webp) |  cDDIM ID/OOD 曲线比较 |
| [fig7_train_vs_ood_positions.webp](fig7_train_vs_ood_positions.webp) | 训练和 OOD 用户位置 |
| [fig6_id_cddim_vs_oneshot.svg](fig6_id_cddim_vs_oneshot.svg) | ID 两模型对比曲线 |
| [fig7_ood_cddim_vs_oneshot.svg](fig7_ood_cddim_vs_oneshot.svg) | OOD 两模型对比曲线 |


## 两模型对照（D≤2）

| 实验 | cDDIM | One-shot U-Net |
|---|---:|---:|
| ID 0–100 m | 约 79% | 约 75% |
| OOD 100–200 m | 约 69% | 约 57% |

结论：ID 性能接近，OOD 中 cDDIM 的主要发射波束索引匹配率更高。指标来自 500 个测试位置的阶段性实验，与论文中 100 训练/10,000 生成参考样本和多基线设置不同，不能当作完全相同的实验协议。
