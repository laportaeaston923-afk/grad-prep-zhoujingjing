# cDDIM · Results

这里保存的是本机实验上传的**原始 PNG 图片**，不再引用之前删除的 WEBP/SVG。

| 图片 | 说明 |
|---|---|
| [loss.png](loss.png) | cDDIM 200 Epoch 训练 Loss |
| [fig5_three_way_ID.png](fig5_three_way_ID.png) | cDDIM、One-shot、参考信道热力图 |
| [fig6_id_cddim_vs_oneshot.png](fig6_id_cddim_vs_oneshot.png) | ID：cDDIM vs One-shot CDF |
| [fig7_ood_cddim_vs_oneshot.png](fig7_ood_cddim_vs_oneshot.png) | OOD：cDDIM vs One-shot CDF |
| [fig6_id_vs_fig7_ood.png](fig6_id_vs_fig7_ood.png) | cDDIM ID / OOD CDF |
| [fig7_train_vs_ood_positions.png](fig7_train_vs_ood_positions.png) | 训练与 OOD 测试位置 |

## 结果说明

训练 10,000 组信道、200 Epoch；ID 和 OOD 各测试 500 个位置。固定主发射波束索引误差 **D ≤ 2**：

- **ID：** cDDIM 约 79%，One-shot U-Net 约 75%。
- **OOD：** cDDIM 约 69%，One-shot U-Net 约 57%。

两模型在 ID 较接近；OOD 中 cDDIM 相对更好。比例为截图近似值，并非原始 CSV 的精确读数。此结果是 Fig. 6/7 风格的双模型对照，**不是论文全部基线的完整复现**。

[返回项目 README](../README.md)
