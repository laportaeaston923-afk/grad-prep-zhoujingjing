# cDDIM 实验结果

以下图片来自本机实际运行截图，压缩后保存，不是论文原图，也不是重新编造的数据。

| 文件 | 内容 |
|---|---|
| [loss.webp](loss.webp) | 200 Epoch 训练 Loss |
| [fig5_style_comparison.webp](fig5_style_comparison.webp) | 5 组生成/参考波束域信道热力图 |
| [fig6_id_cdf.webp](fig6_id_cdf.webp) | ID 0–100m，N=500，256 Steps |
| [fig7_ood_cdf.webp](fig7_ood_cdf.webp) | OOD 100–200m，N=500，256 Steps |
| [fig6_id_vs_fig7_ood.webp](fig6_id_vs_fig7_ood.webp) | ID 与 OOD CDF 对比 |
| [fig7_train_vs_ood_positions.webp](fig7_train_vs_ood_positions.webp) | 训练与 OOD 测试位置 |

## 结果概览

训练末轮 Loss ≈ 0.05043；测试指标为最强发射波束索引误差 `D=|b_generated-b_reference|`。

- **ID：** D≤2 的比例约 **80%**。
- **OOD：** D≤2 的比例约 **69%**。
- **泛化：** 训练区域外的波束索引匹配下降，尚未与 GAN、One-shot U-Net 等基线对照。

本机模型 `latest.pt`、QuaDRiGa MAT 信道数据和原始评估 CSV 不在本次聊天可直接获取的文件中，因此未上传，不用编造的数值替代它们。
