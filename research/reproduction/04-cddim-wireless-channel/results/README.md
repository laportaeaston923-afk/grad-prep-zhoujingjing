# cDDIM / One-shot U-Net 结果

本目录保留本机实验的原始 PNG 结果图。训练信道 10,000 组、200 Epoch，ID/OOD 各 500 个测试位置。

## Results

| 文件 | 内容 |
|---|---|
| [loss.png](loss.png) | cDDIM 200 Epoch 训练 Loss |
| [fig5_three_way_ID.png](fig5_three_way_ID.png) | Fig. 5 风格：cDDIM、One-shot 与真实信道的对比 |
| [fig6_id_cddim_vs_oneshot.png](fig6_id_cddim_vs_oneshot.png) | Fig. 6：ID 区域两模型 CDF 对比 |
| [fig7_ood_cddim_vs_oneshot.png](fig7_ood_cddim_vs_oneshot.png) | Fig. 7：OOD 区域两模型 CDF 对比 |
| [fig6_id_vs_fig7_ood.png](fig6_id_vs_fig7_ood.png) | cDDIM ID / OOD 的 CDF 对比 |
| [fig7_train_vs_ood_positions.png](fig7_train_vs_ood_positions.png) | 训练区域及 OOD 测试位置 |

## 对照结果（D≤2）

| 区域 | cDDIM | One-shot U-Net |
|---|---:|---:|
| ID：0–100 m | 约 79% | 约 75% |
| OOD：100–200 m | 约 69% | 约 57% |

上述比例此前从本机测试图近似读取，精确数值应以本机保存的 CSV 为准。结果表现为：ID 两者较接近，OOD 中 cDDIM 主发射波束索引的匹配率更高。

说明：原先用于展示的 WEBP 压缩截图与 SVG 近似重绘图已从当前目录移除，由实际上传的 PNG 替代。论文 Fig. 5–7 完整对照还包括 cGAN、无 DFT cDDIM 等基线，当前不是严格的数值复现。
