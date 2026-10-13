# cDDIM · 无线信道生成

复现论文 *Generating High Dimensional User-Specific Wireless Channels Using Diffusion Models*，使用 QuaDRiGa 信道数据对比 **cDDIM** 与 **One-shot U-Net**。

论文：[arXiv](https://arxiv.org/abs/2409.03924) · 官方代码：[taekyunl/cDDIM](https://github.com/taekyunl/cDDIM)

## Results

| 实验结果 | 图片 |
|---|---|
| cDDIM 训练 Loss | [loss.png](results/loss.png) |
| Fig. 5：cDDIM / One-shot / Reference 信道热力图 | [fig5_three_way_ID.png](results/fig5_three_way_ID.png) |
| Fig. 6：ID 区域两模型 CDF 对比 | [fig6_id_cddim_vs_oneshot.png](results/fig6_id_cddim_vs_oneshot.png) |
| Fig. 7：OOD 区域两模型 CDF 对比 | [fig7_ood_cddim_vs_oneshot.png](results/fig7_ood_cddim_vs_oneshot.png) |
| cDDIM 的 ID / OOD CDF 对比 | [fig6_id_vs_fig7_ood.png](results/fig6_id_vs_fig7_ood.png) |
| 训练 / OOD 用户位置分布 | [fig7_train_vs_ood_positions.png](results/fig7_train_vs_ood_positions.png) |

[查看全部 Results](results/)

## Code

[查看代码](code/)：One-shot U-Net 模型、训练、CDF 评估和 Windows 运行脚本。


## 实验设置与结论

QuaDRiGa BERLIN_UMa_LOS，28 GHz，4×32 复数 MIMO 信道；以用户三维位置为条件。

- 训练数据：10,000 组，200 Epoch；ID / OOD 各测试 500 个位置。
- cDDIM：256 步生成；One-shot U-Net：一步生成。
- 指标：最强发射波束索引误差 `D` 的 CDF，即误差不超过 `D` 的信道比例。

| 误差 D ≤ 2（截图近似值） | cDDIM | One-shot U-Net |
|---|---:|---:|
| ID（0–100 m） | 约 79% | 约 75% |
| OOD（100–200 m） | **约 69%** | 约 57% |

**观察：** ID 区域两模型表现接近；OOD 区域 cDDIM 的主波束索引匹配率更高，表现出更好的分布外泛化趋势。

[返回复现总览](../README.md)
