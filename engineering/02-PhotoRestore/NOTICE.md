# 来源与使用范围

## 模型与第三方代码

本项目没有从零训练照片超分辨率模型，也不声称提出新的扩散算法。推理采用 [ResShift 官方仓库](https://github.com/zsyOAOA/ResShift) 的网络、残差平移扩散与官方预训练权重。

- 固定上游版本（journal 分支）：`bb03b7d21614cace01787e097c8a6ab6b945227d`。
- 使用模型：`resshift_realsrx4_s4_v3.pth` 和 `autoencoder_vq_f4.pth`，来源为官方 `v2.0` release。
- 推理所需的上游源码子集在 `vendor/resshift/`，文件清单见 `vendor/UPSTREAM.json`。保留 [上游 LICENSE](vendor/resshift/LICENSE)。本地推理适配、Web 服务、交互、校验、实验脚本为本工程新增实现。
- 适配层将 VQ 单头注意力换为等价的 PyTorch SDPA；没有改变权重和扩散方程。另加入分块、重叠融合、任务进度与取消。
- **上游采用 S-Lab License 1.0，允许符合其条款的非商业用途。商业使用需向上游另行取得许可。** 项目和截图不会改变模型的授权条件。

论文：Zongsheng Yue et al., *ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting*, NeurIPS 2023。参见 [论文](https://arxiv.org/abs/2307.12348) 与官方仓库。

## 示例图片

三个示例来自 [scikit-image 官方数据集说明](https://scikit-image.org/docs/stable/api/skimage.data.html)，随工程保存源图用于复现实验：

| 文件夹 | 原图 | 来源 / 授权说明 |
|---|---|---|
| `results/examples/cat/` | Chelsea | Stefan van der Walt，CC0 |
| `results/examples/coffee/` | Coffee | Rachel Michetti，CC0；照片场景为 Pikolo Espresso Bar |
| `results/examples/astronaut/` | Eileen Collins | NASA，官方数据集注明为公有领域 |

`source.png` 是示例源图；`reference.png` 为调整尺寸的完整参考；`input.png` 是人工降质输入；`output.png` 为本机实际推理结果。它们不构成真实用户失焦照片的数据集。

当前使用官方 journal 扩展版本的 v3 模型与对应实现，扩展论文见 [期刊扩展论文](https://arxiv.org/abs/2403.07319)。旧版 v1 的来源与指标在 results/quality-check/ 保留。
