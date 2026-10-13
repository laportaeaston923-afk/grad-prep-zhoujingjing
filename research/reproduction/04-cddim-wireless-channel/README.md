# cDDIM · 无线信道生成

基于论文 *Generating High Dimensional User-Specific Wireless Channels Using Diffusion Models* 进行阶段性复现。参考：[论文](https://arxiv.org/abs/2409.03924) · [作者官方代码](https://github.com/taekyunl/cDDIM)。

- **[code/](code/)**：cDDIM 项目配套的 One-shot U-Net 训练与 ID/OOD 评估代码，以及运行脚本。
- **[results/](results/)**：训练 Loss、Fig. 5 风格热力图、ID/OOD CDF、**新增 One-shot 对照图**。

## 实验设置

QuaDRiGa BERLIN_UMa_LOS、28 GHz；复数 MIMO 信道矩阵大小 4×32，三维位置条件。当前实验采用 **10,000 组训练信道、200 Epoch、500 个 ID 与 500 个 OOD 位置**。cDDIM 采用 256 步确定性去噪；One-shot U-Net 直接将噪声和位置映射为信道，采样时只调用一次 U-Net。两者采用同一类网络骨干，并分别训练。

| 指标（约） | cDDIM | One-shot U-Net |
|---|---:|---:|
| ID：D = 0 | 27% | 25% |
| ID：D ≤ 2 | **79%** | 75% |
| OOD：D = 0 | **25%** | 17% |
| OOD：D ≤ 2 | **69%** | 57% |

这里的 **D** 是生成信道和参考信道最强发射波束索引之差的绝对值，CDF 是满足误差 ≤ D 的测试比例。以上数值来自用户本机实验图的近似读取；原始 CSV 未上传，故没有伪造精确数值。

## 结果与分析

在 ID 区域，两个模型总体接近；在 100–200 m OOD 区域，cDDIM 的曲线整体高于 One-shot。以 D≤2 为例，OOD 近似匹配比例由 57% 提高至 69%，与论文强调的 cDDIM 分布外优势趋势一致。但当前结果不能证明差距只由 256 步引起。

**限制：** 这是用户本机的简化版复现，不是论文 Fig. 5–7 的完整数值复现。论文这组对比采用 100 训练样本、10,000 生成及参考信道，并包含 cGAN、无 DFT cDDIM 等模型。当前未提供这些完整对照、原始 One-shot Loss/权重及 CSV；仓库不应将示例数据、截图估读值视为原始指标文件。

## 如何运行

将仓库的 `code/` 作为 Windows 项目根目录：准备好 `outputs/paper/latest.pt`、相应 QuaDRiGa MAT 文件以及此前 Fig. 6/7 生成的 CDF CSV 和 JSON。激活原有 `deepStudy` 环境，进入 `code/` 运行：

```powershell
.\run_oneshot_compare.bat
```

该脚本在缺少 One-shot 权重时才训练，已有权重则跳过训练并评估；不覆盖现有 cDDIM checkpoint。训练中断后可运行 `resume_oneshot.bat`。

之前的 cDDIM 训练与 Fig. 6/7 脚本仍保存在本机原项目中；本次仓库更新主要补充 One-shot 对照源码。不要将 GitHub 目录误认为已包含完整原始数据与所有 cDDIM 脚本。

[返回复现目录](../README.md)
