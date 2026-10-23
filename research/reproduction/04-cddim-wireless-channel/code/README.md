# cDDIM 对照代码

One-shot U-Net 的训练与 ID / OOD 对照。

| 文件 | 用途 |
|---|---|
| [src/model.py](src/model.py) | 条件 U-Net 网络 |
| [src/dataset.py](src/dataset.py) | 信道数据读取与预处理 |
| [src/train_oneshot.py](src/train_oneshot.py) | One-shot 模型训练与保存 |
| [src/compare_oneshot.py](src/compare_oneshot.py) | 对齐 ID/OOD 测试位置并绘制两模型 CDF |
| [src/plot_oneshot_loss.py](src/plot_oneshot_loss.py) | One-shot 训练 Loss 可视化 |
| [run_oneshot_compare.bat](run_oneshot_compare.bat) | 一键训练及评估 |
| [resume_oneshot.bat](resume_oneshot.bat) | 继续 One-shot 训练 |

## 运行

环境：Windows、Python、PyTorch、NumPy、SciPy、Matplotlib，可使用 CUDA。

运行前需把原 cDDIM 模型 `outputs/paper/latest.pt`、QuaDRiGa MAT 数据、此前 ID/OOD CDF 的 CSV 和 JSON 放回脚本要求的位置。

```powershell
conda activate deepStudy
.\run_oneshot_compare.bat
```

运行后，新结果输出至 `outputs/oneshot/`，不会覆盖 cDDIM 权重。

**说明：** 这里只包含 One-shot 对照补丁；原 cDDIM 全部训练和 MATLAB 数据生成脚本仍需从本机项目补齐。大型 `.mat` 数据与 `.pt` 权重未上传。

[返回项目 README](../README.md)
