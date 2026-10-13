# One-shot U-Net 对照代码

此目录包含 One-shot U-Net 的训练与评估代码，不覆盖原先 cDDIM 实验。

```text
code/
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── train_oneshot.py
│   ├── compare_oneshot.py
│   └── plot_oneshot_loss.py
├── run_oneshot_compare.bat
└── resume_oneshot.bat
```

- `train_oneshot.py` 从 `outputs/paper/latest.pt` 读取训练划分和网络配置，使用随机噪声 + 三维位置 **直接预测干净信道**，保存 One-shot 的新权重及真实 MSE Loss。
- `compare_oneshot.py` 读取保存的 cDDIM ID/OOD 统计结果，分别在相同位置计算 One-shot 波束索引误差 CDF 并作图。
- `plot_oneshot_loss.py` 绘制 One-shot 自己的 MSE 曲线；注意它与 cDDIM 噪声预测 MSE 不是同一个目标。
- 需要 `numpy scipy matplotlib torch`、正式 QuaDRiGa `.mat` 文件、先前 cDDIM checkpoint、Fig.6/7 的 CSV/JSON；这些大型文件不在仓库中。

PowerShell 示例：

```powershell
conda activate deepStudy
.\run_oneshot_compare.bat
```

注意 **`run_oneshot_compare.bat` 使用相对路径**，请在 `code/` 目录运行。仓库保存的是当时编写的 One-shot 补丁脚本，而不是通过 GitHub CI 验证过的完整训练环境。cDDIM 原始训练脚本与本机实验产物需从实验机另行保存和对应版本核对。

[返回项目](../README.md)
