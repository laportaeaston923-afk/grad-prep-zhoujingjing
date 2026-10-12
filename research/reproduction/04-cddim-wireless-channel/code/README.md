# 代码

这次 cDDIM 的 **完整 Python、BAT 和 MATLAB 实验脚本目前保存在运行实验的 Windows 电脑中**，并未随用户上传到当前对话。为避免把不完整或不同版本的代码冒充成实际运行版本，仓库这里只保留源码整理说明。

请从本机项目 `E:\all\cDDIM_reproduction_for_E_all\cDDIM_reproduction` 复制以下文件到本目录：

- `src/`（包含 `train.py`、`model.py`、`diffusion.py`、`dataset.py`、`evaluate.py`、`fig6_id.py`、`fig7_ood.py` 等）
- `matlab/generate_ood_quadriga.m`
- `requirements.txt` 和 `run_paper.bat`、`run_fig6_id.bat`、`run_fig7_ood.bat`

保留原始脚本和记录；若改变目录结构，需同步检查 BAT 中的 `outputs\paper` 路径。权重 `*.pt`、原始 MAT 数据 `*.mat` 不应直接上传。

作者官方实现：[taekyunl/cDDIM](https://github.com/taekyunl/cDDIM)。本机跑的是根据其方法重新整理的简化版，不能直接用官方脚本替代本机实测代码。
