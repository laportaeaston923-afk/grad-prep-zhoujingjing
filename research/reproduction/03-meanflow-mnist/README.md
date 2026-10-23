# MeanFlow / iMF · MNIST

MF、iMF、CFG 各完成 20,000 步。iMF 的 FID 低于 MF；Loss 未呈现论文中的持续下降，CFG 的类别控制仍较弱。

[代码](code/) · [结果汇总](results/dit_stable/结果汇总.md) · [实验配置](results/dit_stable/实验说明.md)

## 结果

EMA、1-NFE、seed=2026；最终 FID 使用 10,000 张 MNIST 测试图和 10,000 张生成图。

| 模型 | 最终 FID-10000 ↓ | 最后 1,000 步 t≠r loss | 类别一致率 ↑ |
|---|---:|---:|---:|
| MF | 237.38 | 4.5237 | 12.88% |
| iMF | **211.80** | **2.7897** | 11.34% |
| iMF-CFG（ω=4，扫描最优） | 222.73 | 5.9071* | 10.74% |


*CFG 目标不同，Loss 不与 MF / iMF 直接比较。类别一致率由测试准确率 98.68% 的独立分类器计算。过程曲线采用 FID-2000，最终结果采用 FID-10000；不与论文 ImageNet FID-50K 直接比较。

![Loss、FID 与 CFG](results/dit_stable/comparison_curves.png)

## 安装与运行

在本目录运行命令。先安装适配 NVIDIA 显卡的 CUDA PyTorch，再安装依赖，建议使用独立 Python 环境：

```sh
python -m pip install -r requirements.txt
python code/smoke_test.py
```

分阶段训练并保留 4k/8k/12k/16k/20k 权重，自动生成 FID-2000 曲线、训练日志和分类器指标：

```sh
python code/continue_experiments.py
```

该脚本支持读取已有 checkpoint 续训，新实验结果写入 `results/new_run/`，保留已上传的原始结果。本次训练权重和 MNIST 数据遵循仓库忽略规则保留在实验机器，不包含在 Git 中。重新训练后可执行：

```sh
python code/plot_results.py --device cuda --results results/new_run
python code/final_figures.py --results results/new_run
python code/fid_curve.py --runs-root runs_dit_stable --data-root data --results results/new_run --samples 2000
python code/fid_curve.py --runs-root runs_dit_stable --data-root data --results results/new_run/fid10000 --steps 20000 --samples 10000
```

新实验使用独立结果目录。数据、权重保存在实验机器；评估需要 CUDA，首次评估需下载 Inception 权重。`legacy_unet/` 为旧实现，当前模型为 `code/models/dit.py`。

参考：[iMF 论文](https://arxiv.org/html/2512.02012v1) · [第三方实现](https://github.com/haidog-yaqub/MeanFlow) · [pytorch-fid](https://github.com/mseitzer/pytorch-fid) · [上游许可](UPSTREAM_LICENSE)

[返回复现目录](../README.md)
