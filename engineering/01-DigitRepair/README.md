# DigitRepair

基于 MNIST 的条件扩散数字修复。支持上传、掩码、多候选、去噪回放和结果导出。

[代码](src/) · [实验与功能说明](DETAILS.md) · [指标](results/metrics.json)

## 运行

Python 3.11+，在仓库根目录执行：

```bash
python -m venv .venv
# Windows PowerShell；macOS/Linux 使用 source .venv/bin/activate
.venv/Scripts/Activate.ps1
pip install -r engineering/01-DigitRepair/requirements.txt
python engineering/01-DigitRepair/src/server.py
```

打开 http://127.0.0.1:8082/ 。自带权重 `results/repair.pt`，自动使用 CUDA 或 CPU。静态页面只展示预计算示例，新修复需要本地后端。

上传图会处理为 28×28 灰度图，适合居中的单个数字。没有完整参考图时不计算 RMSE、MAE、PSNR；可见像素保持不变。

## 结果

官方 MNIST 测试集抽取 80 张，每类 8 张；各图采用固定种子。下表为缺损区域平均 RMSE：

| 遮挡类型 | 黑色填充 RMSE | 调和插值 RMSE | DDIM 20 步 | DDIM 50 步 | DDPM 100 步 |
|---|---:|---:|---:|---:|---:|
| 中心 | 0.5441 | 0.4288 | 0.4061 | 0.4055 | 0.3716 |
| 笔画 | 0.4015 | 0.2785 | 0.2215 | 0.2221 | 0.2159 |
| 右半幅 | 0.3462 | 0.4164 | 0.2651 | 0.2667 | 0.2432 |


![中心缺损修复](results/recovery_center.png)

网络参数 483,745，训练 8,000 次更新。以上为单次训练结果，大面积缺损可能改变数字结构。模型配置、其余图表、重训命令与测试见 [详细说明](DETAILS.md)。

[返回工程目录](../README.md)
