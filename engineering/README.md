# 工程项目

当前保留 [01 · DigitRepair：条件扩散手写数字修复工作台](01-DigitRepair/)。

支持上传数字、画掩码、DDPM/DDIM 实时修复、多候选差异展示、去噪回放、实验导入导出。

```bash
pip install -r engineering/02-DigitRepair/requirements.txt
python engineering/02-DigitRepair/src/server.py
```

浏览器访问 http://127.0.0.1:8082/ 。权重、实验结果和截图见项目目录；静态页面只能查看预计算示例。
