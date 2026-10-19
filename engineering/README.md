# 工程项目

| 工程 | 内容 | 启动地址 |
|---|---|---|
| [01 · DigitRepair](01-DigitRepair/) | 条件扩散手写数字修复：掩码、多候选、回放与评估 | http://127.0.0.1:8082/ |
| [02 · PhotoRestore](02-PhotoRestore/) | 真实照片清晰化：本地登录、预训练扩散超分辨率、上传、对比与下载 | http://127.0.0.1:8083/ |

两个项目独立运行。代码、实验结果与截图分别保存在对应目录，运行方式、模型来源与限制见各自 README。

## 照片清晰化项目快速开始

在仓库根目录执行：

```powershell
pip install -r engineering/02-PhotoRestore/requirements.txt
python engineering/02-PhotoRestore/scripts/setup_models.py
python engineering/02-PhotoRestore/src/server.py
```

打开 http://127.0.0.1:8083/ 。GitHub 仅托管代码，网页推理需要启动本地 Python 服务。已有 E 盘工程的权重已下载，可直接启动。
