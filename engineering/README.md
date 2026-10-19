# 工程项目

| 工程 | 内容 | 运行方式 |
|---|---|---|
| [01 · GenCompare](01-GenCompare/) | 七种生成方法的优缺点、真实小图像生成实验、原理可视化、自定义模型结果导入 | 直接打开 `01-GenCompare/src/index.html`；详见该项目 README |
| [02 · DigitRepair](02-DigitRepair/) | 条件扩散手写数字修复：上传、画掩码、实时推理、多候选、回放、参考评估与下载 | 启动 Python 本地服务；详见该项目 README |

两个项目相互独立，代码、权重、结果和截图分别保存在各自目录。第二个工程不覆盖第一个工程。

## 快速启动

**工程 01（无需安装依赖）**：下载仓库后打开 `engineering/01-GenCompare/src/index.html`。

**工程 02（实际模型推理）**：在仓库根目录执行：

```bash
pip install -r engineering/02-DigitRepair/requirements.txt
python engineering/02-DigitRepair/src/server.py
```

访问 `http://127.0.0.1:8082/`。静态打开第二个工程页面只能体验明确标注的预计算示例，新的修复需要 Python 服务。
