# grad-prep-zhoujingjing

研究生科研记录：论文阅读、代码复现与工程实践。

## 个人简介

- 姓名：周晶晶
- 研究方向：扩散模型与无线感知

## 任务进度总览

### 科研任务

| 项目 | 状态 | 进度 |
|---|---|---|
| 论文阅读 | 持续更新 | 已整理 11 篇笔记 |
| 代码复现 | 持续更新 | 已整理 6 个项目，完成范围见下表 |
| 研究想法 | 待验证 | [查看研究问题](research/ideas/) |

### 工程实践

| 项目 | 状态 | 内容 |
|---|---|---|
| [DigitRepair](engineering/01-DigitRepair/) | 已有可运行版本 | 条件扩散手写数字修复 |

## 目录结构

| 路径 | 内容 |
|---|---|
| [research/paper-notes/](research/paper-notes/) | 论文阅读笔记 |
| [research/papers/](research/papers/) | 原文链接与 PDF |
| [research/reproduction/](research/reproduction/) | 复现代码与结果 |
| [research/ideas/](research/ideas/) | 研究想法 |
| [engineering/](engineering/) | 工程项目 |


<details open>
<summary>详细目录分布</summary>

- [engineering/](engineering/)：工程项目
  - [01-DigitRepair/](engineering/01-DigitRepair/)：手写数字修复
    - [results/](engineering/01-DigitRepair/results/)：权重、指标与结果图
    - [screenshots/](engineering/01-DigitRepair/screenshots/)：界面截图
    - [src/](engineering/01-DigitRepair/src/)：前后端与模型代码
      - [ml/](engineering/01-DigitRepair/src/ml/)：模型、训练与评估
      - [web/](engineering/01-DigitRepair/src/web/)：网页界面
    - [tests/](engineering/01-DigitRepair/tests/)：测试
- [research/](research/)：科研资料
  - [ideas/](research/ideas/)：研究想法
  - [paper-notes/](research/paper-notes/)：11 篇论文笔记
  - [papers/](research/papers/)：论文原文、PDF 与来源索引
  - [reproduction/](research/reproduction/)：六个复现项目
    - [01-ddpm-cifar10/](research/reproduction/01-ddpm-cifar10/)：DDPM · CIFAR-10
      - [code/](research/reproduction/01-ddpm-cifar10/code/)：代码
      - [results/](research/reproduction/01-ddpm-cifar10/results/)：结果图与实验记录
    - [02-ddim-cifar10/](research/reproduction/02-ddim-cifar10/)：DDIM · CIFAR-10
      - [code/](research/reproduction/02-ddim-cifar10/code/)：代码
      - [results/](research/reproduction/02-ddim-cifar10/results/)：结果图与实验记录
    - [03-meanflow-mnist/](research/reproduction/03-meanflow-mnist/)：MF / iMF / CFG · MNIST
      - [code/](research/reproduction/03-meanflow-mnist/code/)：代码
        - [models/](research/reproduction/03-meanflow-mnist/code/models/)：DiT 模型
      - [legacy_unet/](research/reproduction/03-meanflow-mnist/legacy_unet/)：旧 U-Net 实现
      - [results/](research/reproduction/03-meanflow-mnist/results/)：结果图与实验记录
        - [dit_stable/](research/reproduction/03-meanflow-mnist/results/dit_stable/)：DiT 实验图表、日志与指标
          - [fid10000/](research/reproduction/03-meanflow-mnist/results/dit_stable/fid10000/)：FID-10000 评估
    - [04-cddim-wireless-channel/](research/reproduction/04-cddim-wireless-channel/)：cDDIM 无线信道对照
      - [code/](research/reproduction/04-cddim-wireless-channel/code/)：代码
        - [src/](research/reproduction/04-cddim-wireless-channel/code/src/)：One-shot U-Net 对照代码
      - [results/](research/reproduction/04-cddim-wireless-channel/results/)：结果图与实验记录
    - [05-widar3-csi-dfs-bvp/](research/reproduction/05-widar3-csi-dfs-bvp/)：CSI → DFS → BVP
      - [code/](research/reproduction/05-widar3-csi-dfs-bvp/code/)：代码
        - [BVP/](research/reproduction/05-widar3-csi-dfs-bvp/code/BVP/)：BVP 算法
        - [Data/](research/reproduction/05-widar3-csi-dfs-bvp/code/Data/)：CSI 输入数据说明
        - [Doppler2VelocityMapping/](research/reproduction/05-widar3-csi-dfs-bvp/code/Doppler2VelocityMapping/)：多普勒与速度映射
        - [csi_tool_box/](research/reproduction/05-widar3-csi-dfs-bvp/code/csi_tool_box/)：CSI 读取与校正
        - [tftb/](research/reproduction/05-widar3-csi-dfs-bvp/code/tftb/)：时频分析依赖
          - [mfiles/](research/reproduction/05-widar3-csi-dfs-bvp/code/tftb/mfiles/)：MATLAB 函数
      - [results/](research/reproduction/05-widar3-csi-dfs-bvp/results/)：结果图与实验记录
        - [userA-1-1-1-1/](research/reproduction/05-widar3-csi-dfs-bvp/results/userA-1-1-1-1/)：单样本结果
    - [06-csi-bert2/](research/reproduction/06-csi-bert2/)：CSI-BERT2
      - [code/](research/reproduction/06-csi-bert2/code/)：代码
        - [src/](research/reproduction/06-csi-bert2/code/src/)：作者代码
          - [WiCount/](research/reproduction/06-csi-bert2/code/src/WiCount/)：WiCount 数据与说明
            - [data_process_example/](research/reproduction/06-csi-bert2/code/src/WiCount/data_process_example/)：数据处理示例
          - [img/](research/reproduction/06-csi-bert2/code/src/img/)：作者文档配图
      - [results/](research/reproduction/06-csi-bert2/results/)：结果图与实验记录
        - [figures/](research/reproduction/06-csi-bert2/results/figures/)：结果图
          - [full_reconstruction/](research/reproduction/06-csi-bert2/results/figures/full_reconstruction/)：完整重建图
          - [generated_examples/](research/reproduction/06-csi-bert2/results/figures/generated_examples/)：生成示例
        - [runs/](research/reproduction/06-csi-bert2/results/runs/)：各组实验日志与指标
          - [verified_run/](research/reproduction/06-csi-bert2/results/runs/verified_run/)
          - [wicount_gru/](research/reproduction/06-csi-bert2/results/runs/wicount_gru/)
          - [wicount_lstm/](research/reproduction/06-csi-bert2/results/runs/wicount_lstm/)
          - [wicount_paper_v2/](research/reproduction/06-csi-bert2/results/runs/wicount_paper_v2/)
          - [wicount_rnn/](research/reproduction/06-csi-bert2/results/runs/wicount_rnn/)
          - [wifall_gru/](research/reproduction/06-csi-bert2/results/runs/wifall_gru/)
          - [wifall_lstm/](research/reproduction/06-csi-bert2/results/runs/wifall_lstm/)
          - [wifall_paper_v2/](research/reproduction/06-csi-bert2/results/runs/wifall_paper_v2/)
          - [wifall_rnn/](research/reproduction/06-csi-bert2/results/runs/wifall_rnn/)
          - [wig_paper_config/](research/reproduction/06-csi-bert2/results/runs/wig_paper_config/)
          - [wigesture_gru/](research/reproduction/06-csi-bert2/results/runs/wigesture_gru/)
          - [wigesture_lstm/](research/reproduction/06-csi-bert2/results/runs/wigesture_lstm/)
          - [wigesture_no_arl/](research/reproduction/06-csi-bert2/results/runs/wigesture_no_arl/)
          - [wigesture_no_gan/](research/reproduction/06-csi-bert2/results/runs/wigesture_no_gan/)
          - [wigesture_no_time/](research/reproduction/06-csi-bert2/results/runs/wigesture_no_time/)
          - [wigesture_paper_v2/](research/reproduction/06-csi-bert2/results/runs/wigesture_paper_v2/)
          - [wigesture_rnn/](research/reproduction/06-csi-bert2/results/runs/wigesture_rnn/)
          - [wigesture_scratch/](research/reproduction/06-csi-bert2/results/runs/wigesture_scratch/)

</details>

## 已完成内容速览

### 论文笔记

| 论文 | 内容 |
|---|---|
| [CLAR](research/paper-notes/01-CLAR.md) | CSI 数据增强与活动识别 |
| [Latent Diffusion Models](research/paper-notes/02-latent%20diffusion%20model.md) | 潜在空间扩散 |
| [LatentCSI](research/paper-notes/03-LatentCSI.md) | CSI 到图像生成 |
| [WiFi-Diffusion](research/paper-notes/04-wifi%20diffusion.md) | 无线电地图估计 |
| [MeanFlow](research/paper-notes/05-meanflow.md) | 平均速度与一步生成 |
| [Echo-Conditioned DDPM](research/paper-notes/06-Echo-Conditioned%20Denoising%20Diffusion.md) | RF 多目标跟踪 |
| [mmWaveFlow](research/paper-notes/07-mmWaveFlow.md) | 毫米波点云增强与生成 |
| [cDDIM](research/paper-notes/08-cDDIM.md) | 位置条件无线信道生成 |
| [ELF](research/paper-notes/09-ELF.md) | 连续语言嵌入生成 |
| [CSI-BERT2](research/paper-notes/10-CSI-BERT2.md) | CSI 恢复、预测与分类 |
| [SRLoc](research/paper-notes/11-SRLoc.md) | 对抗 RSSI 修复与定位防御 |

[论文出处与原文](research/papers/)

### 代码复现

| 项目 | 当前结果 |
|---|---|
| [DDPM · CIFAR-10](research/reproduction/01-ddpm-cifar10/) | 120 Epoch AMP 训练、生成图与去噪过程 |
| [DDIM · CIFAR-10](research/reproduction/02-ddim-cifar10/) | 复用 DDPM 权重，50 步采样 |
| [MeanFlow / iMF · MNIST](research/reproduction/03-meanflow-mnist/) | MF、iMF、CFG 各 20,000 步；Loss、FID 与生成图 |
| [cDDIM · 无线信道](research/reproduction/04-cddim-wireless-channel/) | ID / OOD 与 One-shot U-Net 对照 |
| [Widar3.0](research/reproduction/05-widar3-csi-dfs-bvp/) | 单样本 CSI → DFS → BVP |
| [CSI-BERT2](research/reproduction/06-csi-bert2/) | 三个数据集的核心实验、基线与消融；未完整达到论文指标 |

## 提交记录

[查看全部提交](https://github.com/laportaeaston923-afk/grad-prep-zhoujingjing/commits/main/)
