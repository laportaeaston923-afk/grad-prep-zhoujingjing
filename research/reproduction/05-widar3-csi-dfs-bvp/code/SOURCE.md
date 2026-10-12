# 代码来源与修改范围

官方项目：<https://tns.thss.tsinghua.edu.cn/widar3.0/>。

代码下载：官方项目指向的清华云盘内 `BVPExtractionCode.zip`。原始压缩包SHA256：

```text
d95d431a4e6c76bf549be4ec6cefb486d2e9c037ab2f347cd8e7d37713f16f45
```

| 文件 / 目录 | 来源 | 本次处理 |
|---|---|---|
| `get_doppler_spectrum.m` | Widar3.0官方包 | 原样保留 |
| `get_A_matrix.m`、`get_rotated_spectrum.m`、`get_CastM_matrix.m` | 官方包 | 原样保留 |
| `Doppler2VelocityMapping/` | 官方包 | 原样保留EMD/L0目标、映射与原始求解脚本 |
| `generate_vs.m`、`mesh_bvp.m` | 官方包 | 保留作原始入口/绘图参考 |
| `csi_tool_box/` | 官方包附带CSI工具，含Daniel Halperin等开发者代码 | 保留实际STFT流程需要的文件、原版权头和旧MEX |
| `tftb/` | 第三方Time-Frequency Toolbox，官方包附带 | 保留实际调用函数、AUTHORS、README、COPYING；原许可为GPL |
| `BVP/` | 官方包的候选参考 | 原样保留，不作为已确认样本真值 |
| `setup_reproduction.m` | 本次新增 | 路径与依赖检查 |
| `step1_csi_to_dfs.m` | 本次新增 | 数据检查、调用作者DFS、排序频率、保存并绘图 |
| `step2_dfs_to_bvp.m` | 本次新增 | 沿用作者参数/目标，顺序求解、保存日志和BVP |
| `compare_bvp.m` | 本次新增 | 候选参考并排图与逐段指标 |
| `verify_and_plot.py` | 本次新增 | 对上传数组进行独立核算与重绘 |
| 学习笔记、问答、HTML | 根据本次实际讨论整理 | 修正数据包/时间/快照等容易误解的比喻 |

仓库版相比本次用户运行的新增入口，只将 `outdir` 从 `code/results/` 改为同级 `results/`；保持原行数。核心作者算法文件没有改动，逐文件原始SHA256见 `SOURCE_HASHES.json`。

官方源包的README把入口写成 `generate_vs.mat`，实际文件是 `generate_vs.m`。本项目使用新增分步入口，避免自动开启并行池，并输出可核验的日志。

保留所有源文件中的作者署名及第三方许可。未给官方Widar算法文件另行补充或宣称一个新的统一开源许可证。引用论文：Yue Zheng, Yi Zhang, Kun Qian, Guidong Zhang, Yunhao Liu, Chenshu Wu, Zheng Yang, *Zero-Effort Cross-Domain Gesture Recognition with Wi-Fi*, ACM MobiSys 2019，DOI: 10.1145/3307334.3326081。
