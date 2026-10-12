# 代码与学习资料

## 先读什么

- `NOTES.md`：14步图解、日常解释、数学公式、文件名和准确行号，包含85个术语说明。
- `QUESTIONS.md`：本次学习中提出的24组问题，重点解释CSI与数据包、RSSI校正、参考天线、窗口与时间、频谱快照、100快照平均、几何关系、反推和人体坐标。
- `learning_notes.html`：互动学习笔记，下载后用浏览器打开；GitHub源文件页不会直接执行它。
- `SOURCE.md`：作者原代码、附带第三方工具与新增代码的区别。

## MATLAB 运行

1. 按 `Data/README.md` 准备6份官方示例 `.dat`。
2. 在MATLAB将本目录设为当前文件夹。
3. 先运行第一步，确认6路DFS图正常，再运行第二步与对比。

```matlab
step1_csi_to_dfs
step2_dfs_to_bvp
compare_bvp
```

结果在项目同级 `results/userA-1-1-1-1/`。仓库版只改了新增入口的输出路径，原核心算法保持原样；笔记对应的行号也保留。

| 工具箱 | 用到的函数 |
|---|---|
| Signal Processing Toolbox | `butter`、CSI工具里的 `db` |
| Statistics and Machine Learning Toolbox | `pca` |
| Optimization Toolbox | `fmincon`、`optimoptions` |
| Image Processing Toolbox | `imrotate` |

可用 `ver` 检查已安装工具箱。本次用户在Windows MATLAB成功运行，但未提供具体MATLAB版本。新版MEX兼容性不能一概保证。

作者附带的Windows 64位、Linux 64位与Intel macOS旧MEX一并保留。若 `read_bfee` 报 Invalid MEX-file，先配置MATLAB支持的C编译器，再在本目录执行：

```matlab
mex -setup C
mex('-outdir',fullfile(pwd,'csi_tool_box'), ...
    fullfile(pwd,'csi_tool_box','read_bfee.c'))
```

本整理仅保留实际使用的 **STFT路径** 所需TFTB函数；未整理CWT可选路径的额外依赖。原始 `generate_vs.m` / `DVM_main.m` 作为源码参考保留，默认运行入口为上面三个新增脚本。

## 保存结果的核验与重绘

`verify_and_plot.py` 从已保存的 `DFS.mat`、`BVP.mat` 和包内候选参考核算官方EMD/L0目标，并重绘DFS、BVP和对比图。它不重新采集CSI、不运行MATLAB、不重新优化BVP。

```bash
python -m pip install -r requirements-verification.txt
python verify_and_plot.py --dfs ../results/userA-1-1-1-1/DFS.mat
```

仓库保存27KB的BVP结果，7.8MB的原始 `DFS.mat` 留在本机或由第一步重新生成。可把本机DFS复制到上述路径，或用 `--dfs` 指定其完整路径。核验脚本限定本示例的−90°朝向，验证原/旋转BVP的对应关系、数组尺寸、频率顺序与保存目标值。

核验使用Python 3环境：NumPy 2.3.5、SciPy 1.17.0、Matplotlib 3.10.8。本次核验通过；源代码中没有把候选参考当成人体速度真值。
