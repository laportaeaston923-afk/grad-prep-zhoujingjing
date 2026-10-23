# Widar3.0 · CSI → DFS → BVP

使用 Widar3.0 官方示例，在 MATLAB 跑通无线信号处理：读取已有 CSI，提取6路 DFS，再估计人体坐标系速度分布 BVP。

[学习笔记](code/NOTES.md) · [疑问整理](code/QUESTIONS.md) · [结果图](results/README.md)

## 结果

| 项目 | 结果 |
|---|---|
| 输入 | 官方代码包的 `userA-1-1-1-1`，6个接收机，每机3根接收天线 |
| 采样假设 | 1000包/秒；未按真实包时间戳重采样 |
| DFS | `6×121×1479`，频率格点−60～60 Hz |
| BVP | `20×20×14`，每100个DFS快照平均后求解 |
| 优化状态 | 14段 `exit_flag` 全部为1；耗时约137.08秒 |
| 候选参考对比 | 平均余弦相似度0.615，归一化重叠度0.483；第13段几乎一致 |
| 对同一DFS的平均EMD拟合项 | 本次27.938；包内候选参考29.105，越低越好 |

已跑通单样本流程，未训练分类器或复现跨域识别。退出标志为1不代表全局最优或真实速度完全正确。

包内参考BVP的前缀是 `user`，示例CSI前缀是 `userA`；其样本对应关系未最终确认，因此上述数值是候选参考比较，不是识别准确率。

## 运行

先把官方包内6个 `.dat` 示例放到 `code/Data/`，然后在 MATLAB 把 `code/` 设置为当前文件夹：

```matlab
step1_csi_to_dfs
step2_dfs_to_bvp
compare_bvp
```

输出写入同级 `results/userA-1-1-1-1/`。需要 Signal Processing、Statistics and Machine Learning、Optimization、Image Processing Toolbox；不用训练权重。详细依赖、数据准备及来源见 `code/README.md`。

核心算法来自 [Widar3.0官方项目](https://tns.thss.tsinghua.edu.cn/widar3.0/)。代码来源见 [SOURCE.md](code/SOURCE.md)。结果来自实际 MATLAB 运行；除 CSI 幅度原图外，其余图由保存数组重绘。

[返回复现总览](../README.md)
