# 官方示例CSI数据

这里放原始 `.dat` 输入，仓库不重复提交原始CSI数据。

从 [Widar3.0官方数据与代码目录](https://cloud.tsinghua.edu.cn/d/2760bb9557ca4d09a74d/) 下载 `BVPExtractionCode.zip`，解压后把 `Widar3.0Release-Matlab/Data/` 内的6个文件复制到本目录：

```text
userA-1-1-1-1-r1.dat
userA-1-1-1-1-r2.dat
userA-1-1-1-1-r3.dat
userA-1-1-1-1-r4.dat
userA-1-1-1-1-r5.dat
userA-1-1-1-1-r6.dat
```

如果已经用本次提供的复现包跑通，可直接复制电脑上已有的 `E:\all\Widar3.0Release-Matlab\Data\`，不用再下载。

样本的包数依次为1479、1435、1538、1507、1515、1523，每包为1根发射天线、3根接收天线、30条子载波。作者DFS函数以r1的长度为基准，对其他路截断或补零，并未按真实时间戳做多接收机同步。

换其他样本时要核对几何、朝向、载频、采样率和文件命名；不要仅修改文件名就沿用所有实验参数。
