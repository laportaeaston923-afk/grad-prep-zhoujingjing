# 论文名称
WiFi-Diffusion: Achieving Fine-Grained WiFiRadio Map Estimation with Ultra-Low SamplingRate by Diffusion Models
## 1. 核心内容
在只采集极少量 WiFi 无线电样本的情况下，如何生成整个区域的高分辨率、细粒度 Radio Map。
```text
少量 Radio Samples
       ↓
WiFi-Diffusion
       ↓
完整的 Fine-Grained Radio Map
```
Radio Map 可以理解成：
一个区域中每一个位置的无线信号参数，例如 RSS 的分布地图。
## 2. 已有方法有什么问题？
已有的无线电地图估计方法主要包括`插值方法`和`深度学习方法`。
插值方法根据少量已知 RSS 点，通过数学和空间关系估计未知位置的 RSS，
- 缺点：但是这类方法通常没有充分利用建筑物布局等影响无线信号传播的先验信息，因此在复杂环境下容易产生较大的误差。已有的深度学习方法虽然能够学习复杂的无线传播关系，
- 缺点：当采样率降低到非常低时，性能会明显下降。


## 3. 核心思想
```text
少量 Radio Samples
        ↓
   Boost Block
        ↓
   Generation Block
        ↓
64个Radio Map Candidates
        ↓
   Election Block
        ↓
最终 Radio Map

增强 → 生成 → 选优
```

## 4. 整体框架

<img width="407" height="197" alt="image" src="https://github.com/user-attachments/assets/ae9842c8-24ff-45d1-8aff-aa30b0c93729" />

**`Boost Block`**
利用一些 Radio Map 的先验信息：
- 障碍物布局；
- 已经收集到的 Radio Samples
  先生成一个radio map 得到关键点和随机的采样点加上原先的采样点
**`Generation Block`**
  它使用条件 DDPM 学习无线电地图的分布，在推理阶段使用 DDIM 加速采样，并生成64个候选的 Radio Map。
**`Election Block`**
  从64个候选 Radio Map 中选出最好的一个
真实的 Radio Map 在实际推理时是`未知`的，所以不能直接通过与真实地图比较来选择结果。
因此，该模块利用无线信号传播相关的物理先验，从多个候选 Radio Map 中选择最合理的一个，最终得到高分辨率的无线电地图。
## 5. 总结
利用 Diffusion 的生成能力，在极少的无线电采样点下生成多个可能的细粒度 Radio Map，再结合无线传播规律从中选出最合理的结果。
