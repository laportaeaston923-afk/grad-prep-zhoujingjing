# 论文名称

**ELF: Embedded Language Flows**

- **作者：** Keya Hu、Linlu Qiu、Yiyang Lu、Hanhong Zhao、Tianhong Li、Yoon Kim、Jacob Andreas、Kaiming He（前两位为共同第一作者）
- **出处：** arXiv 技术报告，2026；v1：2026-05-11，v2：2026-06-26（补充蒸馏实验）
- **论文：** [arXiv](https://arxiv.org/abs/2605.10938) · [PDF](https://arxiv.org/pdf/2605.10938)
- **官方代码：** [lillian039/ELF](https://github.com/lillian039/ELF)
- **方法关键词：** 连续扩散语言模型、Flow Matching、DiT、干净嵌入预测、共享权重解码、CFG

## 1. 核心内容

ELF 研究如何在连续向量空间里生成文字。它先把训练文本编码为连续语言嵌入，让网络学习从高斯噪声逐步生成这些嵌入，最后才转换成离散 Token。

**整体流程：** 训练文本 → T5 编码器 → 干净语言向量 → Flow Matching 训练；生成时：随机噪声 → 多轮连续更新 → 共享网络切换到解码模式 → 输出 Token。

## 2. 已有方法有什么问题？

语言最终是离散 Token，而图像扩散和流模型主要处理连续数据。已有连续语言生成方法经常在中间步骤加入 Token 级监督，或依赖独立解码器。作者关注的是：能否让大部分生成过程留在连续空间，只在最后处理离散化，同时获得较好的文本质量？

**这不是第一篇把 Flow Matching 用于语言的工作。** ELF 值得关注的是连续去噪与最终离散解码的配合方式，而不是笼统地说“首次使用流匹配生成文字”。

## 3. 核心思想与训练方式

| 设计 | 作用 |
|---|---|
| 上下文嵌入 | 默认使用预训练 T5 编码器构造连续语言表示 |
| 干净嵌入预测 | 网络输出干净向量，再转换为采样需要的速度 |
| 共享权重 | 同一主干分别执行 denoise 和 decode 两种模式 |
| 最后才离散化 | 中间步骤在连续空间更新，最终通过 Unembedding 输出 Token |

训练插值和速度关系为：

$$
z_t=t x+(1-t)\epsilon,\qquad v=x-\epsilon,\qquad
v_\theta=\frac{\hat{x}_\theta-z_t}{1-t}.
$$

去噪训练比较预测速度与目标速度的 MSE；等价地，它是带时间权重的干净嵌入误差，不能直接理解成不加权的普通向量 MSE。解码训练在 $t=1$ 使用专门的 Token 级破坏构造输入，经共享网络和 Unembedding 后计算交叉熵。生成时在 $t<1$ 更新向量，最后调用解码模式，避免在分母为零处计算上述速度式。

**我的理解：** DiT 是进行预测的神经网络；Flow Matching 定义路径和学习目标，求解器执行更新。它们不是两个并列的预测网络。

## 4. 图表阅读与基础概念

DLM：扩散语言模型

计算机里的文字两种形式：离散 Token；连续向量

训练时，给模型大量现成的英文，让它学习；生成时，ELF 可以从随机数字向量开始，经过反复去噪，最终产生新的英文句子。

掩码式离散 DLM： 初始输入可以全是 \[MASK\]，模型逐步预测、恢复离散 Token。

ELF： 初始输入是随机数字向量，模型逐步调整这些向量，最后才转换成 Token。

Flow Matching（流匹配）理解成一种“教模型如何从起点走到终点”的生成方法。


### Fig1:ELF用更少的生成步骤获得更好的文本生成质量

<img width="600" alt="笔记配图" src="https://raw.githubusercontent.com/laportaeaston923-afk/grad-prep-zhoujingjing/49b47038efaecef76844ea666f59e390ff363ac1/research/paper-notes/assets/09-ELF/media/image3.png" />

蒸馏：让一个大模型把自己学到的知识教给一个小模型，让小模型尽可能达到大模型的效果，同时运行得更快、消耗更少的计算资源。蒸馏并不一定让模型效果变差，但它的主要价值通常是用更低的计算成本，尽可能保留原模型的效果。

> 纵坐标 Gen. PPL：生成困惑度，越低通常表示生成文本越符合评估语言模型的统计规律。
>
> Sampling Steps：从噪声生成最终文本需要多少步。
>
> w/o distill：没有使用蒸馏。
>
> w/ distill：使用了蒸馏来加快生成。

结果： ELF 在不使用蒸馏的情况下，只用 32 步便取得较低的生成困惑度，而且训练 Token 数量比其他模型少约一个数量级。

### Fig2：如何把噪声变成文字

<img width="600" alt="笔记配图" src="https://raw.githubusercontent.com/laportaeaston923-afk/grad-prep-zhoujingjing/49b47038efaecef76844ea666f59e390ff363ac1/research/paper-notes/assets/09-ELF/media/image4.png" />

<img width="600" alt="笔记配图" src="https://raw.githubusercontent.com/laportaeaston923-afk/grad-prep-zhoujingjing/49b47038efaecef76844ea666f59e390ff363ac1/research/paper-notes/assets/09-ELF/media/image5.png" />

不需要每生成一步就判断当前应该是哪一个词，而是先在连续向量空间里完成生成，再一次性转换成文字。

图里的橙色点表示连续嵌入空间中的数据，紫色曲线表示从噪声走向真实数据的生成轨迹
<img width="448" height="131" alt="image" src="https://github.com/user-attachments/assets/bcf4552b-6508-49a9-b766-d1f5a14be0f4" />

<img width="319" height="141" alt="image" src="https://github.com/user-attachments/assets/451d2dc1-8bab-45e0-a16e-350221e5f147" />

### Training（训练阶段）

把真实文字变成 Token，然后通过编码器转换成连续向量x，再添加噪声得到 z_t。

去噪模式和解码模式

去噪模式（Denoise）： 模型预测干净嵌入 $\hat{x}$，使用 MSE 损失训练，让模型学会从噪声中恢复连续向量。

解码模式（Decode）： 模型根据受扰动的嵌入预测真实 Token，使用交叉熵（CE）损失训练。

### 下半部分：Sampling（生成阶段）

从随机高斯噪声 z_0开始，模型不断预测干净嵌入，并转换成速度来更新：


在最后t=1时，网络切换到 Decode 模式，通过 unembedding 层输出离散 Token。

这张图要证明： 一个共享权重的模型，就能完成连续空间去噪和最终 Token 解码，不需要另外训练一个独立解码器。

组会上，Fig. 3 最值得详细讲，因为它解释了整个 ELF 模型究竟如何训练和生成。

### Transformer 与位置编码

作用：告诉模型每个 Token 在句子中的位置。

Transformer 可以用于编码或解码等任务；ELF 的 denoise / decode 是共享主干的两种工作模式，不是各自配一套独立的 Transformer Encoder / Decoder。

<img width="600" alt="笔记配图" src="https://raw.githubusercontent.com/laportaeaston923-afk/grad-prep-zhoujingjing/49b47038efaecef76844ea666f59e390ff363ac1/research/paper-notes/assets/09-ELF/media/image201.png" />

Flow Matching 的作用，就是根据每个阶段的预测计算速度，通过数值求解器逐步更新状态，而不是把某一次预测直接当成最终结果。

随机噪声 → DiT 预测干净向量 → Flow Matching 计算速度 → 更新向量 → 重复多轮 → 最后转换成 Token。

DiT 网络预测干净向量；Flow Matching 为连续生成提供训练目标和采样关系。

<img width="600" alt="笔记配图" src="https://raw.githubusercontent.com/laportaeaston923-afk/grad-prep-zhoujingjing/49b47038efaecef76844ea666f59e390ff363ac1/research/paper-notes/assets/09-ELF/media/image202.png" />

CFG 引导强度对生成效果的影响

这是一张折线图：

> 横坐标 Entropy（熵）： 衡量生成文本的多样性，一般越大表示用词越丰富。
>
> 纵坐标 Gen. PPL： 生成困惑度，通常越低越好。
>
> CFG=0.5、1、1.5……3： 表示不同的引导强度。

作者发现，随着 CFG 增大，生成困惑度下降，但文本多样性也降低。

通俗理解：引导越强，模型生成的文字越符合希望的分布，但可能变得比较保守，重复常见的表达方式。

这张图要证明： CFG 在 ELF 中有效，但需要在生成质量与多样性之间权衡。

<img width="600" alt="笔记配图" src="https://raw.githubusercontent.com/laportaeaston923-afk/grad-prep-zhoujingjing/49b47038efaecef76844ea666f59e390ff363ac1/research/paper-notes/assets/09-ELF/media/image203.png" />

### Fig. 5(a)：用什么方式把文字变成连续向量？

作者比较五种嵌入方式，包括预训练编码器、从头训练编码器、预训练词嵌入、随机高斯嵌入和可学习词嵌入。

结果： 使用预训练 T5 编码器生成的上下文嵌入，效果最好。

简单说，既要把文字变成数字，也要尽可能保留上下文语义。

## 5. 实验结果说明了什么？

| 实验 | 观察内容 | 阅读结论 |
|---|---|---|
| OpenWebText 文本生成（Fig. 1、7） | Gen. PPL、熵、采样步数与训练预算 | 原文对照设置中，ELF 无蒸馏也能用较少步数取得较好质量 |
| CFG 与消融（Fig. 4、5） | 引导、嵌入、共享解码与采样器 | 引导增强需要与多样性权衡，组件设计应结合消融理解 |
| WMT14 德英翻译（Table 1） | BLEU | ELF 为 26.4，表中 AR 基线为 25.2 |
| XSum 摘要（Table 1） | ROUGE-1 / 2 / L | ELF 为 36.0 / 12.2 / 27.8 |
| v2 附录 B | Progressive Distillation | 进一步研究少步生成，须与正文无蒸馏结果区分 |

这些结果属于论文报告，不能称为个人复现结果。不同系统的编码器、参数量和训练预算并非完全一致；“步数更少”也不直接等于相同比例的实际加速。

## 6. 总结与我的理解

ELF 的重点是把“连续空间中的生成”和“最后一步的离散解码”衔接起来。干净向量预测方便与解码任务共享主干，Flow Matching 则让预测能够用于连续采样。

值得进一步思考的是：较低的生成困惑度不等于事实更准确；熵不能完整代表语义多样性；多步生成仍有计算成本。讨论模型优劣时，还应看人工质量、实际耗时与任务条件。

## 7. 复现状态与后续问题

当前记录是论文阅读，没有个人训练结果。复现时可先使用官方实现，固定版本、采样步数与 CFG，记录 Gen. PPL、熵和耗时，再比较共享解码与不同嵌入。官方 JAX 实现与 PyTorch 分支应按各自说明使用。

## 8. 一句话总结

**ELF 从随机噪声出发，在连续语言嵌入空间中通过 Flow Matching 逐步生成向量，最后用与去噪共享权重的网络把向量转换成文字。**

[返回笔记目录](README.md) · [返回科研主页](../README.md)
