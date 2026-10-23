# ELF: Embedded Language Flows

连续语言嵌入空间中的 Flow Matching、共享权重去噪与最终 Token 解码。

> 根据上传的 Word 阅读笔记整理，保留原有讲解、配图、表格和组会汇报内容。

[返回笔记目录](README.md)

## 阅读记录

Dlm：扩散语言模型

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

去噪模式（Denoise）： 模型预测干净嵌入 \\\hat{x}\\，使用 MSE 损失训练，让模型学会从噪声中恢复连续向量。

解码模式（Decode）： 模型根据受扰动的嵌入预测真实 Token，使用交叉熵（CE）损失训练。

### 下半部分：Sampling（生成阶段）

从随机高斯噪声 z_0开始，模型不断预测干净嵌入，并转换成速度来更新：


在最后t=1时，网络切换到 Decode 模式，通过 unembedding 层输出离散 Token。

这张图要证明： 一个共享权重的模型，就能完成连续空间去噪和最终 Token 解码，不需要另外训练一个独立解码器。

组会上，Fig. 3 最值得详细讲，因为它解释了整个 ELF 模型究竟如何训练和生成。

Transformer

### Positional Encoding：位置编码

作用：告诉模型每个 Token 在句子中的位置。

Encoder（编码器）和 Decoder（解码器）

<img width="600" alt="笔记配图" src="https://raw.githubusercontent.com/laportaeaston923-afk/grad-prep-zhoujingjing/49b47038efaecef76844ea666f59e390ff363ac1/research/paper-notes/assets/09-ELF/media/image201.png" />

Flow Matching 的作用，就是根据每个阶段的预测计算速度，通过数值求解器逐步更新状态，而不是把某一次预测直接当成最终结果。

随机噪声 → DiT 预测干净向量 → Flow Matching 计算速度 → 更新向量 → 重复多轮 → 最后转换成 Token。

Dit神经网络，负责预测干净的向量

flow matching让模型学到预测用于一套完整连续生成过程。

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




[返回笔记目录](README.md) · [返回科研主页](../README.md)
