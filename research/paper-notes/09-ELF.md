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

<img width="600" alt="笔记配图" src="https://raw.githubusercontent.com/laportaeaston923-afk/grad-prep-zhoujingjing/49b47038efaecef76844ea666f59e390ff363ac1/research/paper-notes/assets/09-ELF/media/image1.png" />

<img width="600" alt="笔记配图" src="https://raw.githubusercontent.com/laportaeaston923-afk/grad-prep-zhoujingjing/49b47038efaecef76844ea666f59e390ff363ac1/research/paper-notes/assets/09-ELF/media/image2.png" />

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

### Fig3：

<img width="600" alt="笔记配图" src="https://raw.githubusercontent.com/laportaeaston923-afk/grad-prep-zhoujingjing/49b47038efaecef76844ea666f59e390ff363ac1/research/paper-notes/assets/09-ELF/media/image6.png" />

### Training（训练阶段）

把真实文字变成 Token，然后通过编码器转换成连续向量x，再添加噪声得到 z_t。

去噪模式和解码模式

去噪模式（Denoise）： 模型预测干净嵌入 \\\hat{x}\\，使用 MSE 损失训练，让模型学会从噪声中恢复连续向量。

解码模式（Decode）： 模型根据受扰动的嵌入预测真实 Token，使用交叉熵（CE）损失训练。

### 下半部分：Sampling（生成阶段）

从随机高斯噪声 z_0开始，模型不断预测干净嵌入，并转换成速度来更新：

<img width="600" alt="笔记配图" src="https://raw.githubusercontent.com/laportaeaston923-afk/grad-prep-zhoujingjing/49b47038efaecef76844ea666f59e390ff363ac1/research/paper-notes/assets/09-ELF/media/image7.png" />

在最后t=1时，网络切换到 Decode 模式，通过 unembedding 层输出离散 Token。

这张图要证明： 一个共享权重的模型，就能完成连续空间去噪和最终 Token 解码，不需要另外训练一个独立解码器。

组会上，Fig. 3 最值得详细讲，因为它解释了整个 ELF 模型究竟如何训练和生成。

Transformer

### Positional Encoding：位置编码

作用：告诉模型每个 Token 在句子中的位置。

<img width="600" alt="笔记配图" src="https://raw.githubusercontent.com/laportaeaston923-afk/grad-prep-zhoujingjing/49b47038efaecef76844ea666f59e390ff363ac1/research/paper-notes/assets/09-ELF/media/image10.png" />

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

# ELF：嵌入式语言流模型组会汇报

## 一、研究背景与论文要解决的问题

老师、同学们好，今天我汇报的论文是《ELF: Embedded Language Flows》，中文可以翻译为《嵌入式语言流》。

这篇论文主要研究的是：**如何利用连续空间中的扩散与流匹配方法生成文字，同时提高语言生成的质量和效率。**

首先介绍一下背景。

计算机中的文字通常以离散 Token 的形式表示。例如，一句话可以被切分成多个 Token，再通过 Embedding 层转换成连续向量，供神经网络处理。

传统的自回归语言模型，例如 GPT，通常按照从左到右的顺序逐个预测下一个 Token。而扩散语言模型，也就是 DLM，则可以通过逐步去噪的方式生成文字。

目前的扩散语言模型主要分为两类。

第一类是离散扩散语言模型，例如 MDLM。它直接在离散 Token 空间中进行生成，典型做法是先将部分 Token 替换成 \[MASK\]，再逐步预测和恢复这些 Token。

第二类是连续扩散语言模型。它先将文字转换成连续向量，再在连续空间中进行生成，最后将向量转换回 Token。

但是，已有连续扩散语言模型的生成效果并不总是理想的。因此，作者希望研究：连续扩散语言模型的性能不足，究竟是因为语言本身具有离散性，还是因为现有方法的设计还不够好？

针对这个问题，作者提出了 ELF。

## 二、ELF 的核心思想与创新点

ELF 全称是 Embedded Language Flows，即嵌入式语言流。它是一种基于连续时间 Flow Matching 的扩散语言模型。

它最核心的思想是：**生成文字时，尽可能让整个过程停留在连续语言向量空间中，只在最后一步才将向量转换成离散 Token。**

这与掩码式离散扩散模型有明显区别。

离散扩散模型可以从一串 \[MASK\] 开始，逐步恢复文字；而 ELF 从随机高斯噪声向量开始，逐步调整这些向量，最终才转换成文字。

我认为这篇论文主要有三个创新点。

第一，作者将连续时间 Flow Matching 引入语言生成，使模型能够在连续语言嵌入空间中完成生成，而不是在每一步都进行 Token 级别的离散预测。

第二，作者采用干净向量预测，也就是让神经网络预测最终的干净语言向量，再根据预测结果计算速度。这种设计既适合连续生成，也便于与最终的 Token 解码任务结合。

第三，作者让去噪任务和最终的 Token 解码任务共享同一个网络的权重，不需要额外训练一个完全独立的解码网络。

这三个设计共同构成了 ELF 的核心方法。

## 三、模型具体是怎么运行的？

这里重点结合 Figure 2、Figure 3 和 Figure 10 介绍。

### 1. 训练阶段

训练时，作者先准备真实文本，将 Token 经过编码器转换成干净的连续语言向量，记作 x。

然后，随机生成高斯噪声，记作 ε，并随机选择一个时间步 t，将真实向量和噪声进行线性混合：

zₜ = t x + (1 − t) ε

当 t = 0 时，输入是纯噪声；当 t = 1 时，输入对应干净的语言向量。

接下来，ELF 的 Transformer 主干网络根据当前带噪声的向量和时间步，预测干净语言向量，记作 x̂。

这里需要特别说明：网络虽然预测干净向量，但训练时并不是让模型自己猜测正确的生成轨迹。因为训练数据中的真实向量和随机噪声都已知，所以可以直接计算正确的目标速度：

v = x − ε

ELF 再根据网络预测的干净向量计算预测速度，并与正确的目标速度进行比较，通过均方误差类损失训练网络。

因此，Flow Matching 在训练中提供连续路径和速度学习的目标，而 Transformer 主干网络学习根据当前状态预测干净向量。

### 2. 为什么已经预测了干净向量，还需要 Flow Matching？

这是我理解这篇论文时重点思考的问题。

我最开始认为，既然网络已经预测了干净向量，就可以直接把它转换成文字，没有必要再使用 Flow Matching。

但后来发现，两者的作用不同。

Transformer 网络负责根据当前状态预测干净向量；Flow Matching 则定义连续生成的学习目标，并将网络的预测转换成速度；数值求解器再根据速度更新当前向量。

在生成阶段，模型一开始只有随机噪声，并不知道最终应该生成什么文字。因此，它需要反复执行预测、计算速度和更新向量的过程。

ELF 中的预测速度可以写成：

vθ = (x̂ − zₜ) / (1 − t)

然后根据速度更新当前状态：

zₜ₊Δₜ ≈ zₜ + Δₜ vθ

经过多轮更新，向量逐渐接近干净的语言表示，最后再进行 Token 解码。

所以，Flow Matching 并不是另一个独立的预测网络，也不能保证预测一定准确。它的作用是提供连续生成的规则，让网络的预测能够用于完整的生成过程。

### 3. 最后一步的解码是怎么实现的？

Figure 3 还展示了一个重要设计：去噪和解码使用同一个共享权重的 ELF 网络。

训练时有两条路线。

第一条是去噪路线。网络预测干净向量，使用均方误差类损失训练，让模型学习连续生成过程。

第二条是解码路线。作者让网络在 t = 1 的解码模式下工作，并对 Token 对应的输入表示进行专门的 Token 级破坏，避免模型直接复制干净输入。网络预测向量后，再通过 Unembedding 层映射成词表中各个 Token 的分数，使用交叉熵损失与真实 Token 比较。

这里要注意，解码路线的 Token 级破坏，不等同于普通去噪路线中的高斯噪声混合。

在真正生成时，ELF 不会每一步都把向量转换成文字，而是先在连续空间中完成多步更新，最后才切换到解码模式，通过 Unembedding 输出 Token。

因此，ELF 不需要单独的 Transformer Decoder，而是通过共享权重的网络同时学习去噪和解码。

## 四、实验结果与图表分析

接下来介绍论文的主要实验。

### 1. Figure 1 和 Figure 7：与其他扩散语言模型比较

作者将 ELF 与 MDLM、Duo、FLM、LangFlow 等模型进行比较。

实验结果表明，在 OpenWebText 文本生成任务上，ELF 不经过蒸馏，使用较少的采样步数也能获得较好的生成质量。

论文报告，ELF 使用 32 步采样时，生成困惑度已经优于图中比较的多种先前方法。

在训练数据效率方面，ELF 使用约 45.2B 个有效训练 Token，而部分对比模型使用约 524B–577B 个，训练 Token 预算相差约一个数量级。

这里的 Gen. PPL 指生成困惑度，通常越低越好；Sampling Steps 指生成过程中反复更新向量的步数，步数越少通常意味着推理开销越低。

### 2. Figure 4：CFG 引导强度有什么影响？

CFG 是无分类器引导。

作者发现，增大 CFG 强度通常可以降低生成困惑度，但同时也会降低文本多样性。

这说明引导强度并不是越大越好，而是需要在生成质量和多样性之间进行权衡。

### 3. Figure 5：核心设计是否有效？

Figure 5 是消融实验，主要研究几个设计选择。

第一，比较不同的语言嵌入方式。结果显示，使用预训练 T5 编码器生成的上下文嵌入效果最好，说明语言表示不仅需要转换成数字，还需要尽可能保留上下文语义。

第二，比较去噪器和解码器共享权重与分开训练的方案。实验支持共享权重的设计具有竞争力，同时可以避免额外设置一个独立的解码网络。

第三，比较 ODE 和 SDE 两种采样方式。实验显示，SDE 式采样在较少步数下通常有更好的生成困惑度表现。

### 4. Table 1：条件生成任务的表现

作者还在德语到英语机器翻译和 XSum 文本摘要任务上进行了实验。

机器翻译采用 BLEU 指标，ELF 的得分为 26.4，高于表中自回归基线的 25.2。

在 XSum 摘要任务中，ELF 的 ROUGE-1、ROUGE-2 和 ROUGE-L 分别为 36.0、12.2 和 27.8，均高于表中列出的对比方法。

这些结果说明，ELF 不仅能够进行无条件文本生成，也能够用于机器翻译和文本摘要等条件生成任务。

## 五、论文的不足与我的思考

我认为，这篇论文的优点是设计思路比较清楚：它没有简单地把语言生成限制在离散 Token 空间中，而是充分利用连续嵌入空间和 Flow Matching，并通过共享权重的设计，将去噪和最终解码结合起来。

同时，实验覆盖了无条件文本生成、机器翻译和文本摘要，并对嵌入方式、预测目标、CFG 和采样方法进行了消融分析。

但这篇论文也有一些值得进一步思考的地方。

第一，生成困惑度和熵主要衡量文本的统计规律与多样性，并不能完全代表语义正确性、事实准确性和人工评价下的整体质量。

第二，ELF 虽然能够减少采样步数，但仍然需要迭代更新。Figure 9 的蒸馏实验也表明，如果过度压缩生成步数，生成质量仍然可能下降。

第三，论文的实验主要集中在英文文本生成、德英翻译和英文摘要等设置上，因此还需要进一步研究它在其他语言、任务和更大模型规模下的表现。

## 六、总结

最后总结一下。

这篇论文提出了 ELF，一种基于连续时间 Flow Matching 的扩散语言模型。

它的核心思路是让模型从随机噪声开始，在连续语言嵌入空间中逐步生成干净向量，并在最后一步才将向量转换成离散 Token。

它的主要创新包括干净向量预测、去噪与解码共享权重，以及将连续流匹配方法用于语言生成。

实验结果显示，ELF 在生成质量、采样步数和训练 Token 效率方面具有优势，并且在机器翻译和文本摘要任务上取得了较好的结果。

我认为，这篇论文最值得学习的地方，是它通过重新设计连续生成和离散解码之间的关系，改善了连续扩散语言模型的生成效果，也为后续研究连续语言生成提供了一种思路。

我的汇报到这里结束，谢谢大家。


[返回笔记目录](README.md) · [返回科研主页](../README.md)
