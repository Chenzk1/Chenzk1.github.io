---
title: MTL多任务学习
mathjax: true
date: 2022-01-18 10:28:35
tags:
  - ML
  - MTL
categories:
  - Learning
---

> MTL更多的是一种思想

# 两种模式
- hard share: 不同任务直接共享部分模型参数
- soft share: 不共享参数，添加正则来保证参数的相似

<!-- more -->
# 原理
- 隐式数据增加（implicit data augmentation）：为了训练任务A，通过其他任务的数据来扩充任务A训练过程的数据量。这些数据可以看做是引入额外的噪声，在理想情况下起到提高模型泛化效果的作用
- 注意力机制（Attention focusing）：当训练数据量有限且高维时，模型很难区分出相关的特征和不相关的特征。多任务学习可以使模型更关注于有用的特征
- 监听机制（Eavesdropping）：从任务B中有可能容易学习到特征G，但是从任务A中很难学习得到。通过多任务学习，可以通过任务B学习到特征G，再利用特征G预测任务A
特征偏置（Representation bias）：多任务学习使得模型更容易去选择其他任务容易选择到的特征，这样有助于模型在假设空间下获得更好的模型泛化能力。

# 常见模型结构
- 常见模型：share bottom/MMoE/SNR/ESMM；损失函数按照指定权重融合，或者使用GradNorm做动态融合

## Share bottom
- 最常见的mtl结构，不同任务共享底层

## MMoE(Multi-gate Mixture-of-Experts)
- [paper](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)
- 是一种soft share
- 添加了expert机制获取不同信息
- 添加gate机制对expert进行加权
{% asset_img mmoe.png mmoe %}

## CGC(Customized Gate Control)/PLE(Progressive Layered Extraction)
- MMoE的基础上，把experts分为share experts和domain experts
- 单层多任务网络结构(CGC)，多层多任务网络结构（PLE）
{% asset_img cgc.png cgc %}
{% asset_img ple.png ple %}

## SNR(Sub-Network Routing)
- [paper](https://ojs.aaai.org//index.php/AAAI/article/view/3788)
{% asset_img snr.png snr %}
- SNR-Trans: 下层输出通过权重矩阵变换，再做加权后输出至下一层。会引入更多参数。其中z为二进制。
$$
\left[\begin{array}{l}
\boldsymbol{v}_{1} \\
\boldsymbol{v}_{2}
\end{array}\right]=\left[\begin{array}{lll}
z_{11} W_{11} & z_{12} W_{12} & z_{13} W_{13} \\
z_{21} W_{21} & z_{22} W_{22} & z_{23} W_{23}
\end{array}\right]\left[\begin{array}{l}
u_{1} \\
u_{2} \\
u_{3}
\end{array}\right]
$$
- SNR-Avg: 下层输出加权后输出至下一层
$$
\left[\begin{array}{l}
\boldsymbol{v}_{1} \\
\boldsymbol{v}_{2}
\end{array}\right]=\left[\begin{array}{lll}
z_{11} \boldsymbol{I}_{11} & z_{12} \boldsymbol{I}_{12} & z_{13} \boldsymbol{I}_{13} \\
z_{21} \boldsymbol{I}_{21} & z_{22} \boldsymbol{I}_{22} & z_{23} \boldsymbol{I}_{23}
\end{array}\right]\left[\begin{array}{l}
\boldsymbol{u}_{1} \\
\boldsymbol{u}_{2} \\
\boldsymbol{u}_{3}
\end{array}\right]
$$
- 损失函数：
$$
\min _{\boldsymbol{W}, \boldsymbol{\pi}} \boldsymbol{E}_{\boldsymbol{z} \sim p(\boldsymbol{z} ; \boldsymbol{\pi})} \frac{1}{N} \sum_{i=1}^{N} L\left(f\left(\boldsymbol{x}_{i} ; \boldsymbol{W}, \boldsymbol{z}\right), \boldsymbol{y}_{i}\right)
$$
- z为二进制，无法优化，需要把z松弛为平滑变量，将z变为hardSigmoid，其中s是一个服从q分布的连续的随机变量
$$
z=g(s)=\min (1, \max (0, s))
$$
- 可以继续转换，其中epsilon是一个噪声变量，r(epsilon)是一个无参数的噪声分布，h是一个确定且可微的分布
$$
\min _{\boldsymbol{W}, \boldsymbol{\pi}} \boldsymbol{E}_{\boldsymbol{\epsilon} \sim r(\boldsymbol{\epsilon})} \frac{1}{N} \sum_{i=1}^{N} L\left(f\left(\boldsymbol{x}_{i} ; \boldsymbol{W}, g(h(\boldsymbol{\phi}, \boldsymbol{\epsilon}))\right), \boldsymbol{y}_{i}\right)
$$
- 在实际应用中，结合重采样技术和hard concrete distribution，可以继续做转换，其中u为均匀分布，log(α)为需要学习的参数，其他为超参
$$
\begin{aligned}
u \sim U(0,1), s &=\operatorname{sigmoid}((\log (u)-\log (1-u)+\log (\alpha) / \beta)\\
\bar{s} &=s(\zeta-\gamma)+\gamma, z=\min (1, \max (\bar{s}, 0))
\end{aligned}
$$
- training的时候，加入z的L0正则，则最终loss为：
$$
\begin{array}{r}
\boldsymbol{E}_{\boldsymbol{\epsilon} \sim r(\boldsymbol{\epsilon})} \frac{1}{N} \sum_{i=1}^{N} L\left(f\left(\boldsymbol{x}_{i} ; \boldsymbol{W}, g(h(\boldsymbol{\phi}, \boldsymbol{\epsilon}))\right), \boldsymbol{y}_{i}\right) \\
+\lambda \sum_{j=1}^{|\boldsymbol{z}|} 1-Q\left(s_{j}<0 ; \phi_{j}\right)
\end{array}
$$
- training: 总结来说，模型需要学习的参数是W和隐变量分布变量log(alpha)。训练流程如下所示：
  - 首先，采样一组均匀分布的随机变量u；
  - 其次，计算z来获得网络结构；
  - 最后，将训练数据喂给模型来计算损失函数。W和log(alpha)的梯度通过反馈计算得到。
- serving: 使用如下的estimator来得到z的值
$$
\hat{\boldsymbol{z}}=\min (1, \max (0, \operatorname{sigmoid}(\log (\boldsymbol{\alpha}))(\zeta-\gamma)+\gamma)) \text {. }
$$

When sigmoid $$\left(\log \left(\alpha_{i j}\right)\right)(\zeta-\gamma)+\gamma<0$$, we will have $$\hat{z}_{i j}=0$$ and the resulted model will be sparsely connected.

## Star(One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction)
- [paper](https://arxiv.org/abs/2101.11427)
{% asset_img star.png star %}
- center tower和domain tower的结合，论文中是两塔权重element-wise相乘，也可以改为两塔logits相加
- PN: partitioned normalization, 在BN的基础上，对每个domain引入domain相关的两个scale参数
$$
z^{\prime}=\left(\gamma * \gamma_{p}\right) \frac{z-\mu}{\sqrt{\sigma^{2}+\epsilon}}+\left(\beta+\beta_{p}\right)
$$

## 十字绣(Cross-Stitch Network)
- [paper](https://arxiv.org/pdf/1604.03539.pdf)

## ESMM(Entire Space Multi-Task Model)
- 混合ctr、cvr数据流
- 传统CVR预估模型的本质，不是预测“item被点击，然后被转化”的概率（CTCVR），而是“假设item被点击，那么它被转化”的概率（CVR）。即CVR模型的样本空间，是click空间。
- ESMM
  - 使用了全样本空间，通过预测CTR和CTCVR间接求CVR
  - 解决training样本有偏的问题: training的时候在点击空间，serving的时候在展示空间。
  - 直接在show空间求ctcvr，label太稀疏，建模ctcvr=cvr*ctr，解决稀疏问题
- 利用全概率公式，**隐式学习pCVR**
- [paper](https://arxiv.org/pdf/1804.07931.pdf)
- [blog](https://zhuanlan.zhihu.com/p/57481330)
$$
\underbrace{p(y=1, z=1 \mid x)}_{p C T C V R}=\underbrace{p(y=1 \mid x)}_{p C T R} \times \underbrace{p(z=1 \mid y=1, x)}_{p C V R}
$$
- pCVR只是一个variable，无显式监督信号
{% asset_img esmm.png esmm %}

# 其他
## GradNorm
- [paper](https://openreview.net/pdf?id=H1bM1fZCW)
- 分两个loss，加权多任务loss以及每个任务权重的loss，后者只过share bottom
- 两个作用
  - 动态调整不同任务损失函数的权重：不同目标重要性不同/收敛的程度/loss的大小 diff较大，可以考虑使用
  - share bottom的更新受到每个任务权重的影响
- https://blog.csdn.net/Leon_winter/article/details/105014677

## share vs not share
- not share bias emb: tend to useful
- share sparse: tend to useful
- share bottom use small learning rate, tower use adam or bigger learning rate

# Refs
- [一篇综述](https://blog.nowcoder.net/n/8a9d69d063c546b291a3c9a5091cfbbe)
