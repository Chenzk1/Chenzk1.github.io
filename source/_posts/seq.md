---
title: 序列模型
mathjax: true
date: 2022-03-09 11:29:58
tags:
  - ML
  - Seq Model
categories:
  - MachineLearning
  - 搜广推
---

广告/推荐中的序列模型
<!-- more -->

# DIN(Deep Interest Network)

[paper](https://dl.acm.org/doi/abs/10.1145/3219819.3219823)

## Model
- 思想：针对不同的候选广告，用户历史行为与该广告的权重是不同的。
- 方法：以当前候选广告为query，用户历史行为为key/value，利用attention机制，求query与key的attention score，并利用attention score加权value
{% asset_img din.png din %}

$$ V_{u}=f\left(V_{a}\right)=\sum_{i=1}^{N} w_{i} * V_{i}=\sum_{i=1}^{N} g\left(V_{i}, V_{a}\right) * V_{i} $$

- Vi表示behavior id i的嵌入向量，比如good_id,shop_id等。Vu是所有behavior ids的加权和，表示的是用户兴趣；Va是候选广告的嵌入向量；wi是候选广告影响着每个behavior id的权重，也就是Local Activation。wi通过Activation Unit计算得出，这一块用函数去拟合，表示为g(Vi,Va)。

## Tricks
### GAUC
- group auc
- auc的评估是对所有样本排序后，评估正样本排在负样本的概率。但推荐、广告系统中，关注的是单个用户的排序。
- gauc是在单个用户auc基础上，按照点击次数、show次数等进行加权平均，消除用户偏差对模型的影响。n为用户数量。
$$ \mathrm{GAUC}=\frac{\sum_{i=1}^{n} w_{i} * \mathrm{AUC}_{i}}{\sum_{i=1}^{n} w_{i}}=\frac{\sum_{i=1}^{n} \text { impression }_{i} * \mathrm{AUC}_{i}}{\sum_{i=1}^{n} \text { impression }_{i}} $$

### DICE
- Data Dependent Activation Function
$$ f(s)=p(s) . s+(1-p(s)) \cdot \alpha s, p(s)=\frac{1}{1+e^{-\frac{s-E(s)}{\sqrt{V} a r(s)+\epsilon}}} $$
- 可以视为Batch Normalization的变化。使得**激活函数随着一个batch的数据分布做自适应调整**——data dependent。
$$ f(s)=\operatorname{sigmoid}(B N(s)) \cdot s+(1-\operatorname{sigmoid}(B N(s))) \cdot \alpha s $$
- pi是一个概率值，这个概率值决定着输出是取 s 或者是 alpha_i * s，起到了一个整流器的作用。pi的计算分为两步：
  - 首先，对 x 进行均值归一化处理，这使得整流点是在数据的均值处，实现了 data dependent 的想法；
  - 其次，经过一个 sigmoid 函数的计算，得到了一个 0 到 1 的概率值。
- 另外，期望和方差使用每次训练的 mini batch data 直接计算，并类似于 Momentum 使用了 指数加权平均：

$$
\begin{aligned}
E\left[s_{i}\right]_{t+1}^{\prime} &=E\left[s_{i}\right]_{t}^{\prime}+\alpha E\left[s_{i}\right]_{t+1} \\
\operatorname{Var}\left[s_{i}\right]_{t+1}^{\prime} &=\operatorname{Var}\left[s_{i}\right]_{t}^{\prime}+\alpha \operatorname{Var}\left[s_{i}\right]_{t+1}
\end{aligned}
$$

### Adaptive L2 Regularization
- 两点改进
  - 针对 feature id 出现的频率，来自适应的调整他们正则化的强度:
    - 对于出现频率高的，给与较小的正则化强度；
    - 对于出现频率低的，给予较大的正则化强度。
  - 正则化涉及的参数限制在了仅在Mini-batch出现过的特征所影响的权重, 有效地缓解了过拟合的问题

$$ L_{2}(W) \approx \sum_{j=1}^{K} \sum_{m=1}^{B} \frac{\alpha_{m j}}{n_{j}}\left\|w_{j}\right\|_{2}^{2} $$
- amj表示是否至少有一个样本的id为j的特征出现在mini-batch中；nj表示feature id为j出现的次数，惩罚了出现频率低的item

# DIEN(Deep Interest Evolution Network for Click-Through Rate Prediction)

[paper](https://arxiv.org/abs/1809.03672v1), [git](https://github.com/mouna99/dien)

- DIN: **将用户历史行为视作用户兴趣**，并用attention机制来捕捉target和历史行为的相对兴趣
- DIEN: **行为不等于兴趣，需要从行为中挖掘兴趣，并考虑兴趣的动态变化**
  - 兴趣抽取层：计算一个辅助loss，来提升兴趣表达(每个历史行为embedding学习)的准确性< GRU + Loss>
  - 兴趣进化层：更加准确的表达用户兴趣的动态变化性(加权历史行为embedding的学习)< AUGRU(GRU + attention) >

## Model
{% asset_img dien.jpeg dien %}

### Interest Extractor层：兴趣的挖掘。利用GRU+辅助loss来挖掘用户兴趣。
- 辅助loss: 第t个时间步输入e(t)，GRU输出隐单元h(t)，第t步loss: 令下一个时间步的输入向量e(t+1)作为正样本，随机采样负样本e(t+1)。
$$ \begin{aligned} L_{a u x}=-& \frac{1}{N}\left(\sum_{i=1}^{N} \sum_{t} \log \sigma\left(\mathbf{h}_{t}^{i}, \mathbf{e}_{b}^{i}[t+1]\right)\right.\\ &\left.+\log \left(1-\sigma\left(\mathbf{h}_{t}^{i}, \hat{\mathbf{e}}_{b}^{i}[t+1]\right)\right)\right) \end{aligned} $$
- GRU, u表示update gate, r 表示reset gate, h'表示候选的隐藏状态,通过tanh缩放到-1～1之间，对Ht-1进行reset同时+Item, ht 通过last hidden states * (1-update) 表示遗忘，update * h'表示记忆当前状态
$$ \begin{aligned}
\mathbf{u}_{t} &=\sigma\left(W^{u} \mathbf{i}_{t}+U^{u} \mathbf{h}_{t-1}+\mathbf{b}^{u}\right) \\
\mathbf{r}_{t} &=\sigma\left(W^{r} \mathbf{i}_{t}+U^{r} \mathbf{h}_{t-1}+\mathbf{b}^{r}\right) \\
\tilde{\mathbf{h}}_{t} &=\tanh \left(W^{h} \mathbf{i}_{t}+\mathbf{r}_{t} \circ U^{h} \mathbf{h}_{t-1}+\mathbf{b}^{h}\right) \\
\mathbf{h}_{t} &=\left(\mathbf{1}-\mathbf{u}_{t}\right) \circ \mathbf{h}_{t-1}+\mathbf{u}_{t} \circ \tilde{\mathbf{h}}_{t}
\end{aligned} $$

### Interest Evolving层
- Interest Evolving层对与target item相关的兴趣演化轨迹进行建模：利用注意力机制+GRU
- AUGRU: attention update-gate gru。利用attention与update gate相乘，替换原始的update gate
- 变体
  - GRU with attentional input（AIGRU）: 较为简单，直接将attention系数和输入相乘
  - Attention based GRU（AGRU）：采用问答领域文章提到的一种方法，直接将attention系数来替换GRU的update gate，直接对隐状态进行更新

## Problems
- GRU的耗时
- 可以借鉴的：aug loss?

# DSIN(Deep Session Interest Network for Click-Through Rate Prediction)
[paper](https://arxiv.org/abs/1905.06482)
- **session内行为同构，不同session行为异构**
  - Bias encoding + Transformer：获取session内的兴趣表达
  - Bi-LSTM: session间的序列关系
  - activation unit: 类似于din，获得行为序列表达与target item的关系

## Model
{% asset_img dsin.png dsin %}

### Session Division Layer
- 长度为n的seq，切分为K个session，session内有T个item，一个item D维，一个session 30min
- 得到Q ∈ R^(K\*T\*D)

### Session Interest Extractor Layer
- **session内的序列建模**
- bias encoding: BE(k, t, c)指session k，item t下，位置c的偏置值
$$ \mathbf{BE}_{(k, t, c)}=\mathbf{w}_{k}^{K}+\mathbf{w}_{t}^{T}+\mathbf{w}_{c}^{C} $$

$$ \mathbf{Q}=\mathbf{Q}+\mathbf{B} \mathbf{E} $$
- 分别对每个session k：
  - 过一个Transformer得到$$ {I}_{k}^{Q} $$
  - 做avg，得到Ik，shape为K\*D：
  $$ \mathbf{I}_{k}=\operatorname{Avg}\left(\mathbf{I}_{k}^{Q}\right) $$

### Session Interest Interacting Layer
- **session间的序列关系**
- Bi-LSTM

$$
\begin{array}{l}
\mathbf{i}_{t}=\sigma\left(\mathbf{W}_{x i} \mathbf{I}_{t}+\mathbf{W}_{h i} \mathbf{h}_{t-1}+\mathbf{W}_{c i} \mathbf{c}_{t-1}+\mathbf{b}_{i}\right)\\
\mathbf{f}_{t}=\sigma\left(\mathbf{W}_{x f} \mathbf{I}_{t}+\mathbf{W}_{h f} \mathbf{h}_{t-1}+\mathbf{W}_{c f} \mathbf{c}_{t-1}+\mathbf{b}_{f}\right)\\
\mathbf{c}_{t}=\mathbf{f}_{t} \mathbf{c}_{t-1}+\mathbf{i}_{t} \tanh \left(\mathbf{W}_{x c} \mathbf{I}_{t}+\mathbf{W}_{h c} \mathbf{h}_{t-1}+\mathbf{b}_{c}\right)\\
\mathbf{o}_{t}=\sigma\left(\mathbf{W}_{x o} \mathbf{I}_{t}+\mathbf{W}_{h o} \mathbf{h}_{t-1}+\mathbf{W}_{c o} \mathbf{c}_{t}+\mathbf{b}_{o}\right)\\
\mathbf{h}_{t}=\mathbf{o}_{t} \tanh \left(\mathbf{c}_{t}\right)\\
\mathbf{H}_{t}=\overrightarrow{\mathbf{h}_{f t}} \oplus \overleftarrow{\mathbf{h}_{b t}}
\end{array}
$$

### Session Interest Activating Layer
- **与target交互**
- 包括两部分：
  - Ik与target的attention：未建模session间关系
  - Hk与target的attention：建模session间关系之后

## Experiments
{% asset_img dsin_result.jpeg dsin_result %}

## 讨论
- DIN-RNN的效果差与DIN，而DSIN-BE的效果好于DSIN-BE-No-SIIL：说明切分session后，序列建模有效，而切分前，序列建模效果有损。原因猜测为，用户行为在长期来看是跳跃的，序列建模可能在某些时间点出有很大的噪声

# DMIN(Deep Multi-Interest Network for Click-through Rate Prediction)

[paper](https://www.researchgate.net/publication/345125472_Deep_Multi-Interest_Network_for_Click-through_Rate_Prediction)、[推荐系统遇上深度学习(一零零)-[阿里]深度多兴趣网络DMIN](https://www.jianshu.com/p/69929b24bb37)

- 利用multi-head attention机制，建模用户多兴趣
- 引入DIEN中的aux loss

## Model
{% asset_img dmin.jpg dmin %}

### Behavior Refiner Layer
- multi-head attention，提炼用户行为序列，为每个历史行为生成一个新的embedding。HR为head数（R表示Refine），xb的shape: T\*D，W: T\*Dh，得到T\*Dh，concat后得到T\*(Dh*HR)
$$
\begin{aligned}
\operatorname{head}_{h} &=\operatorname{Attention}\left(\mathbf{x}_{b} \mathbf{W}_{h}^{Q}, \mathbf{x}_{b} \mathbf{W}_{h}^{K}, \mathbf{x}_{b} \mathbf{W}_{h}^{V}\right) \\
&=\operatorname{Softmax}\left(\frac{\mathbf{x}_{b} \mathbf{W}_{h}^{Q} \cdot\left(\mathbf{x}_{b} \mathbf{W}_{h}^{K}\right)^{\top}}{\sqrt{d_{h}}} \cdot \mathbf{x}_{b} \mathbf{W}_{h}^{V}\right)
\end{aligned}
$$

$$
\mathrm{Z}=\text { MultiHead }\left(\mathbf{x}_{b}\right)=\text { Concat }\left(\text { head }_{1}, \text { head }_{2}, \ldots, \text { head }_{H_{R}}\right) \mathbf{W}^{O}
$$
- aux loss: 同DIEN

### Multi-Interest Extractor Layer
- multi-head attention，HE个head
$$
\begin{aligned}
\text { head }_{h}^{\prime} &=\operatorname{Attention}\left(\mathbf{Z W}_{h}^{\prime Q}, \mathbf{Z W}_{h}^{\prime K}, \mathbf{Z W}_{h}^{\prime V}\right) \\
&=\operatorname{Softmax}\left(\frac{\mathbf{Z W}_{h}^{\prime} \begin{array}{l}
Q \\
h
\end{array} \cdot\left(\mathbf{Z W}_{h}^{\prime K}\right)^{\top}}{\sqrt{d_{\text {model }}}} \cdot \mathbf{Z W}_{h}^{\prime V}\right)
\end{aligned}
$$
- 类似din，得到第h个兴趣，Ijh为第h个head的第j个item，xt为target item,pj为position encoding：
$$
\text { interest }_{h}=\sum_{j=1}^{T} a\left(\mathbf{I}_{j h}, \mathbf{x}_{t}, \mathbf{p}_{j}\right) \mathbf{I}_{j h}=\sum_{j=1}^{T} w_{j} \mathbf{I}_{j h}
$$

# MIND(Multi-Interest Network with Dynamic Routing for Recommendation at Tmall)

[paper](https://arxiv.org/pdf/1904.08030.pdf)、[推荐系统遇上深度学习(七十四)-[天猫]MIND：多兴趣向量召回](https://www.jianshu.com/p/5e339afbf2e7)

- DIN：通过attention机制建模用户多样的兴趣，并得到一个embedding。MIND采用了另一种表达用户多样兴趣的思路：用多个embedding表示用户多样兴趣，具体来说，可以对用户历史行为的embedding进行聚类，聚类后的每个簇代表用户的一组兴趣。

## Method

{% asset_img mind.jpg mind %}

### behavior2interest dynamic routing
#### 胶囊网络

- 背景：传统的神经网络输入一组标量，对这组标量求加权和，之后输入非线性激活函数得到一个标量的输出。而**Capsule输入是一组向量，对这组向量进行仿射变换之后求加权和，把加权和输入非线性激活函数，如此经过j次迭代得到一个向量的输出**。Hinton提出Capsule Network是为了解决传统的CNN中只能编码某个特征是否存在而**无法编码特征的orientation**。
- 来自[zhihu](https://zhuanlan.zhihu.com/p/68897114)
{% asset_img capsule.jpeg capsule %}
- 一个两层胶囊网络（从low-level到high-level）：
  - low-level: m个Nl维度的vector; high-level: n个Nh维度的vector
  - 对每一个low-level vector做映射，然后做加权融合（softmax），再做非线性激活
  - 其中softmax需要参数，参数需要迭代求解：初始化为0，即权重为1/m；后续迭代更新
{% asset_img capsule2.png capsule2 %}

#### B2I Dynamic Routing
 MIND中的capsule: 
 （在MIND的中我们只要记住Capsule可以接受一组向量输入，输出一个向量；如果我们K个capsule，就会有K个输出向量）