---
title: CS224WLec4-PageRank
mathjax: true
date: 2021-12-18 21:34:53
tags:
  - cs224w
  - graph
  - ML
  - PageRank
categories:
  - Learning
  - cs224w
---

# PageRank
- 一个node的重要性由指向它的节点的重要性决定 --> **指向该node的节点数越多，且这些节点重要性越高，则该节点越重要**。node j的重要性（传递函数）为：
$$ r_{j}=\sum_{i \rightarrow j} \frac{r_{i}}{d_{i}} $$
$$d_{i} \ldots out-degree of node i$$
- 矩阵形式：
  - 随机邻接矩阵M。j有$$d_{j}$$个出链，如果j -> i，则$$M_{ij}=1/d_{j}$$，因此每一列上的值要么为0，要么为$$1/d_{j}$$，且加和为1，称为列随机矩阵
  - rank vector r: $$r_{j}$$为node j的重要性score，且$$\sum_{i} r_{i}=1$$
  - n\*1 = n\*n \* n\*1: 
  $$ \boldsymbol{r}=\boldsymbol{M} \cdot \boldsymbol{r} $$,
  $$\quad r_{j}=\sum_{i \rightarrow j} \frac{r_{i}}{d_{i}} $$
- PageRank VS RandomWalk: 当random walk到达静态分布状态时满足$$ \boldsymbol{r}=\boldsymbol{M} \cdot \boldsymbol{r} $$，即**PageRank得到的重要性向量v是random walk的静态分布**
- PangRank VS Eigenvector: **PageRank得到的重要性向量v是当特征值为1时得到的主特征向量**
<!-- more -->

## Solve the equation
- method: power iteration
  - initialize: 初始化，给每个node一个page rank值。e.g. 
$$
\boldsymbol{r}^{0}=[1 / N, \ldots, 1 / N]^{T}
$$
  - iterate: 迭代
$$
r_{j}^{(t+1)}=\sum_{i \rightarrow j} \frac{r_{i}^{(t)}}{d_{i}}
$$
即：
$$
\boldsymbol{r}^{(t+1)}=\boldsymbol{M} \cdot \boldsymbol{r}^{t}
$$
  - stop: 可以选择其他度量方式
$$
\left|\boldsymbol{r}^{(t+1)}-\boldsymbol{r}^{t}\right|_{1}<\varepsilon
$$

### Problems
- dead ends: 遇到死胡同 --》 某个节点所在列的M为全0 --》随机矩阵不再随机，秩 < n --》pagerank会收敛为全0
  - solution: 填充M中的该列，例如填充为1/n
- spider trap: 在某个节点处死循环
  - solution: 允许random surfer跳到一个随机page。加一个超参β，表示是否沿着这条link走的概率，1-β表示随机jump的概率 --》相当于改了M
- Google solution: teleport rankpage。β表示page j由其他page的出链决定，1-β表示由其他随机page决定
  - equation
  $$
  r_{j}=\sum_{i \rightarrow j} \beta \frac{r_{i}}{d_{i}}+(1-\beta) \frac{1}{N}
  $$
  - matrix G
  $$ \boldsymbol{G}=\beta {\boldsymbol{M}}+(1-\beta)[1/N]_{N*N} $$

{% asset_img pagerank1.png pagerank1 %}

# PageRank扩展
- Personalized PageRank: teleport到在指定的子图S
- Random walks with restarts: teleport到start page
{% asset_img pagerank2.png pagerank2 %}

# Limitations
- 无法获得训练集里没有的node的embedding，必须重新训练
- 无法捕获结构相似性：例如同一张图中的局部结构相似性
- 无法利用node/edge以及图的特征：例如无法利用节点的其他属性
- 解决方法：deep learning, GNN
