---
title: GraphEmbedding-LINE
mathjax: true
date: 2021-12-20 15:29:30
tags:
  - Graph
  - LINE
categories:
  - MachineLearning
---

paper: [LINE: Large-scale Information Network Embedding](https://arxiv.org/pdf/1503.03578.pdf)

# Intro
- 包含一阶信息（直接相连的两节点的邻近度）和二阶信息（一对顶点的邻近度用临近节点的相似性衡量）的GE学习模型：**只用到了一阶邻点**
- DeepWalk/node2vec: 基于高阶相似度的GE学习模型

<!-- more -->

# Methods
- 边权重矩阵W：损失函数计算时用
- 节点权重矩阵：节点采样时用

## 一阶相似度
- **只针对无向图，不支持有向图**
- 边(u,v)的权重Wuv表示u和v之间的一阶相似性，如果在u和v之间无边，它们的一阶相似性为0。
- 经验分布$$ \hat{p}_{1}(i, j)=\frac{w_{i j}}{W} $$，这里W是W矩阵的模
- 预测分布 $$ p_{1}\left(v_{i}, v_{j}\right)=\frac{1}{1+\exp \left(-\vec{u}_{i}^{T} \cdot \vec{u}_{j}\right)} $$
- 目标函数 $$ O_{1}=d\left(\hat{p}_{1}(\cdot, \cdot), p_{1}(\cdot, \cdot)\right) $$, d为距离函数，例如当d为KL散度且忽略常数项时（同交叉熵）：
$$
O_{1}=-\sum_{(i, j) \in E} w_{i j} \log p_{1}\left(v_{i}, v_{j}\right)
$$

## 二阶相似度
- 如果没有同样的相邻节点，二阶相似性为0
- 经验分布 $$ \hat{p}_{2}\left(v_{j} \mid v_{i}\right)=\frac{w_{i j}}{d_{i}} $$，di是节点i的带权出度，$$w_{i j}$$是边(i,j)的权重，$$ d_{i}=\sum_{k \in N(i)} w_{i k} $$
- 预测分布 $$ p_{2}\left(v_{j} \mid v_{i}\right)=\frac{\exp \left(\vec{u}_{j}^{\prime T} \cdot \vec{u}_{i}\right)}{\sum_{k=1}^{|V|} \exp \left(\vec{u}_{k}^{\prime T} \cdot \vec{u}_{i}\right)} $$
- 目标函数 $$ O_{2}=\sum_{i \in V} \lambda_{i} d\left(\hat{p}_{2}\left(\cdot \mid v_{i}\right), p_{2}\left(\cdot \mid v_{i}\right)\right) $$，这里$$\lambda_{i}$$为对节点i的加权，一般取 $$\lambda_{i}$$ 为 $$d_{i}$$，并取KL散度为距离函数时：$$ O_{2}=-\sum_{(i, j) \in E} w_{i j} \log p_{2}\left(v_{j} \mid v_{i}\right) $$

## 结合一阶和二阶
- method1: 训完一阶训二阶，并将两者embedding concat
- method2: 一起训

## 其他损失函数
- pair loss: 用learn2rank的loss

## 优化
### 二阶相似度计算的负采样
- 对softmax的优化：负采样，$$\sigma$$为sigmoid函数，$$P_{n}(v) \propto d_{v}^{3 / 4}$$利用出度构成的节点权重做采样，与w2v分布一致
$$
\log \sigma\left(\vec{u}_{j}^{\prime T} \cdot \vec{u}_{i}\right)+\sum_{i=1}^{K} E_{v_{n} \sim P_{n}(v)}\left[\log \sigma\left(-\vec{u}_{n}^{\prime T} \cdot \vec{u}_{i}\right)\right]
$$

### 边采样
- O2的梯度与边权重wij有关，若学习率的设定由小权重决定，则遇到大权重时会梯度爆炸；反之则会梯度消失
$$
\frac{\partial O_{2}}{\partial \vec{u}_{i}}=w_{i j} \cdot \frac{\partial \log p_{2}\left(v_{j} \mid v_{i}\right)}{\partial \vec{u}_{i}}
$$
- 简单的优化方式：将权重为w的edge变为w条二进制edge --> oom --> edge sampling
- 采样算法：alias算法，可以达到O(1)复杂度

## Problems
- 新节点的GE（新节点）：
  - 新节点和已有节点相连：优化如下目标函数 $$ -\sum_{j \in N(i)} w_{j i} \log p_{1}\left(v_{j}, v_{i}\right), \text { or }-\sum_{j \in N(i)} w_{j i} \log p_{2}\left(v_{j} \mid v_{i}\right) $$
  - 新节点不和已有节点相连：文中没给 --》利用side info
- 低度数顶点（孤岛节点）：对于一些顶点由于其邻接点非常少会导致embedding向量的学习不充分，论文提到可以利用邻居的邻居构造样本进行学习，这里也暴露出LINE方法仅考虑一阶和二阶相似性，对高阶信息的利用不足。

# 注意点
- 边权重矩阵和节点权重矩阵（pagerank/出度/centrality/clustering coefficient/...）的设计
- 不同的loss: KL/cross-entropy/rankloss/...
- 采样算法

# Pros & Cons
- 效率高
- 只用了一跳信息
- 对孤岛节点和新节点没有好的处理