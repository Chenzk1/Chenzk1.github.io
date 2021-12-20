---
title: CS224WLec1-Graph Intro & Graph Representations
mathjax: false
date: 2021-12-12 13:10:09
tags: 
  - cs224w
  - graph
categories: 
  - Learning
  - cs224w
---

# Intro
- title: Machine Learning with Graphs
- year: Fall 2021
- [CS224W page](http://web.stanford.edu/class/cs224w/)
- [note](https://snap-stanford.github.io/cs224w-notes/)、[note中文翻译](https://yasoz.github.io/cs224w-zh/#/Introduction-and-Graph-Structure)
<!-- more -->
- professor：[Jure Leskovec](https://profiles.stanford.edu/jure-leskovec)
- videos: [bilibili](https://www.bilibili.com/video/BV1RZ4y1c7Co/?spm_id_from=333.788.recommend_more_video.0)
- labs: [lab\*5](https://docs.google.com/document/d/e/2PACX-1vRMprg-Uz9oEnjXOJpRPJ5oyEXRnJAz9qVeEB04sucx2o2RtQ-HRfom6IWS5ONhfoly0TOmJM7BxIzJ/pub)
- schedule: [schedule](http://web.stanford.edu/class/cs224w/index.html#schedule)
- goal: graph的表征学习和用于graph的机器学习算法
- topics
  - Traditional methods: Graphlets, Graph Kernels
  - Methods for node embeddings: DeepWalk, Node2Vec
  - Graph Neural Networks: GCN, GraphSAGE, GAT, Theory of GNNs
  - Knowledge graphs and reasoning: TransE, BetaE
  - Deep generative models for graphs: GraphRNN
  - Applications to Biomedicine, Science, Industry

# Applications
- graph level: graph classification,例如分子属性预测
- node level: node classification, 比如用户/商品分类
- edge level: link prediction，例如knowledge graph completioni、推荐系统、药物副作用预测
- community(subgraph) level: clustering，比如social circle detections
- others
  - graph generation
  - graph evolution
  - ...

# Representation

G=(V, E, R, T)
- node: V
- edge: E
- relation type: R
- node type: T

## 分类
- directed & undirected
- node degree
  - avg degree: 
$$
\bar{k}=\langle k\rangle=\frac{1}{N} \sum_{i=1}^{N} k_{i}=\frac{2 E}{N}
$$
  - in-degree & out-degree
- bipartite graph二部图: 包含两种不同的node，node只和另外一部的node连接。可以通过投影的方式转化为folded/projected bipartite graphs

## 表示
- Adjacency Matrix：无向图时为对称矩阵。**graph大多数时为高度稀疏矩阵（degree远小于节点数），邻接矩阵会造成内存浪费**
- Adjacency List: 对每一个节点存储其neighbors
- 图的附加属性：weight/ranking/type/sign/...

