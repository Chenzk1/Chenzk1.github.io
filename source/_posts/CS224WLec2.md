---
title: CS224WLec2-Traditional Methods for ML on Graphs
mathjax: true
date: 2021-12-12 17:46:27
tags: 
  - cs224w
  - graph
  - ML
categories: Learning
---

# Pipeline

design features for nodes/link/graphs --> obtain features for all data --> train an ML model --> apply the model
<!-- more -->

# Feature design
- *focus on undirected graphs*

## Node Features

### centrality中心性
- **Degree counts #(edges) that a node touches**
- degree只考虑neibors数量，不考虑neibor的不同重要性

1. Eigenvector centrality
- 定义：节点的重要性由邻居节点的重要性决定。节点v的centrality是邻居centrality的加总，N(v)为v的neibors集合
$$
c_{v}=\frac{1}{\lambda} \sum_{u \in N(v)} c_{u}
$$

可以将其写为矩阵形式，得到 $ \lambda c=A c $ ，A为邻接矩阵，$ \lambda{max} $ 总为正且唯一，因此可以将其对应的$ c_{max} $作为eigenvector

2. betweenness centrality
- 如果一个节点处在很多节点对的最短路径上，那么这个节点是重要的
$$
c_{v}=\sum_{s \neq v \neq t} \frac{\#(\text { shortest paths betwen } s \text { and } t \text { that contain } v)}{\#(\text { shortest paths between } s \text { and } t)}
$$

3. closeness centrality
- 一个节点距其他节点之间距离最短，那么认为这个节点是重要的
$$
c_{v}=\frac{1}{\sum_{u \neq v} \text { shortest path length between } u \text { and } v}
$$

### clustering coefficient
- **Clustering coefficient counts #(triangles) that a node touches.**
- 节点的neighbors的连接情况
$$
e_{v}=\frac{\#(\text { edges among neighboring nodes })}{\left(\begin{array}{c}
k_{v} \\
2
\end{array}\right)} \in[0,1]
$$

![](https://img-blog.csdnimg.cn/2021052810445748.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
![](https://img-blog.csdnimg.cn/20210528105114750.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)

### Graphlets
- Rooted connected induced non-isomorphic subgraphs, 有根连通异构子图

![](https://img-blog.csdnimg.cn/20210528121841575.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
- 图中标的数字代表根节点可选的位置。例如对于$ G_0 $，两个节点是等价的（对称的），所以只有一种graphlet；对于$ G_1 $，根节点有在中间和在边上两种选择，上下两个边上的点是等价的，所以只有两种graphlet。其他的类似。节点数为2-5情况下一共能产生如图所示73种graphlet。**这73个graphlet的核心概念就是不同的形状，不同的位置。图里的数字表示graphlets的数量**
- Graphlet Degree Vector (GDV): A count vector of graphslets rooted at a given node
- **GDV counts #(graphlets) that a node touches**
![](https://img-blog.csdnimg.cn/20210528123912356.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
- 以v作为节点的有根连通异构子图共4个，分别为a b c d为节点的子图。a有两种情况；b一种情况；c不存在是因为graphlet是induced subgraph，c可以induce为b；d有两种情况。所以得到的GDV为[2, 1, 0, 2]
- 考虑2-5个节点的graphlets，我们得到一个长度为73个坐标coordinate（就前图所示一共73种graphlet）的向量GDV，描述该点的局部拓扑结构topology of node’s neighborhood，可以捕获距离为4 hops的互联性interconnectivities
- 相比节点度数或clustering coefficient，GDV能够描述两个节点之间更详细的节点局部拓扑结构相似性local topological similarity。

### Summary
- 节点的重要性importance-based: node degree/centrality
- 节点邻域的拓扑结构structure-based: node degree/clustering cofficient/graphlet count vector

## Link features
