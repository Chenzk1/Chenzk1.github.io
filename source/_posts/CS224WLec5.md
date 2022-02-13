---
title: CS224WLec5-Label Propagation for Node Classification[LIP]
mathjax: true
date: 2021-12-19 14:59:40
tags:
  - cs224w
  - Graph
categories:
  - Learning
  - cs224w
---

[ppt](http://web.stanford.edu/class/cs224w/slides/05-message.pdf)

# Semi-supervised binary node classification
- goal: 知道部分节点的label求其他节点的label
- main assumption：网络中存在的同质性。同类的node倾向于相互连接，或者聚在一起。
- framework
  - 初始化
  - 迭代
  - 收敛
- approaches
  - relational classification
  - iterative classification
  - correct & smooth
- 以节点二分类为例

<!-- more -->

# Approaches
## Probabilistic relational classifier
- **只用到了邻居节点的label来做分类**
- step1: 初始化所有unlabeled nodes为0.5
- step2: update所有unlabeled nodes的预测值 $$ P\left(Y_{v}=c\right)=\frac{1}{\sum_{(v, u) \in E} A_{v, u}} \sum_{(v, u) \in E} A_{v, u} P\left(Y_{u}=c\right) $$
  - 这里A为邻接矩阵，若边有权重，则Au,v是带权邻接矩阵
- step3: 重复step2直到收敛：达到最大steps，或者所有节点label不再更新
- challenges: 
  - 不能保证收敛
  - 没有用到节点特征信息

## Iterative classification
- Classify node v based on its attributes $$f_v$$ as well as labels $$z_v$$ of neighbor set $$N_v$$
- $$z_v$$: v的neibors的label特征，可以是
  - $$N_v$$中每种label的histogram
  - $$N_v$$中数量最多的label
  - 不同label的数量
- 流程
{% asset_img classifier1.png classifier1 %}

## Collective classification: correct & smooth
- recent state-of-the-art collective classification method
- 一种后处理方法
- 基于的假设：相邻节点的误差应该接近
- correct step
  - 初始error: labeled node: error = ground truth minus soft label; unlabeled node: error=0
  - 利用扩散矩阵 $$ \widetilde{\boldsymbol{A}} $$求取下一步的error
  $$ \boldsymbol{E}^{(t+1)} \leftarrow(1-\alpha) \cdot \boldsymbol{E}^{(t)}+\alpha \cdot \widetilde{\boldsymbol{A}} \boldsymbol{E}^{(t)} $$

