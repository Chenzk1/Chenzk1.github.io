---
title: ML-Naive Bayes及其sklearn实现
date: 2021-12-10
categories: 
    - MachineLearning
tags:  
    - ML
    - NaiveBayes
mathjax: true
---
[TOC]

P(B|A) = P(A|B)*P(B)/P(A)

朴素：特征之间相互独立

# 算法流程

1. x = {a1, a2, ..., am}为待分类项，a是特征。
2. 类别集合C = {y1, ..., yn}.
3. 计算P(y1|x), P(y2|x) ...
4. P(yk|x) = max{P(yi|x)}，则x属于yk类

<!-- more -->

**总结：**某类在待分类项出现的条件下的概率是所有类中最大的，这个分类项就属于这一类。

e.g.判断一个黑人来自哪个洲，求取每个洲黑人的比率，非洲最高，选非洲。

其中x = {a1, a2, ..., am}，即P(C|a1,a2...) = P(C)\*P(a1,a2,...|C)/P(a1,a2...)。posterior = prior \* likelihood / evidence, 这里evidence是常数，不影响。

----->求解P(C) \* P(a1,a2,a3...|C)

----->链式法则：P(C) \* P(a2,a3...|C, a1) \* P(a1|C)

---> ...

---> P(C) \* P(a1|C) \* P(a2|C, a1) \* P(a3|C, a1, a2)...
由于特征之间的相互独立性，a2发生于a1无关，转化为

---> P(C) \* P(a1|C) \* P(a2|C) ...  \* P(am|C)

----->问题转化为求取条件概率：

1. 找到一个已知分类的待分类项集合，这个集合叫做训练样本集。
2. 统计得到在各类别下各个特征属性的条件概率估计。