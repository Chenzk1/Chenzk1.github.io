---
title: tfidf_bm25
mathjax: true
date: 2022-03-18 10:02:08
tags:
    - Tf-idf
    - BM25
categories:
    - MachineLearning
---

tfidf_bm25
<!-- more -->
[Ref](https://my.oschina.net/stanleysun/blog/1617727)

# TF-IDF
- TF Score ＝ 某个词在文档中出现的次数 ／ 文档的长度
- IDF = log(N/n)
- Lucence中的TF-IDF: simlarity = log(numDocs / (docFreq + 1)) * sqrt(tf) * (1/sqrt(length))
    - numDocs:索引中文档数量，对应前文中的N。lucence不是(也不可能)把整个互联网的文档作为基数，而是把索引中的文档总数作为基数。
    - docFreq: 包含关键字的文档数量，对应前文中的n。
    - tf: 关键字在文档中出现的次数。
    - length: 文档的长度。
- 上面的公式在Lucence系统里做计算时会被拆分成三个部分：
    - IDF Score = log(numDocs / (docFreq + 1))
    - TF Score = sqrt(tf)
    - fieldNorms = 1/sqrt(length)
- fieldNorms 是对文本长度的归一化(Normalization)。所以，上面公式也可以表示成: simlarity = IDF score * TF score * fieldNorms

# BM25
- BM: best match
- 与TF-IDF的比较：
    - 1）BM25 = IDF * TF * QF
    - 2）QF为查询词在query中的权重
    - 3）TF做了一些改善，包括：对TF的值做了限定，限制在0~k+1；引入了平均文档长度，并将文档长度/平均文档长度引入TF公式，使得文档长度越大，TF Score随TF的增加越快增长到上界

## TF
- 传统 TF Score = sqrt(tf)
- BM25的 TF Score = ((k + 1) * tf) / (k + tf)
- 下面是两种计算方法中，词频对TF Score影响的走势图。从图中可以看到，当tf增加时，TF Score跟着增加，但是BM25的TF Score会被限制在0~k+1之间。它可以无限逼近k+1，但永远无法触达它。这在业务上可以理解为某一个因素的影响强度不能是无限的，而是有个最大值，这也符合我们对文本相关性逻辑的理解。
{% asset_img bm251.png bm251 %}
- 文档长度：BM25还引入了平均文档长度的概念，单个文档长度对相关性的影响力与它和平均长度的比值有关系。BM25的TF公式里，除了k外，引入另外两个参数：L和b。L是文档长度与平均长度的比值。如果文档长度是平均长度的2倍，则L＝2。b是一个常数，它的作用是规定L对评分的影响有多大。加了L和b的公式变为：
```
# 即文档长度越大，TF Score值越小
TF Score = ((k + 1) * tf) / (k * (1.0 - b + b * L) + tf) 
```
- 下面是不同L的条件下，词频对TFScore影响的走势图：
{% asset_img bm252.png bm252 %}
- 从图上可以看到，文档越短，它逼近上限的速度越快，反之则越慢。这是可以理解的，对于只有几个词的内容，比如文章“标题”，只需要匹配很少的几个词，就可以确定相关性。而对于大篇幅的内容，比如一本书的内容，需要匹配很多词才能知道它的重点是讲什么。上文说到，参数b的作用是设定L对评分的影响有多大。如果把b设置为0，则L完全失去对评分的影响力。b的值越大，L对总评分的影响力越大。

## IDF
$$
\sum_{i: q_{i}=d_{i}=1} \log \frac{\left(r_{i}+0.5\right) /\left(R-r_{i}+0.5\right)}{\left(n_{i}-r_{i}+0.5\right) /\left((N-R)-\left(n_{i}-r_{i}\right)+0.5\right)}
$$
{% asset_img bm253.png bm253 %}


## 查询词权重
- qf是查询词在用户查询中的频率，但一般用户查询都比较短，qf通常是1，K2是经验参数。
$$ \frac{\left(k_{2}+1\right) qf_{i}}{k_{2}+qf_{}} $$
- 此时，相似度最终的完整公式为
$$
\begin{aligned}
\sum_{i \in Q} \log \frac{\left(r_{i}+0.5\right) /\left(R-r_{i}+0.5\right)}{\left(n_{i}-r_{i}+0.5\right) /\left(N-n_{i}-R+r_{i}+0.5\right)} \cdot \frac{\left(k_{1}+1\right) f_{i}}{k+f_{i}} \cdot \frac{\left(k_{2}+1\right) q}{k_{2}+q f i} \\
& \downarrow \\
k &=k_{1}\left((1-b)+b \cdot \frac{d l}{a v d_1}\right)
\end{aligned}
$$

