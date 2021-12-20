---
title: 6.824Lecture1-MapReduce
date: 2021-12-11 19:08:24
tags: 
  - 6.824
  - MapReduce
categories: 
  - Learning
  - 6.824
---

# Backgorund
- big data
- lots of kv data stuctures, like inverted index

# Method
## Abstract view

{% asset_img workflow.png workflow %}

<!-- more --> 
- **split** files from GFS to disks
- master分配worker执行map任务，生成k,v值，存入disk，map回传disk地址给master
- master传递地址给reduce worker，reduce worker使用RPC读disk数据
- **sort by key**, 并将所有values聚合
- reduce
- master返回reduce结果给GFS

## Fault tolerance
- fail -> rerun: worker rerun不会产生其他问题
- worker fault: Master 周期性的 ping 每个 Worker，如果指定时间内没回应就是挂了。将这个 Worker 标记为失效，分配给这个失效 Worker 的任务将被重新分配给其他 Worker；
- master fault: 中止整个 MapReduce 运算，重新执行。

# Performance
- bandwidth: 尽量保持worker和其执行的文件在同一台机器上，会相近的机器
- slow workers: 吊起执行快的作为backup worker，取最先执行完的worker

# Cons
- 实时性
- 复杂需求时需要大量相互依赖的mr逻辑 -》难开发，难调试

# Refs
- [paper](https://static.googleusercontent.com/media/research.google.com/zh-CN//archive/mapreduce-osdi04.pdf)
- [video](https://www.youtube.com/watch?v=WtZ7pcRSkOA)
- [note](https://mp.weixin.qq.com/s/I0PBo_O8sl18O5cgMvQPYA)
- [the reason why google gives up MapReduce](https://www.the-paper-trail.org/post/2014-06-25-the-elephant-was-a-trojan-horse-on-the-death-of-map-reduce-at-google/)


**作为一种范式，而非产品？**
