---
title: P、NP、NP-hard问题
mathjax: true
date: 2021-12-15 11:16:27
tags: 
  - math
categories: Learning
---

# 定义

- P Problem: **任意性**，对于**任意的**输入规模n，问题都可以在n的多项式时间内得到解决；
- NP(Non-deterministic Polynomial) Problem: **存在性**，**可以**在多项式的时间里验证一个解的问题；
- NPC(Non-deterministic Polynomial Complete) Problem: 满足两个条件 (1)是一个NP问题 (2)所有的NP问题都可以约化到它。可以理解为**NP的泛化问题**。
- NP-Hard Problem: 满足NPC问题的第二条，但不一定要满足第一条 --> **不一定可以在多项式时间内解决的问题**

# 搞笑版P=NP证明

> 反证法。设P = NP。令y为一个P = NP的证明。证明y可以用一个合格的计算机科学家在多项式时间内验证，我们认定这样的科学家的存在性为真。但是，因为P = NP，该证明y可以在多项式时间内由这样的科学家发现。但是这样的发现还没有发生（虽然这样的科学家试图发现这样的一个证明），我们得到了矛盾。