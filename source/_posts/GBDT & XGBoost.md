---
title: ML-GBDT & XGBoost
categories: 
    - Learning
tags:  
    - ML
mathjax: true
---
[TOC]

[Doc](https://xgboost.readthedocs.io/en/latest/tutorials/model.html)
[Slides](https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)

<!-- more -->

# 为何要推导出目标函数而不是直接增加树

![Objective function](http://i.imgur.com/quPhp1K.png)

- 理论上：搞清楚learning的目的，以及其收敛性。
- 工程上：
  - gi和hi是对loss function的一次、二次导
  - 目标函数以及整个学习过程只依赖于gi和hi
  - 可以根据实际问题，自定义loss function

# Summary

![Summary](http://i.imgur.com/L7PhJwO.png)

# 原理

## 损失函数

$$ \text{obj} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t)}) + \sum_{i=1}^t\Omega(f_i)$$
l为loss，\\ \Omega \\ 为正则项

- loss：采用加法策略，第t颗树时：
  $$ \hat{y}_i^{(0)} = 0 $$
  $$ \hat{y}_i^{(1)} = f_1(x_i) = \hat{y}_i^{(0)} + f_1(x_i) $$
  $$ \hat{y}_i^{(2)} = f_1(x_i) + f_2(x_i)= \hat{y}_i^{(1)} + f_2(x_i) $$
  $$ \dots $$
  $$ \hat{y}_i^{(t)} = \sum_{k=1}^t f_k(x_i)= \hat{y}_i^{(t-1)} + f_t(x_i) $$
  在添加第t颗树时，需要优化的目标函数为：
  $$ \sum_{i=1}^n [g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \Omega(f_t) $$
  其中h和f：
  $$ g_i = \partial_{\hat{y}_i^{(t-1)}} l(y_i, \hat{y}_i^{(t-1)}) $$
  $$ h_i = \partial_{\hat{y}_i^{(t-1)}}^2 l(y_i, \hat{y}_i^{(t-1)}) $$
  note: 是对谁的导
- 正则项：复杂度：
  $$ \Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2 $$
  其中w是叶子上的score vector，T是叶子数量

## DART Booster

为了解决过拟合，会随机drop trees:

- 训练速度可能慢于gbtree
- 由于随机性，早停可能不稳定

# 特性

## Monotonic Constraints单调性限制

- 一个可选特性:
  会限制模型的结果按照某个特征 单调的进行增减
  
  也就是说可以降低模型对数据的敏感度，如果明确已知某个特征与预测结果呈单调关系时，那在生成模型的时候就会跟特征数据的单调性有关。

## Feature Interaction Constraints单调性限制

- 一个可选特性：
  不用时，在tree生成的时候，一棵树上的节点会无限制地选用多个特征

  设置此特性时，可以规定，哪些特征可以有interaction（一般独立变量之间可以interaction，非独立变量的话可能会引入噪声）
- 好处：
  - 预测时更小的噪声
  - 对模型更好地控制

## Instance Weight File

- 规定了模型训练时data中每一条instance的权重
- 有些instance质量较差，或与前一示例相比变化不大，所以可以调节其所占权重

# 调参

## Overfitting

与overfitting有关的参数：

- 直接控制模型复杂度：max_depth, min_child_weight and gamma.
- 增加模型随机性以使得模型对噪声有更强的鲁棒性：
  - subsample and colsample_bytree. 
  - Reduce stepsize eta. Remember to increase num_round when you do so.

## Imbalanced Dataset

- 只关注测量指标的大小
  - 平衡数据集 via scale_pos_weight
  - 使用AUC作为metric
- 关注预测正确的概率
  - 此时不能re-balance数据集
  - Set parameter max_delta_step to a finite number (say 1) to help convergence