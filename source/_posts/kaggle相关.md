---
title: Kaggle-Kaggle相关
date: 2021-12-10
categories: 
    - MachineLearning
tags:  
    - ML
    - Kaggle
---
[TOC]

# 如何在 Kaggle 首战中进入前 10%

[原文](https://dnc1994.com/2016/04/rank-10-percent-in-first-kaggle-competition/)

<!-- more -->

## 流程

### Exploration Data Analysis(EDA)

#### Visualization

matplotlib + seaborn

- 查看目标变量的分布。当分布不平衡时，根据评分标准和具体模型的使用不同，可能会严重影响性能。
- 对 Numerical Variable，可以用 Box Plot 来直观地查看它的分布。
- 对于坐标类数据，可以用 Scatter Plot 来查看它们的分布趋势和是否有离群点的存在。
- 对于分类问题，将数据根据 Label 的不同着不同的颜色绘制出来，这对 Feature 的构造很有帮助。
- 绘制变量之间两两的分布和相关度图表。

[example_visualization](https://www.kaggle.com/benhamner/python-data-visualizations)

#### Statistical Tests

可视化为定性，这里专注于定量，例如对于新创造的特征，可以将其加入原模型当中，看结果的变化。

在某些比赛中，由于数据分布比较奇葩或是噪声过强，Public LB(Leader board)的分数可能会跟 Local CV(Cross Validation)的结果相去甚远。可以根据一些统计测试的结果来粗略地建立一个阈值，用来衡量一次分数的提高究竟是实质的提高还是由于数据的随机性导致的。

### Data Preprossing

处理策略主要依赖于EDA中得到的结论。

- 有时数据会分散在几个不同的文件中，需要 Join 起来。
- 处理 Missing Data。
- 处理 Outlier。
- 必要时转换某些 Categorical Variable 的表示方式。例如应用one-hot encoding(pd.get_dummies)将categorical variable转化为数字变量。
- 有些 Float 变量可能是从未知的 Int 变量转换得到的，这个过程中发生精度损失会在数据中产生不必要的 Noise，即两个数值原本是相同的却在小数点后某一位开始有不同。这对 Model 可能会产生很负面的影响，需要设法去除或者减弱 Noise。

### Feature Engineering

#### Feature Selection

总的来说，应该**生成尽量多的 Feature，相信 Model 能够挑出最有用的 Feature**。但有时先做一遍 Feature Selection 也能带来一些好处：

- Feature 越少，训练越快。
- 有些 Feature 之间可能存在线性关系，影响 Model 的性能。
- 通过挑选出最重要的 Feature，可以将它们之间进行各种运算和操作的结果作为新的 Feature，可能带来意外的提高。
- Feature Selection 最实用的方法也就是看 Random Forest 训练完以后得到的 Feature Importance 了。其他有一些更复杂的算法在理论上更加 Robust，但是缺乏实用高效的实现。从原理上来讲，增加 Random Forest 中树的数量可以在一定程度上加强其对于 Noisy Data 的 Robustness。

看 Feature Importance 对于某些数据经过脱敏处理的比赛尤其重要。这可以免得你浪费大把时间在琢磨一个不重要的变量的意义上。(脱敏：数据脱敏(Data Masking),又称数据漂白、数据去隐私化或数据变形。百度百科对数据脱敏的定义为：指对某些敏感信息通过脱敏规则进行数据的变形，实现敏感隐私数据的可靠保护。在涉及客户安全数据或者一些商业性敏感数据的情况下，在不违反系统规则条件下，对真实数据进行改造并提供测试使用，如身份证号、手机号、卡号、客户号等个人信息都需要进行数据脱敏。)

#### Feature Encoding

假设有一个 Categorical Variable 一共有几万个取值可能，那么创建 Dummy Variables 的方法就不可行了。这时一个比较好的方法是根据 Feature Importance 或是这些取值本身在数据中的出现频率，为最重要（比如说前 95% 的 Importance）那些取值（有很大可能只有几个或是十几个）创建 Dummy Variables，而所有其他取值都归到一个“其他”类里面。

### Model Selection

Base Model:

- SVM
- Linear Regression
- Logistic Regression
- Neural Networks

Most Used Models:

- Gradient Boosting
- Random Forest
- Extra Randomized Trees

  **XGBoost**

#### Model Training

通过Grid Search来确定模型的最佳参数。
e.g.

- sklearn 的 RandomForestClassifier 来说，比较重要的就是随机森林中树的数量 n_estimators 以及在训练每棵树时最多选择的特征数量 max_features。
- Xgboost 的调参。通常认为对它性能影响较大的参数有：
  - eta：每次迭代完成后更新权重时的步长。越小训练越慢。
  - num_round：总共迭代的次数。
  - subsample：训练每棵树时用来训练的数据占全部的比例。用于防止 Overfitting。
  - colsample_bytree：训练每棵树时用来训练的特征的比例，类似 RandomForestClassifier 的 max_features。
  - max_depth：每棵树的最大深度限制。与 Random Forest 不同，Gradient Boosting 如果不对深度加以限制，最终是会 Overfit 的。
  - early_stopping_rounds：用于控制在 Out Of Sample 的验证集上连续多少个迭代的分数都没有提高后就提前终止训练。用于防止 Overfitting。
  
  一般的调参步骤是：

  1. 将训练数据的一部分划出来作为验证集。
  2. 先将 eta 设得比较高（比如 0.1），num_round 设为 300 ~ 500。
  3. 用 Grid Search 对其他参数进行搜索。
  4. 逐步将 eta 降低，找到最佳值。
  5. 以验证集为 watchlist，用找到的最佳参数组合重新在训练集上训练。注意观察算法的输出，看每次迭代后在验证集上分数的变化情况，从而得到最佳的 early_stopping_rounds。

  *所有具有随机性的 Model 一般都会有一个 seed 或是 random_state 参数用于控制随机种子。得到一个好的 Model 后，在记录参数时务必也记录下这个值，从而能够在之后重现 Model。*

#### Cross Validation

一般5-fold。

fold越多训练越慢。

#### Ensemble Generation

常见的 Ensemble 方法有这么几种：

- Bagging：使用训练数据的不同随机子集来训练每个 Base Model，最后进行每个 Base Model 权重相同的 Vote。也即 Random Forest 的原理。
- Boosting：迭代地训练 Base Model，每次根据上一个迭代中预测错误的情况修改训练样本的权重。也即 Gradient Boosting 的原理。比 Bagging 效果好，但更容易 Overfit。
- Blending：用不相交的数据训练不同的 Base Model，将它们的输出取（加权）平均。实现简单，但对训练数据利用少了。
- Stacking：接下来会详细介绍。

从理论上讲，Ensemble 要成功，有两个要素：

- Base Model 之间的相关性要尽可能的小。这就是为什么非 Tree-based Model 往往表现不是最好但还是要将它们包括在 Ensemble 里面的原因。Ensemble 的 Diversity 越大，最终 Model 的 Bias 就越低。
- Base Model 之间的性能表现不能差距太大。这其实是一个 Trade-off，在实际中很有可能表现相近的 Model 只有寥寥几个而且它们之间相关性还不低。但是实践告诉我们即使在这种情况下 Ensemble 还是能大幅提高成绩。

### Pipeline

workflow比较复杂，因此一个高自动化的pipeline比较重要。

这里是以一个例子：[example](https://github.com/ChenglongChen/Kaggle_CrowdFlower)