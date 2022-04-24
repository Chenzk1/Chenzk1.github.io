---
title: 机器学习面试题-RNN&LSTM
date: 2021-12-10
categories: 
    - MachineLearning
tags:  
    - DL
    - NLP
    - RNN
    - LSTM
mathjax: true
---
<meta name="referrer" content="no-referrer"/>
[TOC]

## LSTM产生的原因

- **RNN在处理长期依赖**（时间序列上距离较远的节点）时会遇到巨大的困难，因为计算距离较远的节点之间的联系时会涉及雅可比矩阵的多次相乘，会造成**梯度消失或者梯度膨胀**的现象。RNN结构之所以出现梯度爆炸或者梯度消失，最本质的原因是因为梯度在传递过程中存在极大数量的连乘 。
- 相对于RNN，LSTM的神经元加入了**输入门i、遗忘门f、输出门o 、内部记忆单元c** 

<!-- more -->

## 分别介绍一下输入门i、遗忘门f、输出门o 、内部记忆单元c

- 内部记忆单元$c$

  -  类似于传送带。直接在整个链上运行，只有一些少量的线性交互。信息在上面流传保持不变会很容易。  
  - 操作步骤：
  - 示意图
    ![](https://img-blog.csdn.net/20170919124608594?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHJlYWRlcmw=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

* 遗忘门$f$

  * 将**内部记忆单元**中的信息选择性的遗忘

  * 操作步骤：

    * 读取：$h_{t-1}$、$x_t$，
    * 输出：$f_{t}=\sigma\left(W_{f} \cdot\left[h_{t-1}, x_{t}\right]+b_{f}\right)$
    * $\sigma$表示一个在 0 到 1 之间的数值。1 表示“完全保留”，0 表示“完全舍弃”

  * 示意图
    ![遗忘门](https://pic4.zhimg.com/80/v2-11ca9e4a19504874202ac9880da9840f_1440w.jpg)

  $$
  f_{t}=\sigma\left(W_{f} \cdot\left[h_{t-1}, x_{t}\right]+b_{f}\right)
  $$

* 输入门$i$

  * 将新的信息记录到**内部记忆单元**中

  * 操作步骤：

    * 步骤一：$sigmoid$ 层称 **输入门层**决定什么值我们将要更新。

    * 步骤二：$tanh$ 层创建一个新的候选值向量$\tilde{C}_t$加入到状态中。其示意图如下：

      ![1578624753318](https://img-blog.csdn.net/20170301115512234?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvSmVycl9feQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
      $$
      i_{t}=\sigma\left(W_{i} \cdot\left[h_{t-1}, x_{t}\right]+b_{i}\right)
      $$

      $$
      \tilde{C}_{t}=\tanh \left(W_{C} \cdot\left[h_{t-1}, x_{t}\right]+b_{C}\right)
      $$

    * 步骤三：将$C_{t-1}$更新为$C_{t}$。将旧状态与$f_t$相乘，丢弃掉我们确定需要丢弃的信息。接着加上$i_t * \tilde{C}_t$得到新的候选值，根据我们决定更新每个状态的程度进行变化。其示意图如下：

      ![1578624724604](https://img-blog.csdn.net/20170301120227745?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvSmVycl9feQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
      $$
      C_{t}=f_{t} * C_{t-1}+i_{t} * \tilde{C}_{t}
      $$

* 输出门$o$

  * 确定隐层$h_t$输出什么值

  * 操作步骤：

    * 步骤一：通过sigmoid 层来确定细胞状态的哪个部分将输出。

    * 步骤二：把细胞状态通过 tanh 进行处理，并将它和 sigmoid 门的输出相乘，最终我们仅仅会输出我们确定输出的那部分。

      ![输出门](https://pic4.zhimg.com/80/v2-f928df2c02e17fb5da95bf8354880613_1440w.jpg)
      $$
      \begin{array}{l}
      {o_{t}=\sigma\left(W_{o}\left[h_{t-1}, x_{t}\right]+b_{o}\right)} \\
      {h_{t}=o_{t} * \tanh \left(C_{t}\right)}
      \end{array}
      $$