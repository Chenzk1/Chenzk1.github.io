---
title: CV-目标检测综述
categories: 
    - Learning
tags:  
    - ML
    - DL
    - CV
mathjax: true
---
<meta name="referrer" content="no-referrer"/>

## 目标检测

### 定义

- 图像识别+定位
- 识别：分类问题，准确率
- 定位：分类/回归问题，找到一个框/4个坐标，IOU
  
## 传统目标检测

### 用回归做定位问题

- 训练一个cnn网络，在最后一个卷积层后分两个head，一个head做分类，另一个回归
- 先fine tuning分类任务，再fine tuning回归
- 缺点：回归问题，很难做

### 用分类做定位问题

- 滑动窗口（选择不同位置不同大小的区域），对其定位，对每个框内的图像做分类
- 缺点：窗口冗余、复杂度高、多物体多分类时复杂度更高