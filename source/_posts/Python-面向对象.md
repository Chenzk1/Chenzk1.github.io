---
title: Python-面向对象
date: 2021-12-10
tags: Python
categories: 
    - Language
    - Python
---

### 属性命名

* 属性以双下划线开头，类内变量，实例无法访问。但可以通过某些方式访问，例如Student例中定义了_\_name变量，可以用_Student_name来实现访问，但不建议，因为不同的解释器的转化方式不一样。
* 单下划线可以打开，但需要注意不能随意更改。
* 双下划线结尾与开头，特殊变量，类内可以访问，实例不知。

<!-- more -->

### 多态

开闭原则：定义一个类Animal及其多个之类Dog/Cat/...，当定义一个函数或操作时：

- 对扩展开放：允许新增Animal的子类；
- 对修改封闭：不需要修改依赖Animal类型的run_twice()等函数，仍然可以传入Dog/Cat等类。
事实上，不需要继承也可以实现多态————鸭子类型。

### 若干方法

- isinstance(object,class) 判断是否属于某个类
- dir() 列举出一个对象的属性和方法
- getattr()、setattr()、hasattr()可以获得、添加、查询是否需要某个属性
  - \_\_slots\_\_ 限制可以添加的属性，\_\_slots\_\_ = ('name', 'age') # 用tuple定义允许绑定的属性名称
- 装饰器