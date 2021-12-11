---
title: Python-sort&sorted
tags: Python
categories: Learning
---

## 不同

- sorted
  - 返回已排序**列表**
  - built-in函数，接受任何可迭代对象
- sort
  - 原位排序，返回None
  - 是list的成员函数，因此只接受list

## 相同

- 稳定排序

<!-- more -->

### 参数

- key：接受一个参数，返回用于排序的**键**
  - operator模块函数：一些常用key函数的封装
    - itemgetter()：适用tuple/list的list
    - attrgetter()：适用dict/object的list
    - methodcaller()
- reverse

### 自定义排序规则

#### Python2: cmp函数

```python
def cmp(a, b):
  # 如果逻辑上认为 a < b ，返回 -1
  # 如果逻辑上认为 a > b , 返回 1
  # 如果逻辑上认为 a == b, 返回 0 
  pass

a = [2,3,1,2]
a = sorted(a, cmp=cmp)
```

#### python3：无cmp，只存key。两种自定义排序规则的做法。

- 做法1：利用functools module里封装的cmp_to_key

```python
def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K
```

```python
import functools
def cmp(a, b):
    if b < a:
        return -1
    if a < b:
        return 1
    return 0
a = [1, 2, 5, 4]
print(sorted(a, key=functools.cmp_to_key(cmp)))
```

- 做法2：自己实现cmp类

```python
class LargerNumKey(str):
    def __lt__(x, y):
        return x+y > y+x

largest_num = ''.join(sorted(map(str, nums), key=LargerNumKey))
```