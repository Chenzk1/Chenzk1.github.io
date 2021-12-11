---
title: C++-自建数组及STL中的array&vector
date: 2021-12-10
categories: 
    - Learning
tags:  
    - C++
---

## 相同点

- 三者皆使用连续内存，且array和vector的底层由自建数组实现
- 都可以用下标访问

<!-- more -->

## 不同点

1. vector属于变长容器，即可以根据数据的插入删除重新构建容器容量；但array和数组属于定长容器。
2. vector提供了可以动态插入和删除元素的机制，而array和数组则无法做到，或者说array和数组需要完成该功能则需要自己实现完成。
3. 由于vector的动态内存变化的机制，**在插入和删除时，需要考虑迭代的是否失效的问题**，vector容器增长时，不是在原有内存空间中添加，而是重新申请+分配内存。
4. vector和array提供了更好的数据访问机制，即可以使用front和back以及at访问方式，使得访问更加安全。而数组只能通过下标访问，在程序的设计过程中，更容易引发访问错误。
5. vector和array提供了更好的遍历机制，有正向迭代器和反向迭代器两种
6. vector和array提供了size和判空的获取机制，而数组只能通过遍历或者通过额外的变量记录数组的size。
7. vector和array提供了两个容器对象的内容交换，即swap的机制，而数组对于交换只能通过遍历的方式，逐个元素交换的方式使用
8. array提供了初始化所有成员的方法fill

## 自建数组

### 静态数组&动态数组

#### 静态数组

- **在栈中**
- 在编译期间在栈中分配好内存的数组，在运行期间不能改变存储空间，运行后由**系统自动释放**。
- 必须用常量指定数组大小
- 数组名为第一个数组元素

#### 动态数组

- 程序运行时才分配内存的数组
- **在堆中**
- 数组名为指向第一个数组元素的指针

### 初始化

- int a[3]; 局部作用域中，只做了声明，未初始化，其值未定
- int a[3]; 静态数组以及直接声明在某个namespace中（函数外部），默认初始化为全0
- int a[3] = {}; 初始化为0
- int a[3] = {1}; 初始化为{1, 0, 0}
- int a[] = {1,2,3,4}; 声明大小为4的数组，并初始化为{1,2,3,4}
- int a[] {1,2,3,4}; **通用初始化**方法，声明和初始化程序之间不用等号
- int* a = new int[10]; //new 分配了一个大小为10的未初始化的int数组，并返回指向该数组第一个元素的指针，此指针初始化了指针a
  - a = {4, -3, 5, -2, -1, 2, 6, -2, 3}; // 错误，注意这种用大括号的数组赋值只能用在声明时，此处已经不是声明，所以出错。
  - int *a = new int[10] ();  // 默认初始化，每个元素初始化为0,括号内不能写其他值，只能初始化为0
  - int* a = new int[n];// t
  - string* Dynamic_Arr4 = new string[size]{"aa", "bb","cc", "dd", string(2, 'e') };      //显式的初始化
  - delete [ ] Dynamic_Arr4；//动态数组的释放
- 维度为变量时，不能在声明时同时初始化，即int a[b]={1};是不合法的。 
- 多维数组为1维数组的特殊形式，定义多维数组时，如果不对它进行初始化，必须标明每个维度的大小；如果进行了显式的初始化，可以不标明最高维度的大小，（也就是第一个维度，当第一个维度不标明大小，则不需进行初始化）

## array

- [ref](https://www.cplusplus.com/reference/array/array/)
- 大小固定

## vector

- [ref](https://www.cplusplus.com/reference/vector/vector/)
- 动态数组
  
### 初始化

- 构造函数
```C++
vector():创建一个空vector
vector(int nSize):创建一个vector,元素个数为nSize
vector(int nSize,const t& t):创建一个vector，元素个数为nSize,且值均为t
vector(const vector&):复制构造函数
vector(begin,end):复制[begin,end)区间内另一个数组的元素到vector中
```
``` C++
  std::vector<int> first;                                // empty vector of ints
  std::vector<int> second (4,100);                       // four ints with value 100
  std::vector<int> third (second.begin(),second.end());  // iterating through second
  std::vector<int> fourth (third);                       // a copy of third

  // the iterator constructor can also be used to construct from arrays:
  int myints[] = {16,2,77,29};
  std::vector<int> fifth (myints, myints + sizeof(myints) / sizeof(int) );
```
  - 二维vector若要使用A[i].push_back(3)等方式初始化，则应事先定义好第一维的大小，例如
  ```C++
  vector<vector<int>> ans(5);
  ans[3].push_back(3);
  ```

### 增加

```C++
void push_back(const T& x):向量尾部增加一个元素X
iterator insert(iterator it,const T& x):向量中迭代器指向元素前增加一个元素x
iterator insert(iterator it,int n,const T& x):向量中迭代器指向元素前增加n个相同的元素x
iterator insert(iterator it,const_iterator first,const_iterator last):向量中迭代器指向元素前插入另一个相同类型向量的[first,last)间的数据
```

### 删除

```C++
iterator erase(iterator it):删除向量中迭代器指向元素
iterator erase(iterator first,iterator last):删除向量中[first,last)中元素
void pop_back():删除向量中最后一个元素
void clear():清空向量中所有元素
```

### 遍历

```C++
reference at(int pos):返回pos位置元素的引用
reference front():返回首元素的引用
reference back():返回尾元素的引用
iterator begin():返回向量头指针，指向第一个元素
iterator end():返回向量尾指针，指向向量最后一个元素的下一个位置
reverse_iterator rbegin():反向迭代器，指向最后一个元素
reverse_iterator rend():反向迭代器，指向第一个元素之前的位置
```

### 判断

```C++
bool empty() const:判断向量是否为空，若为空，则向量中无元素
```

### 大小

```C++
int size() const:返回向量中元素的个数
int capacity() const:返回当前向量所能容纳的最大元素值
int max_size() const:返回最大可允许的vector元素数量值
```

### 其他

```C++
void swap(vector&):交换两个同类型向量的数据
void assign(int n,const T& x):设置向量中第n个元素的值为x
void assign(const_iterator first,const_iterator last):向量中[first,last)中元素设置成当前向量元素
```