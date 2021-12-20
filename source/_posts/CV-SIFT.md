---
title: CV-SIFT
date: 2021-12-10
categories: 
    - Learning
tags:  
    - ML
    - CV
    - SIFT
mathjax: true
---
<meta name="referrer" content="no-referrer"/>

[TOC]

## SIFT(Scale-invariant feature transform)综述

- 尺度不变特征转换：SIFT是一种在**多尺度空间上提取局部特征**的方法。它在空间尺度中寻找极值点，并提取出其位置、尺度、旋转不变量，此算法由 David Lowe在1999年所发表，2004年完善总结。

<!-- more -->

### 优点

* SIFT特征是图像的局部特征，其对旋转、尺度缩放、亮度变化保持不变性，对视角变化、仿射变换、噪声也保持一定程度的稳定性；
* 独特性（Distinctiveness）好，信息量丰富，适用于在海量特征数据库中进行快速、准确的匹配；
* 多量性，即使少数的几个物体也可以产生大量的SIFT特征向量；
* 高速性，经优化的SIFT匹配算法甚至可以达到实时的要求；
* 可扩展性，可以很方便的与其他形式的特征向量进行联合。

### 缺点

* 实时性不高
* 有时特征点数量较少
* 对边缘光滑的目标无法准确提取特征点
  
### 整体流程

1. 建立尺度空间，即DoG金字塔
2. 关键点检测及定位，获得每个关键点的：**尺度、位置**
3. 特征点方向赋值，获得每个特征点的：**方向**
4. 关键点特征描述
   
![整体流程](https://img2018.cnblogs.com/blog/1471528/201903/1471528-20190330204125547-535493662.png)

## 建立DoG金字塔

* 获取多尺度空间：Lindeberg等人已证明高斯卷积核是实现尺度变换的唯一变换核，并且是唯一的线性核

### 预备知识

#### 二维高斯函数

- 以(m/2, n/2)为中心的二维高斯分布
$$
G(x, y)=\frac{1}{2 \pi \sigma^{2}} e^{-\frac{(x-m / 2)^{2}+(y-n / 2)^{2}}{2 \sigma^{2}}}
$$
- 生成的曲面等高线是从中心开始呈正态分布的同心圆，权重值从中心向周围越来越小。这样进行模糊处理比其它的均衡模糊滤波器更高地保留了边缘效果。
- 实际应用中，在计算高斯函数的离散近似时，在大概3σ距离之外的像素都可以看作不起作用，因此图像处理程序只需要计算$
(6 \sigma+1) \times(6 \sigma+1)
$的矩阵

#### 二维高斯函数的优化

1. 若高斯模板大小为m\*n，当图像大小为M\*N时，计算复杂度为O(m\*n\*M\*N)
2. $\sigma$越大，高斯模板分散性越强，处理边缘点时，丢失的图像信息越多，可能会造成黑边

- 可以证明，二维高斯运算可以优化为：水平方向进行一维高斯矩阵变换加上竖直方向的一维高斯矩阵变换得到。时间复杂度为：$
O(n \times M \times N)+O(m \times M \times N)
$。[证明](https://blog.csdn.net/lwx309025167/article/details/82761474)
![证明](https://img-blog.csdn.net/20180919111255142?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x3eDMwOTAyNTE2Nw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

#### 高斯拉普拉斯算子(Laplacion of Gaussian, LoG)

- 作用：突出边缘，更清晰，对比度更强
- 拉普拉斯：拉普拉斯算子是图像二阶空间导数的二维各向同性测度。**拉普拉斯算子可以突出图像中强度发生快速变化的区域**，因此常用在边缘检测任务当中。
- 高斯：在进行Laplacian操作之前通常需要先用高斯平滑滤波器对图像进行平滑处理，以**降低Laplacian操作对于噪声的敏感性**。
- 输入输出均为灰度图
- 连续拉普拉斯：
$$
L(x, y)=\frac{\partial^{2} I}{\partial x^{2}}+\frac{\partial^{2} I}{\partial y^{2}}
$$
- 离散拉普拉斯，例如：

| 0 | -1 | 0 |
| -----| ---- | ---- |
| -1 | 4 | -1 |
| 0 | -1 | -1 |

| -1 | -1 | -1 |
| -----| ---- | ---- |
| -1 | 8 | -1 |
| -1 | -1 | -1 |
- 表达：以0为中心，$\sigma$为标准差的LoG
$$
L o G(x, y)=-\frac{1}{\pi \sigma^{4}}\left[1-\frac{x^{2}+y^{2}}{2 \sigma^{2}}\right] e^{-\frac{x^{2}+y^{2}}{2 \sigma^{2}}}
$$
![LoG示意图](https://pic1.zhimg.com/80/v2-f824bd5eae07235ebf531fa0b546ba98_1440w.jpg)
> 推导过程：![推导过程](https://pic4.zhimg.com/80/v2-b220086b94625a8a00ce68eb2a4bd0e3_1440w.jpg)
- 示例：一维LoG滤波器对于边缘的响应：![一维LoG滤波器对于边缘的响应](https://pic4.zhimg.com/80/v2-12ce9caee0278bf8855efdecfcba7fc7_1440w.jpg)

#### 高斯差分算子(Difference of Gaussian, DOG)

- 可以近似LoG算子，减少运算量，由Lindeberg在1994证明
- DoG的运算如下，即DoG为两不同尺度的高斯算子平滑后的图像之差，具体证明见：[证明](https://blog.csdn.net/kieven2008/article/details/104309440)
$$
\begin{aligned}
D(x, y, \sigma) &=(G(x, y, k \sigma)-G(x, y, \sigma)) * I(x, y) \\
&=L(x, y, k \sigma)-L(x, y, \sigma)
\end{aligned}
$$

#### 尺度空间理论

- 定义
  - 单尺度 ---> 多尺度
  - 尺度空间中各尺度图像的模糊程度逐渐变大，能够模拟人在距离目标由近到远时目标在视网膜上的形成过程。
- 要求：满足视觉不变性
  - 亮度/灰度不变性，对比度不变性
  - 平移不变性、尺度不变性、欧几里得不变性、仿射不变性
- 方法：
  - Tony Lindeberg指出**尺度规范化的LoG(Laplacion of Gaussian)算子具有真正的尺度不变性**。Lowe使用高斯差分金字塔近似LoG算子，是尺度空间检测稳定的关键点。
- 表示：一个图像的尺度空间$L(x,y,\sigma)$，可以表示为一个变化尺度的高斯函数与原图像的卷积
$$
L(x, y, \sigma)=G(x, y, \sigma) * I(x, y)
$$
$$
G(x, y, \sigma)=\frac{1}{2 \pi \sigma^{2}} e^{-\frac{(x-m / 2)^{2}+(y-n / 2)^{2}}{2 \sigma^{2}}}
$$
  - m，n表示高斯模板中心，维度为$
(6 \sigma+1) \times(6 \sigma+1)
$，(x, y)代表图像的像素位置。$\sigma$是尺度空间因子，值越小表示图像被平滑的越少，相应的尺度也就越小。**大尺度对应于图像的概貌特征，小尺度对应于图像的细节特征。**

### 建立DoG金字塔

- 极值：图像域和尺度域上的极值
![步骤](https://img2018.cnblogs.com/blog/1471528/201903/1471528-20190330204227104-18141123.png)

#### 构建高斯金字塔

1. 先将原图像扩大一倍之后作为高斯金字塔的第1组第1层，将第1组第1层图像经高斯卷积（其实就是高斯平滑或称高斯滤波）之后作为第1组金字塔的第2层。对于参数σ，在Sift算子中取的是固定值1.6。
2. 将σ乘以一个比例系数k，得到一个新的平滑因子σ=k*σ，用它来平滑第1组第2层图像，结果图像作为第3层。
3. 如此这般重复，最后得到L层图像，在同一组中，每一层图像的尺寸都是一样的，只是平滑系数不一样。它们对应的平滑系数分别为：0，σ，kσ，k^2σ,k^3σ……k^(L-2)σ。
4.  将第1组**倒数第三层**图像作比例因子为2的降采样，得到的图像作为第2组的第1层，然后对第2组的第1层图像做平滑因子为σ的高斯平滑，得到第2组的第2层，就像步骤2中一样，如此得到第2组的L层图像，同组内它们的尺寸是一样的，对应的平滑系数分别为：0，σ，kσ，k^2σ,k^3σ……k^(L-2)σ。但是在尺寸方面第2组是第1组图像的一半。
5. 反复执行，得到一共O组，每组L层，共计O*L个图像，这些图像一起就构成了高斯金字塔，结构如下：
![高斯金字塔](https://img-blog.csdn.net/20160917212318336)

#### 构建高斯差分(DOG)金字塔

- 方法
![高斯差分金字塔](https://img-blog.csdn.net/20160917223500317)
- 经过归一化的高斯差分金字塔
![一个示例](https://img-blog.csdn.net/20160917232151713)

#### 具体构建过程中的参数计算

- 极值点：每一个像素点要和它所有的相邻点比较，看其是否比它的图像域和尺度域的相邻点大或者小。要获取S个尺度的极值点--->需要S+2个DOG空间-->S+3层高斯金字塔
- 实际计算时S在3到5之间
- 输入：组数o，层数S，$k=2^{\frac{1}{S}}$，以及计算每组每层的尺度参数$\sigma(o, s)=\sigma_{0} 2^{\circ+\frac{s}{S}} o \in[0, \ldots, O-1], s \in[0, \ldots, S+2]$
  - $\sigma_{0}$为初始尺度
- 相同组间不同层的尺度参数$\sigma$为k倍之差，不同组相同层之间尺度参数$\sigma$为2倍之差
  
## 关键点定位

### 离散极值点检测

- DoG金字塔中，每个点与周围8+前后两个尺度9*2=26个点比较，来确定极值点

### 关键点（连续极值点）定位

- 动机
  - 获得连续空间的极值点（位置、尺度、极值大小），并去除低对比度的关键点
  - 去除不稳定的边缘响应点（DoG算子会产生较强的边缘响应），增强匹配稳定性、提高降噪能力

#### 关键点定位

- 离散空间的极值点只是局部区域
![离散空间极值点与连续空间极值点的区别](https://img2018.cnblogs.com/blog/1471528/201903/1471528-20190330205035199-488432412.jpg)
- 为了获得连续空间的极值点，需要对离散空间DoG函数进行曲线插值/拟合。拟合的方法是利用DoG函数在尺度空间上的泰勒展开。
1. 在任意一个坐标为$X_0=(x_0,y_0,\sigma_0)$的极值点处做DoG函数的泰勒展开，舍掉2阶以后的项，结果如下：
$$
f\left(\left[\begin{array}{l}
x \\
y \\
\sigma
\end{array}\right]\right) \approx f\left(\left[\begin{array}{l}
x_{0} \\
y_{0} \\
\sigma_{0}
\end{array}\right]\right) \mid+\left[\begin{array}{lll}
\frac{\partial f}{\partial x} & \frac{\partial f}{\partial y} & \frac{\partial f}{\partial \sigma}
\end{array}\right]\left(\left[\begin{array}{l}
x \\
y \\
\sigma
\end{array}\right]-\left[\begin{array}{l}
x_{0} \\
y_{0} \\
\sigma_{0}
\end{array}\right]\right) + 
\frac{1}{2}\left(\left[\begin{array}{llll}
x & y & \sigma
\end{array}\right]-\left[\begin{array}{lll}
x_{0} & y_{0} & \sigma_{0}
\end{array}\right]\right)\left[\begin{array}{lll}
\frac{\partial^{2} f}{\partial x \partial x} & \frac{\partial^{2} f}{\partial x \partial y} & \frac{\partial^{2} f}{\partial x \partial \sigma} \\
\frac{\partial^{2} f}{\partial x \partial y} & \frac{\partial^{2} f}{\partial y \partial y} & \frac{\partial^{2} f}{\partial y \partial \sigma} \\
\frac{\partial^{2} f}{\partial x \partial \sigma} & \frac{\partial^{2} f}{\partial y \partial \sigma} & \frac{\partial^{2} f}{\partial \sigma \partial \sigma}
\end{array}\right]\left(\left[\begin{array}{l}
x \\
y \\
\sigma
\end{array}\right]-\left[\begin{array}{l}
x_{0} \\
y_{0} \\
\sigma_{0}
\end{array}\right]\right)
$$
其向量形式如下：
$$
D(X)=D+\frac{\partial D^{T}}{\partial X} X+\frac{1}{2} X^{T} \frac{\partial^{2} D}{\partial X^{2}} X
$$
2. 求导并让方程==0，求得极值点相对$X_0=(x_0,y_0,\sigma_0)$的偏移量为$\hat{X}=-\frac{\partial^{2} D^{-1}}{\partial X^{2}} \frac{\partial D}{\partial X}$。
3. 当此偏移量在任一维度（x, y, $\sigma$）上大于0.5时，意味着此极值点已经去临近的点上了，因此改变该极值点位置。
4. 不断迭代1~3。终止条件：收敛/超出迭代次数（Lowe设定为5）/超出图像边界（此时应删除该点）。记录所有点的位置（原位置+偏移量）以及尺度$\sigma(o, s)$.
5. 为加强抗噪声的能力，删除$|D(x)|$过小的点（Lowe论文中使用0.03，Rob Hess等人实现时使用0.04)。

#### 边缘效应消除

- 一个定义不好的DoG算子的极值在横跨边缘的地方有较大的主曲率，而在垂直边缘的方向有较小的主曲率。即**两个主曲率的比值越大**。
  - 主曲率：曲面的每个方向都有法曲率，那么就有最大最小的法曲率，这个最大最小值就是主曲率，对应的曲线在这点的切线方向就是主曲率方向。这两个方向是垂直的。
1. 获取特征点处的Hessian矩阵，主曲率通过一个2x2的Hessian矩阵H求出：
$$
H=\left[\begin{array}{ll}
D_{x x} & D_{x y} \\
D_{xy} & D_{yy}
\end{array}\right]
$$
2. H的特征值$\alpha$和$\beta$代表了x方向和y方向的梯度。
   1. 先求出H的对角线元素之和以及H的行列式
   $$
   \begin{array}{l}
   \operatorname{Tr}(H)=D_{x x}+D_{x}=\alpha+\beta \\
   \operatorname{Det}(H)=D_{x x} D_{y}-\left(D_{x}\right)^{2}=\alpha \beta
   \end{array}
   $$
   2. 设$\alpha$较大，令$\alpha=r \beta$，则$
\frac{T r(H)^{2}}{D e t(H)}=\frac{(\alpha+\beta)^{2}}{\alpha \beta}=\frac{(r \beta+\beta)^{2}}{r \beta^{2}}=\frac{(r+1)^{2}}{r}
$
   3. r值越大，说明两个特征值的比值越大，即在某一个方向的梯度值越大，而在另一个方向的梯度值越小，而边缘恰恰就是这种情况。所以为了剔除边缘响应点，需要让r小于一定的阈值，因此，为了检测主曲率是否在某域值r下，只需检测$\frac{\operatorname{Tr}(H)^{2}}{\operatorname{Det}(H)}<\frac{(r+1)^{2}}{r}$，若此式子成立，保留关键点，反之剔除。
      1. Lowe取r=10.

#### 其他

- 以上的关键点定位过程中用到了离散的导数，具体求导时应用了[有限差分法](https://blog.csdn.net/qq_41679006/article/details/80975436)求导
- 以上过程中需要用到三阶矩阵求逆，可逆矩阵A及其逆如下
$$
A=\left(\begin{array}{lll}
a_{00} & a_{01} & a_{02} \\
a_{10} & a_{11} & a_{12} \\
a_{20} & a_{21} & a_{22}
\end{array}\right)
$$
$$
A^{-1}=\frac{1}{|A|}\left(\begin{array}{ccc}
a_{11} a_{22}-a_{21} a_{12} & -\left(a_{01} a_{22}-a_{21} a_{02}\right) & a_{01} a_{12}-a_{02} a_{11} \\
a_{12} a_{20}-a_{22} a_{10} & -\left(a_{02} a_{20}-a_{22} a_{00}\right) & a_{02} a_{10}-a_{00} a_{12} \\
a_{10} a_{21}-a_{20} a_{11} & -\left(a_{00} a_{21}-a_{20} a_{01}\right) & a_{00} a_{11}-a_{01} a_{10}
\end{array}\right)
$$

## 关键点方向匹配

- 为了使最后所得的描述符具有旋转不变性，需要利用图像的局部特征为给每一个关键点分配一个基准方向。使用图像梯度的方法求取局部结构的稳定方向。
1. 对于在DOG金字塔中检测出的关键点点，采集其所在高斯金字塔图像3σ邻域窗口内像素的梯度和方向分布特征。梯度的模值和方向如下：
$$
\begin{array}{l}
m(x, y)=\sqrt{(L(x+1, y)-L(x-1, y))^{2}+(L(x, y+1)-L(x, y-1))^{2}} \\
\left.\theta(x, y)=\tan ^{-1}((L(x, y+1)-L(x, y-1)) / L(x+1, y)-L(x-1, y))\right)
\end{array}
$$
L为关键点所在的尺度空间值，按Lowe的建议，梯度的模值m(x,y)按$1.5\sigma(o,s)$的尺度进行高斯加权，按照常用的$3\sigma$原则，取邻域窗口的半径为$3 \times 1.5 \sigma(o,s)$
1. 求得关键点及其邻域内像素的梯度和方向，并使用直方图进行统计。梯度直方图将0~360度的方向范围分为36个柱(bins)，其中每柱10度。![如图](https://img2018.cnblogs.com/blog/1471528/201903/1471528-20190330205657098-53794026.png)
2. 方向直方图的峰值则代表了该特征点处邻域梯度的方向，以直方图中最大值作为该关键点的主方向。为了增强匹配的鲁棒性，只保留峰值大于主方向峰值80％的方向作为该关键点的辅方向。因此，**对于同一梯度值的多个峰值的关键点位置，在相同位置和尺度将会有多个关键点被创建但方向不同**。仅有15％的关键点被赋予多个方向，但可以明显的提高关键点匹配的稳定性。实际编程实现中，就是把该关键点复制成多份关键点，并将方向值分别赋给这些复制后的关键点，并且，离散的梯度方向直方图要进行插值拟合处理，来求得更精确的方向角度值。
   1. 梯度直方图的平滑处理：为了避免梯度方向受噪声的影响，还可以对梯度直方图进行平滑以及进行抛物线插值处理。[具体方法](https://www.cnblogs.com/Alliswell-WP/p/SIFT.html)。

## 特征描述子计算

- 作用：**表示关键点邻域高斯图像梯度统计结果**。
- 方法：通过对关键点周围图像区域分块，计算块内梯度直方图，生成具有独特性的向量，这个向量是该区域图像信息的一种抽象，具有唯一性&独特性。
  - Lowe建议描述子使用在关键点尺度空间内4\*4的窗口中计算的8个方向的梯度信息，共4\*4\*8=128维向量表征。

### 确定计算描述子所需的图像区域

1. 将关键点附近的邻域划分为d*d(Lowe建议d=4)个子区域，每个子区域做为一个种子点，每个种子点有8个方向。
2. 每个子区域的大小，也使用$3\sigma$原则确定，即子区域边长为$3\sigma$。
3. 则所需图像区域边长为$3\sigma\times(d+1)$
4. 考虑到旋转因素(方便下一步将坐标轴旋转到关键点的方向)，用圆代替矩阵，实际计算所需的图像区域半径为：
$radius=\frac{3\sigma\times\sqrt{2}\times(d+1)}{2}$。计算结果四舍五入取整。![](https://img2018.cnblogs.com/blog/1471528/201903/1471528-20190330210025431-395170345.jpg)

### 坐标轴旋转至主方向

- 坐标轴旋转
![坐标轴旋转至主方向](https://img2018.cnblogs.com/blog/1471528/201903/1471528-20190330210107953-2078955000.png
)
- 旋转后邻域内采样点的新坐标：
$$
\left(\begin{array}{l}
x^{\prime} \\
y^{\prime}
\end{array}\right)=\left(\begin{array}{ll}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{array}\right)\left(\begin{array}{l}
x \\
y
\end{array}\right)(x, y \in[-\text {radius, radius}]
$$

### 梯度直方图生成

- 如下图绿色部分的坐标系中，旋转后采样点$(x^{\prime}, y^{\prime})$的新坐标为
$$
\left(\begin{array}{l}
x^{n} \\
y^{n}
\end{array}\right)=\frac{1}{3 \sigma_{-} o c t}\left(\begin{array}{l}
x^{\prime} \\
y^{t}
\end{array}\right)+\frac{d}{2}
$$
- Lowe建议子区域的像素的梯度大小按$\sigma=0.5d$进行高斯加权，(a, b)为关键点在DoG图像中的位置坐标，则
$$
w=m(a+x, b+y)^{*} e^{-\frac{\left(x^{\prime}\right)^{2}+\left(y^{\prime}\right)^{2}}{2 \times(0.5 d)^{2}}}
$$
![](https://niecongchong.github.io/img/2019-08-06-24.jpg)
- 与求主方向不同，此时每个种子区域的梯度直方图在0-360之间划分为8个方向区间，每个区间为45度，即每个种子点有8个方向的梯度强度信息。所以共4\*4\*8=128个梯度。

### 三线性插值

- 三线性：x, y, 方向
![](https://img2018.cnblogs.com/blog/1471528/201903/1471528-20190330210313666-1453481217.png)
- 如图中的红色点，落在第0行和第1行之间，对这两行都有贡献。对第0行第3列种子点的贡献因子为dr，对第1行第3列的贡献因子为1-dr，同理，对邻近两列的贡献因子为dc和1-dc，对邻近两个方向的贡献因子为do和1-do。k,m,n为0或1，则最终累加在每个方向上的梯度大小为：
$$
\text { weight }=w^{*} d r^{k} *(1-d r)^{1-k} * d c^{m *}(1-d c)^{1-m *} d o^{n *}(1-d o)^{1-n}
$$

### 特征描述子以及归一化特征描述子

- 如上统计的4\*4\*8=128个梯度信息即为该关键点的特征向量。特征向量形成后，为了去除光照变化的影响，需要对它们进行归一化处理，对于图像灰度值整体漂移，图像各点的梯度是邻域像素相减得到，所以也能去除。
- 得到的描述子向量为$H=\left(h_{1}, h_{2}, \ldots, h_{128}\right)$
- 归一化后的描述子向量为$L=\left(l_{1}, l_{2}, \ldots, l_{128}\right)$

### 特征描述子门限化

- 非线性光照，相机饱和度变化对造成某些方向的梯度值过大，而对方向的影响微弱。因此设置门限值(向量归一化后，一般取0.2)截断较大的梯度值。然后，再进行一次归一化处理，提高特征的鉴别性。

### 特征描述向量排序

- 按特征点的尺度对特征描述向量进行排序。
  
## reference

- https://www.cnblogs.com/Alliswell-WP/p/SIFT.html
- https://blog.csdn.net/zddblog/article/details/7521424
- https://blog.csdn.net/qq_41679006/article/details/80975436