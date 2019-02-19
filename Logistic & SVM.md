
# 逻辑回归(logistic regression) 与支持向量机(SVM)

作者：白露佳 

本文在翻译 *Computer Age Statistical Inference: Algorithms, Evidence, and Data Science*的同时，加入了自己的例子和思考，主要介绍了逻辑回归与支持向量机方法的联系以及支持向量机的原理，包括硬间隔，软间隔分类器以及核方法等等，理解上需要一定逻辑回归以及运筹学知识，有助于对机器学习方法和传统统计方法的理解。

### 1 步骤
用逻辑回归进行分类主要分为两步:  
- 拟合条件概率 $Pr(Y = 1 | X = x)$
- 如果 $Pr(Y = 1 | X = x) \geq 0.5$ 将Y分类为1  

支持向量机则没有第一步，直接进行分类

## 2 逻辑回归的局限性

逻辑回归面临一个尴尬的问题： 如果训练数据线性可分, 逻辑回归将无法进行分类。
“训练数据线性可分”是指，在特征空间（X的空间），两个类别的数据点可以由一条直线分隔开。在这种情况下，
极大似然估计会失效：一些参数会趋向于正无穷。  
***
如: 

|X\Y|Y = 1|Y = 0 |
|----|----|----|
|X = 0|30|0|
|X = 1|0|50|


拟合条件概率 $Pr(Y = 1 | X = 0)$:  
模型为 $$\log \frac{p_0}{1-p_0} = \alpha_0 + \alpha X$$
似然函数 $$L\propto p_0^{n_{11}}(1-p_0)^{n_{12}}$$
其中，$n_{ij}$为 第$i$行,第$j$列的频数，$p_i$为$X = i$时 $Y = 1$的概率。  

$p_0$的极大似然估计$$\hat{p}_0 =  \frac{n_{11}}{n_{11}+n_{12}} = 1$$
此时优比$ \frac{p_0}{1-p_0} \to \infty $无意义，无法进行估计。

***
这样的情形看似不可能，但是在处理“宽”的基因数据的时候几乎一定会发生。当$p >> n $，即特征的维数远大于数据量，我们总能找到这样的分割超平面。找到一个最优的分割超平面真是支持向量机（SVM）的出发点

## 3 SVM 中的统计思想

SVM  实际上遵循着统计上一个传统的方法，通过**非线性变换（nonlinear transformation)** 和 **基的展开（basis expansion）** 来扩充特征空间。  
这样的方法有一个经典的例子——在线性回归中加入**交互项**，这个在扩大的空间中的线性模型是原来空间中的非线性模型。
为了扩充特征空间，我们一般使用**“核技巧”（kernel trick）**。这样无论解释变量的维数p有多大，我们都能在n维空间中进行计算。而“核技巧”，可以证明，等价于在**再生成希尔伯特空间(reproducing Hilbert space) **上进行估计。



## 4 SVM的原理

###  4.1最优分割超平面（optimal separating hyperplane)


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from matplotlib.lines import Line2D

fig = plt.figure(num=None, figsize=(13, 6), dpi=80, facecolor='w', edgecolor='k')
# we create 40 separable points
X, y = make_blobs(n_samples=40, centers=2, random_state=6)

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)



fig.add_subplot(122)
plt.scatter(X[:, 0], X[:, 1], c = y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')

ax0 = fig.add_subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
# 两条line的数据
line1 = [(4, -4), (11, -9)]
line2 = [(4, -7), (11, -5)]
line3 = [(4, -5), (11, -7)]
(line1_xs, line1_ys) = zip(*line1)
(line2_xs, line2_ys) = zip(*line2)
(line3_xs, line3_ys) = zip(*line3)

# 创建两条线，并添加
ax0.add_line(Line2D(line1_xs, line1_ys, linewidth=1,color = "black"))
ax0.add_line(Line2D(line2_xs, line2_ys, linewidth=1,color = "black"))
ax0.add_line(Line2D(line3_xs, line3_ys, linewidth=1,color = "black"))

```




    <matplotlib.lines.Line2D at 0x1a2319e048>




![png](output_10_1.png)


**Figure1** 展示了$\mathbb{R}^2$上的一些样本点，每一个点属于两类中的一类（蓝色或者棕色）。我们可以定义如果点为蓝色时$y = +1$,棕色时$y = -1$。这样我们可以建立二分类分类器：$f(x) = \beta_0 + x' \beta$ —— $f(x_0)>0$时 将$x_0$分类为+1，$f(x_0)<0$,将$x_0$分类为-1,在$f(x_0) = 0$我们随机分类（如抛硬币决定）。分类器也可写为：$C(x) = sign[f(x)]$，决策边界为集合${x|f(x) = 0}$。Figure1左图中，三条直线都可以将不同类的数据点分开。
**最优的分割超平面**是使两类数据之间的**间隔（margin）**最大的线性分类器。其中隐含的假设是，训练数据的间隔越大，分类器对未来观测数据的分类效果越好。


通过简单的初等几何知识，我们可以得到数据点$x_0$到线性决策边界的欧几里得距离（带符号）为
$$\frac{1}{||\beta||_2}f(x_0)$$
故$\frac{1}{||\beta||_2}y_if(x_i)$可以表示数据点$x_i$到决策边界的距离，其中 $y_if(x_i)$称为**间隔（margin)**。于是我们可以把寻找最优的分割超平面的问题写成一个优化问题：$$\max_{\beta_0,\beta}\frac{1}{||\beta||_2}y_if(x_i)$$
即
\begin{align}
&\max_{\beta_0,\beta} M\\
& s.t. \frac{1}{||\beta||_2}y_i(\beta_0+x_i'\beta) \geq M      &&\quad i=1...n\\
\end{align}

可以化简为正则形式(令$M = \frac{1}{||\beta||_2}$)
\begin{align}
&\min_{\beta_0,\beta} ||\beta||_2\\
& s.t. y_i(\beta_0+x_i'\beta )\geq 1      &&\quad i=1...n\\
\end{align}


可以写出该凸优化问题的KKT条件:  
$$
\begin{align}
&y_i(\beta_0+x_i'\beta )\geq 1   &&\quad i=1...n  &&\quad(1)\\
&\lambda_i (1 - y_i(\beta_0+x_i'\beta ) ) = 0 &&\quad i=1...n&&\quad(2)\\
&\nabla ||\beta||^2 - \nabla \sum_i \lambda_i y_i(\beta_0+x_i'\beta ) = 0&&\quad &&\quad(3)\\
&\lambda_i \geq 0 &&\quad i=1...n&&\quad(4)\\
\end{align}
$$

由(3) 可以写出 $\beta = \frac{1}{2} \lambda_i  y_i x_i \triangleq \alpha_i x_i$
由(2)(4)可以知道在$y_i(\beta_0+x_i'\beta ) - 1= 0$（决策边界）处$\lambda_i>0$才成立。
我们定义决策边界上的点为**支持向量(support vector)**，对应的下标集合为**支撑集(support set)**。于是我们可以写出$\beta$的估计与支持向量的关系
$$\hat{\beta} = \sum_{i\in S}\hat{\alpha}_i x_i$$
其中$S$为支撑集(support set)



### 4.2 软间隔分类器（soft margin classifier）
如Figure2所示，当数据不可分，我们将之前的分类器推广成为软间隔分类器，允许数据点违反不等式条件$y_i(\beta_0+x_i'\beta )\geq 1$ 软间隔分类器可以转化为解
\begin{align}
&\max_{\beta_0,\beta} ||\beta||_2\\
& s.t. y_i(\beta_0+x_i'\beta )\geq 1-\epsilon_i,\\
&\epsilon_i \geq 0,i=1...n,and \sum_{i=1}^n \epsilon_i \leq B\\
\end{align}



$B$控制了违反不等式条件的程度，错误分类数据的总距离。同样我们可以写出$\beta$的估计与支持向量的关系
$$\hat{\beta} = \sum_{i\in S}\hat{\alpha}_i x_i$$
其中$S$为支撑集(support set),不仅包括了决策边界上的向量，还包括了违反间隔条件的向量。$B$越大支撑集就越大，解集中的点越多，意味着稳定性越高，方差更低。实际上，对于线性可分的数据我们也可以通过调节B的参数大小，得到正则形式的解（$B$ = 0)。


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# we create 40 separable points
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [1] * 19 + [0] * 20+ [1] * 1

# figure number

fig = plt.figure(num=2, figsize=(13,6), dpi=80, facecolor='w', edgecolor='k')
fignum = 1

# fit the model
for name, penalty in (('unreg', 1), ('reg', 0.05)):
    
    
    clf = svm.SVC(kernel='linear', C=penalty)
    clf.fit(X, Y)

    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
    # 2-d.
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    # plot the line, the points, and the nearest vectors to the plane
    fig.add_subplot(1,2,fignum)
    
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)

    plt.axis('tight')
    x_min = -4.8
    x_max = 4.2
    y_min = -6
    y_max = 6

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    #plt.figure(fignum, figsize=(4, 3))
    #plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    
    fignum = fignum + 1

plt.show()
```


![png](output_16_0.png)


**Figure 2**: 软间隔分类器，右图B值比左图的大

## 4.3 SVM 与损失函数+惩罚项的关系

上一部分中软间隔分类器可以写为:
\begin{align}
&\min_{\beta_0,\beta} ||\beta||_2\\
& s.t. y_i(\beta_0+x_i'\beta )\geq 1-\epsilon_i,\\
&\epsilon_i \geq 0,i=1...n,and \sum_{i=1}^n \epsilon_i \leq B\\
\end{align}


可写为
$$
\min{\beta_0,\beta}  \sum_{i=1}^n [1-y_i(\beta_0+x_i'\beta )]_{+} +  \lambda||\beta||_2^2     \tag{A}
$$


这里** 铰链损失（hinge loss）** $L_H(y,f(x)) = [1-y_i(\beta_0+x_i'\beta )]_{+} $ 是$x_i$分类错误的
成本（cost），如果分类正确则为0。大的$\lambda$值对应着大的$B$值。对于线性可分数据，4.1最优分割超平面的解对应着$\lambda\to 0$的解

逻辑回归的deviance可以写成间隔的函数

$$ L_B(y,f(x)) = -{I(y=-1)\log Pr(y=-1|x)+ I(y=+1)\log Pr(y=+1|x)} $$

其中$$f(x) = \log \frac{Pr(y=+1|x)}{Pr(y=-1|x)}$$  
于是可以得到$$Pr(y=+1|x) = \frac{e^{f(x)}}{e^{f(x)}+e^{-f(x)}},$$ $$Pr(y=-1|x) = \frac{e^{-f(x)}}{e^{f(x)}+e^{-f(x)}}$$
因此 
$$L_B(y,f(x)) = \log [1+e^{-yf(x)}]$$


**岭逻辑回归（ridged logistic regression）** （deviance+惩罚项）可以写成相同的形式:  

$$
\min{\beta_0,\beta} \sum_{i=1}^n \log[1+e^{-y_i(\beta_0+x_i^{'}\beta)}]+  \lambda||\beta||_2^2
\tag{B}
$$
***

那么如何从(B)理解我们一开始提到的逻辑回归的局限性呢，在$\lambda = 0$时，(B)与极大似然估计等价:
$$min_{\beta_0,\beta}\sum_{i=1}^n \log[1+e^{-y_i(\beta_0+x_i^{'}\beta)}]$$
在数据线性可分的情况下，我们可以找到一条直线$f(x) = \beta_0+x^{'}\beta$将数据点分开，$y_i(\beta_0+x_i^{'}\beta)$ 均为正。容易发现,nf(x)也可以分开数据点，并且n越大，目标函数值越小。
也就是说,优化问题的解空间是无解的，$\beta$ 越大，目标函数值越小。即这样计算，会得到无穷大的$\beta$，是不可行的。

***


```python
import numpy as np
import matplotlib.pyplot as plt
def hinge(x):
    if x > 0: 
        return x
    else:
        return 0
plt.figure(num=2, figsize=(13,6), dpi=80, facecolor='w', edgecolor='k')
x=np.linspace(-5,5,1000)  #这个表示在-5到5之间生成1000个x值
y=np.log([(1+np.exp(-i)) for i in x])  #对上述生成的1000个数循环用sigmoid公式求对应的y
ys = ([hinge(1 - i) for i in x]) 
plt.plot(x,y)  #用上述生成的1000个xy值对生成1000个点
plt.plot(x,ys)
plt.xlabel("yf(x)")
plt.ylabel("Loss")
plt.xticks(range(-5,5,1))
plt.show()  #绘制图像
```


![png](output_22_0.png)


**Figure 3**:铰链损失(hinge loss)（桔色）和二项损失（binomial loss）（蓝色）随间隔变化情况


 如上图所示，当间隔（margin）正向增大时，二项损失（binomial loss）趋近于0，间隔（margin）负向增大时，趋向与线性损失函数。从这个方面来讲，与铰链损失（hinge loss）相一致。  
这两者主要的区别是,铰链损失（hinge loss）在+1处有尖锐的拐点（sharp elbow），而二项损失（binomial loss）的变化则比较光滑。也就是说，与支持向量的二进制性质不同（有或无，见4.1），权重$p_i(1-p_i)$随决策边界距离增大光滑递减，二项损失对应的解包含了所有的数据的信息。  
**有趣的是，当$\lambda \to 0$ 岭逻辑回归(B)的解$\hat{\beta}$ 渐进至支持向量机（SVM）(A)的解**

---

如何理解这样的结论呢?
从直观上来看，$\lambda \to 0$时,对$\beta$长度的惩罚减小，相应间隔（margin）yf(x)绝对值增大，远离0。我们知道在这些区域，二项损失（binomial loss）趋近于铰链损失（hinge loss）（正向）或 铰链损失 + 某一常数（负向）。因此在$\lambda \to 0$时$\hat{\beta}$渐进至SVM的解是合理的。

---


### 4.4 SVM核方法简介

我们可以把拟合的线性函数写为

$$\begin{align}
\hat{f}(x) &= \hat{\beta}_0 + x'\hat{\beta}\\
& = \hat{\beta}_0 + \sum_{i \in S} \hat{\alpha}_i \left <x,x_i \right>
\end{align}$$



其中,$\left <x,x_i \right>$表示$x$,$x_i$的内积。  
现在我们考虑在更大的集合$h(x) = [h_1(x),h_2(x),...,h_m(x)]$上展开p维特征向量x。例如，我们使用总阶数为d的多项式基。只要我们能找到快速计算内积 $\left <h(x),h(x_j) \right>$，我们就能同样的在扩大的空间轻松计算出SVM的解。有很多方便的核函数有这样的性质。例如$K_d(x,z) = (1+\left <x,z \right>)^d$ 是总阶数为d的多项式基的展开$h_d$，并且$K_d(x,z) = \left <h_d(x),h_d(x_j) \right>$

多项式核主要在存在性证明中使用。更受欢迎的核（kernels）是径向基函数核（radial kernels）
$$ K(x,z)  = e^{-\gamma ||x-z||_2^2}$$
这个函数恒为正，可以看作在某一特征空间的内积。从理论上来说，这个特征空间是无穷维的，但是在计算上是有限的。

使用核方法，我们可以将(A)推广为:

$$
        \min{\beta_0,\beta}  \sum_{j=1}^n [1-y_j(\alpha_0+\sum_{i=1}^n \alpha_i K(x_j,x_i)]_{+} +  \lambda \alpha'K\alpha   \tag{C}
$$


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


# Our dataset and targets
X = np.c_[(.4, -.7),
          (-1.5, -1),
          (-1.4, -.9),
          (-1.3, -1.2),
          (-1.1, -.2),
          (-1.2, -.4),
          (-.5, 1.2),
          (-1.5, 2.1),
          (1, 1),
          # --
          (1.3, .8),
          (1.2, .5),
          (.2, -2),
          (.5, -2.4),
          (.2, -2.3),
          (0, -2.7),
          (1.3, 2.1)].T
Y = [0] * 8 + [1] * 8

# figure number
fignum = 1
kernel = 'rbf'
fig = plt.figure(num=2, figsize=(13,6), dpi=80, facecolor='w', edgecolor='k')

# fit the model
for penalty in (0.5,1):
    clf = svm.SVC(kernel=kernel, gamma=2,C = penalty)
    clf.fit(X, Y)

    # plot the line, the points, and the nearest vectors to the plane
    fig.add_subplot(1,2,fignum)

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10)
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

    plt.axis('tight')
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolor(XX, YY, Z > 0, cmap=plt.cm.Paired,alpha=0.1)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1
plt.show()
```


![png](output_30_0.png)


**Figure4**: 使用径向基函数核进行SVM分类，右图的B值更大。图中，实线表示原空间内的决策边界，(扩充特征空间中的线性边界)，虚线表示间隔

参考文献:  
Efron, B., & Hastie, T. (2016). Computer Age Statistical Inference: Algorithms, Evidence, and Data Science (Institute of Mathematical Statistics Monographs). Cambridge: Cambridge University Press. doi:10.1017/CBO9781316576533
