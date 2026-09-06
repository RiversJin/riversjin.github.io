---
title: "以代数的方式看待 GPU 寻址"
date: 2026-09-07T00:24:42+08:00
tags: ["AI", "GPU", "Triton", "寻址"]
categories: ["AI"]
description: "从指针张量、tile、广播、mask 和 swizzle 的角度理解 Triton/GPU 寻址。"
comments: true
---

这篇文章是笔者在试图理解triton的时候写下的, 与CUDA/Rocm的经典的SIMT视角不同, triton每次管理的是一个tile, 它在读取/写入的时候需要一个特定形状的指针矩阵, 然后, 我的脑子就开始打结了...

# 指针张量的shape到底是什么?

举一个例子
```python
ptrs = A + rows[:, None] * stride_m + cols[None, :]
values = tl.load(ptrs)
```

这代码看起来很直白, 就是numpy的广播机制. 将行索引扩展列向量, 然后列索引扩展为行向量, 广播得到一个二维指针张量, 最后一个load

从这里, 我就开始懵了. values 有形状, ptrs 也有形状, A是一个指针, 它指向的数据也有某种"形状".
对任意一个元素, `ptr = A + stride_m * row + stride_n * col`
这很好理解, 但是对于一个tile呢?

上面一个tile的values的形状肯定与ptrs相同, 那ptrs如何与内存中一个形状, stride不同的数据对应呢?

或者更绕一点, 如果索引(地址)本身也是tensor, 那是不是还需要一个shape+stride的组合才能解释索引?

先从一个简单的例子开始. 假设A指向一个张量的起始位置, 我们构造三个指针

```python
offsets = tl.arange(0,3)
ptrs = A + offsets
```

那么ptrs就是
```text
[A+0, A+1, A+2]
```

> 要注意元素大小. 这里的+1, +2只代表元素位置, 不是字节;
> 我们的values与ptr形状完全一致, 那么就也是`1 * 3`

现在, 我们稍稍加一点难度
```python
ptrs = A + offsets * 2
```
指针与value的形状不变, 不过指向的数据指向了A那里跳跃的位置`A, A+2, A+4`, 取上来的数据依然是`1 * 3`的形状.

再稍微复杂一点, 比如, 我们要处理二维的数据. 假设我们要访问行跨度为4的矩阵中, 两行, 三列, 也就是
```python
rows = tl.arange(0, 2)
cols = tl.arange(0, 3)
ptrs = A + rows[:, None] * 4 + cols[None, :]
```
这里`rows[:, None]`的形状是`2 * 1`, `cols[None, :]`的形状是`1 * 3`. 二者广播运算后, 每个行索引与列索引结合, 得到一个`2 * 3`的指针张量
$$
A + \begin{bmatrix}
0 & 1 & 2 \\
4 & 5 & 6
\end{bmatrix}
$$
这个二维表格的每个数字, 都对应着去哪里取数的答案. `(1,0)`对应着偏移量4, `(1,2)`对应着偏移量6.
接下来的

```python
values = tl.load(ptrs)
```
就很简单了
```python
values[u, v] = memory[ptrs[u, v]]
```
`tl.load`其实完全不关心原矩阵的排列形式, 它只直接根据指针指向的位置而已.

所以, 理解triton的方式就是... 一个二维指针张量, 定义了一组按二维坐标组织的访问.
指针的shape描述了这些访问读入进来的形状, 而指针张量的数字, 决定了它会访问哪里.

用一个更数学的方式来说

**triton的读取/写入, 是一种从整数格点到一维地址的仿射映射**

# 转置是什么?

我们先举一个最简单的例子

$$
A = \begin{bmatrix}
a & b & c \\
d & e & f
\end{bmatrix}
$$
按照C语言的习惯, 它可以被描述为`int A[2][3]`, 那么它在内存上就是`A_m = a, b, c, d, e, f`.
如果我们将其转置呢? 它应该是
$$
A^T = \begin{bmatrix}
a & d \\
b & e \\
c & f
\end{bmatrix}
$$
在C中, 它就是`int At[3][2]`, 那么内存上就是`B_m = a, d, b, e, c, f`.

但其实, 这里有更取巧的方式, 比如我们定义一个函数`f(i, j) = i + 3j`, 然后从f给出的偏移量中去`A_m`中取数, 那么其实无须将`A_m`变成`B_m`的形式, 就能直接得到转置后的矩阵.

举个🌰, 对转置后的矩阵, 坐标有
$$
\begin{bmatrix}
(0,0) & (0,1) \\
(1,0) & (1,1) \\
(2,0) & (2,1)
\end{bmatrix}
$$
我们将其代入`f`, 就能得到这样一个偏移量矩阵:
$$
Offsets = \begin{bmatrix}
0 & 3 \\
1 & 4 \\
2 & 5
\end{bmatrix}
$$
按照这个矩阵来, 我们就能从A_m中取到
$$
A^T = \begin{bmatrix}
a & d \\
b & e \\
c & f
\end{bmatrix}
$$
你看, 只要我们加一层映射f, 无须移动内存中数据, 就能拿到转置后矩阵.

更一般的, 我们可以用这样一个公式, 对于一个采用 stride 描述存储布局的多维张量, 其逻辑坐标为$(i_0, \dots, i_{d-1})$, 对应的stride是$(s_0, \dots, s_{d-1})$, 它的位置就是
$$
T[i_0, \dots, i_{d-1}] = \left( p_T + \sum_{k=0}^{d-1} i_k s_k \right)
$$
同样是`A_m`这样一块内存, 如果输入集合为
$$
\mathcal{I} = [2] \times [3] = {(0,0), (0, 1), (0,2), (1,0), (1,1), (1,2)}
$$
如果我们用`f(i,j) = 3i +j`的方式, 就能看到矩阵A.
而如果用$L' = [3]\times[2]$, 以及`g(i,j)= i + 3j`, 就能看到转置后的矩阵$A^T$ .

所以, 老实说, 当我看到[Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose)这道题的时候, 我是有点懵的, 什么是把给定的tensor转置? 修改一下stride就行了呀?
$$
G: \underbrace{[3] \times [2]}_{\text{接收 B 的坐标}} \longrightarrow \underbrace{[2] \times [3]}_{\text{返回 A 的坐标}}
$$
对A来说, `f(i,j)=3i+j`, 对$A^T$来说, `f'(i,j) = 3j + i`就行了.

所以这道题与其说是转置, 不如说是使转置后的张量也变为连续的

> 设张量的形状为$n_0 \times n_1 \times \dots \times n_{d-1}$, 步长为$s_0, s_1, \dots, s_{d-1}$
> 若其是内存连续的, 那么有
> 1. 最后一维stride为1 ($s_{d-1} = 1$)
> 2. 其余stride满足 $s_k = n_{k+1} \times s_{k+1} \quad (\text{其中 } k = d-2, d-3, \dots, 0)$

那这个问题, 我们可以这么理解:
假设$A^T=B$, A的坐标值域是$L_A=[2]*[3]$, B的坐标值域是$L_B = [3] * [2]$
A的地址基址是$A_p$ , B的地址基址是`B_p`. $A_p$是输入, 而$B_p$ 是输出的位置.

因为二者都要是连续的, 所以$Stride_A = (3, 1)$, $Stride_B = (2, 1)$

对`B[u][v]`的元素, 它的位置是$B_p + 2u +v$
对`A[x][y]`的元素, 它的位置是$A_p + 3x +y$

转置嘛, 就是`let B[u][v] = A[v][u]`, 就行了. 这个逻辑在SIMT的思路中相当好写, 就
```text
对B的一个元素坐标u, v

val = load(A+3v+u)
store(B+2u+v, val)
```

而在triton这种分tile的就开始拧巴了(当然, 你也可以triton的每个block只处理一个元素, 但代价是会有一点点点点点... 慢)


tile的思路是, 每次会处理"一片元素"

但先等一下, "一片元素"到底是谁的一片? 是A的一片, 还是B的一片?

其实都可以. 我们先沿用上面的思路, 选B的一片.

假设这一片的形状是`2 * 2`, 它从B的`(u0, v0)`位置开始. 那么tile内部的坐标, 可以记为`(i, j)`, 而它对应B中的坐标就是

$$
u = u_0 + i, \qquad v = v_0 + j
$$

这里开始有两套坐标了, 但别急. `(i,j)`只是在说"这一片里面的第几行第几列", `(u,v)`则是在说"它在整个B里面的第几行第几列".

比如取`u0 = 0, v0 = 0`, 我们处理的就是B左上角的这一片:

$$
\begin{bmatrix}
a & d \\
b & e
\end{bmatrix}
$$

这一片里的每个位置, 都有两个地址: 一个是去A哪里读, 另一个是往B哪里写.

读地址来自`A[v][u]`, 所以是

$$
P_A(i,j) = A_p + 3(v_0+j) + (u_0+i)
$$

写地址来自`B[u][v]`, 所以是

$$
P_B(i,j) = B_p + 2(u_0+i) + (v_0+j)
$$

代入这个左上角的例子, 两张指针表就是

$$
P_A = A_p + \begin{bmatrix}
0 & 3 \\
1 & 4
\end{bmatrix}, \qquad
P_B = B_p + \begin{bmatrix}
0 & 1 \\
2 & 3
\end{bmatrix}
$$

注意看, **这两张表的shape完全一样, 但表里的地址不一样**.

从第一张表读取, 我们得到

$$
values = \begin{bmatrix}
a & d \\
b & e
\end{bmatrix}
$$

然后按照第二张表写出去, 就把`a, d, b, e`写到了B的前四个位置.

整个过程里, values的shape没有变, 也没有一个额外的"把values转一下"的动作. 转置的关系, 已经写在两张指针表里面了.

所以, 对于tile来说, 上面的标量代码其实没有变复杂, 只是把单个地址换成了一张地址表:

```python
u = u0 + tl.arange(0, 2)
v = v0 + tl.arange(0, 2)

src_ptrs = A + v[None, :] * 3 + u[:, None]
dst_ptrs = B + u[:, None] * 2 + v[None, :]

values = tl.load(src_ptrs)
tl.store(dst_ptrs, values)
```

`src_ptrs[i,j]`, `values[i,j]`, `dst_ptrs[i,j]`是一一对应的. 同一个位置, 从这里读, 然后写到那里. 指针张量负责给出地址, load/store负责按位置配对. 这也就是文档中[load](https://triton-lang.org/main/python-api/generated/triton.language.load.html)和[store](https://triton-lang.org/main/python-api/generated/triton.language.store.html)的语义.

# 所以tile的shape, 到底是谁的shape?

现在回头看, 我们似乎一直默认, tile必须长得像输入或者输出的一块矩形.

但实际上, tile的shape首先是在定义**这一批访问如何编号**.

在上面的例子里, 我们选择按B的坐标来编号. 所以values看起来就是B的一片, 而从A读取时, 地址是跳着走的.

那如果反过来, 按A的坐标来编号呢?

设A的形状为`M * N`, B的形状为`N * M`, 两者都按行连续存储. 我们取A中从`(m0,n0)`开始的一片:

```python
m = m0 + tl.arange(0, BM)
n = n0 + tl.arange(0, BN)

src_ptrs = A + m[:, None] * N + n[None, :]
dst_ptrs = B + n[None, :] * M + m[:, None]

values = tl.load(src_ptrs)
tl.store(dst_ptrs, values)
```

这里所有指针张量和values的shape, 都是`BM * BN`.

你可能又想问了: 写出去的那片B, 不应该是`BN * BM`吗?

对, **它在B里面占据的区域**是`BN * BM`. 但是这张写地址表, 仍然是按A的局部坐标编号的, 所以是`BM * BN`.

比如A的整个`2 * 3`矩阵, 对应的写偏移量表就是

$$
\begin{bmatrix}
0 & 2 & 4 \\
1 & 3 & 5
\end{bmatrix}
$$

把`a,b,c`分别写到`0,2,4`, 把`d,e,f`分别写到`1,3,5`, 最后的内存自然就是`a,d,b,e,c,f`.

所以, 指针张量画出来是几行几列, 并不意味着它指向的内存区域也要按这个方向排列. 这张表只是一个工作清单, 每个格子里写着一个地址.

# 将这个过程写成映射

这时候, 我们可以把视角再往前推一点. 前面只关心tile里的每个位置要访问哪里, 但真正执行的时候, 总得有某个线程来处理这个位置.

于是, 从一个lane出发, 到最终的内存地址, 可以写成这样一条链:

```text
Lane
  ↓ F
Iteration coord
  ↓ G
Logical tensor coord
  ↓ S
Swizzled coord
  ↓ Layout
Physical address
```

别被这些名字吓到, 我们一个一个来看.

## F: 我负责tile里的哪个位置?

Iteration coord, 我们可以先把它理解成"这一批工作里的坐标". 比如前面那个`2 * 2`的tile, 它的iteration coord就是`(i,j)`.

假设暂时只用4个lane, 每个lane处理一个元素, 那么一种分配方式是

$$
F(\ell) = (\lfloor \ell / 2 \rfloor,\ \ell \bmod 2)
$$

也就是

```text
lane 0 → (0,0)
lane 1 → (0,1)
lane 2 → (1,0)
lane 3 → (1,1)
```

F回答的是: **谁来处理这个格子?**

当然, 一个lane可以处理多个元素. 这时候给F再加一个参数r, 表示"这个lane负责的第几个元素", 写成`F(lane, r)`就行了. 多个warp一起处理时, 也可以把warp编号带上. 图里先只写Lane, 是为了把主线画清楚.

所以, tile的shape并不直接等于线程的排列. 同一个`2 * 2`的tile, 可以分给4个lane, 也可以让2个lane各自处理2个元素. 这是F的选择.

## G: 这个位置对应张量里的哪个元素?

这里继续按B来选tile, 起点是`(u0,v0)`. 对同一个iteration coord `(i,j)`, 写入B时对应的逻辑坐标是

$$
G_B(i,j) = (u_0+i,\ v_0+j)
$$

而读取A时, 因为我们要做转置, 对应的逻辑坐标是

$$
G_A(i,j) = (v_0+j,\ u_0+i)
$$

注意, 这里还没有stride, 也没有内存地址. G只回答: **这份工作要用张量里的哪个元素?**

同一个iteration coord, 对不同的输入输出, 可以有不同的G. 这就是前面两张指针表能按位置配对的原因: 它们共享一套工作编号, 但分别去找自己的元素.

## S和Layout: 这个元素实际放在哪里?

如果张量就是普通的行连续存储, S什么也不用做:

$$
S(x,y) = (x,y)
$$

然后Layout按照stride把坐标变成地址. 对前面的A和B来说, 就是

$$
\operatorname{Layout}_A(x,y) = A_p + 3x+y
$$

$$
\operatorname{Layout}_B(u,v) = B_p + 2u+v
$$

这里把基址也放进Layout里, 地址加法仍然按元素计. 于是整条链可以合起来写成

$$
\operatorname{Addr}_T
= \operatorname{Layout}_T \circ S_T \circ G_T \circ F
$$

$\circ$就是函数复合, 从右往左看. lane先通过F找到自己负责的位置, 再通过G找到逻辑元素, 最后经过S和Layout找到存储地址.

比如lane 1, 在B左上角这个tile里:

```text
lane 1
  ↓ F
(0,1)
  ├─ G_A → (1,0) → S_A不变 → Layout_A → A_p + 3
  └─ G_B → (0,1) → S_B不变 → Layout_B → B_p + 1
```

所以它把d从A的偏移3, 搬到B的偏移1. 绕了一圈, 还是前面的那次load和store, 只是现在我们知道每一步分别在做什么了.

那中间的S为什么要单独留出来? 先放一下, 我们把广播和mask也放进来, 再回头看它.

# 广播: 多个工作位置, 对应同一个元素

假设有这样一个计算:

```text
C[i,j] = A[i,j] + b[j]
```

A和C的shape是`M * N`, b的shape是`N`. 我们的工作空间仍然是二维的, 每个`(i,j)`算一个C元素.

对A来说, G就是

$$
G_A(i,j) = (i,j)
$$

而对b来说, G变成了

$$
G_b(i,j) = j
$$

它把i丢掉了. `(0,2)`, `(1,2)`, `(2,2)`这些不同的工作位置, 全都去读同一个`b[2]`.

**这就是广播在这套框架里的样子: G可以是多对一的映射.**

我们不需要先把b复制成一个`M * N`的矩阵, 只要让不同的位置指向相同的元素就行了. 如果一定要用二维stride来写, 也可以把广播后的b看成

$$
\operatorname{Addr}_b(i,j) = b_p + i \times 0 + j \times 1
$$

也就是stride为`(0,1)`. 沿着第一维走, 地址不动.

这两种写法是在表达同一件事: 一种在G里丢掉坐标, 另一种在Layout里让它乘上0.

再回头看一开始的

```python
rows[:, None] * stride_m + cols[None, :]
```

就很直白了. 对结果里的`(i,j)`, 左边取`rows[i]`, 右边取`cols[j]`. 两边分别忽略一个坐标, 然后在同一套二维工作坐标上相加, 得到地址偏移量. Triton的[广播规则](https://triton-lang.org/main/python-api/triton-semantics.html)描述的就是这种shape扩展, 不需要先复制出完整的数据矩阵.

这里说的是读和计算. 如果写地址也多对一, 多个位置就可能争着写同一个地址, 那是另一个需要处理的问题了.

# mask: 这个工作位置要不要执行访问?

mask没有给出新的地址, 它给出的是一个判断.

比如B的形状是`N * M`, tile起点为`(u0,v0)`, 那么

$$
P(i,j) = (u_0+i < N) \land (v_0+j < M)
$$

就是这次访问的有效条件. 这里的起点和局部坐标都非负, 所以只需要检查上界.

前面的链负责回答"去哪里", P负责回答"去不去":

```text
lane → F → iteration coord → G → S → Layout → address
                  │
                  └─ P → true / false → 是否执行这次访问
```

对load来说, 可以写成这样的伪代码:

```python
q = F(lane, r)
if P(q):
    value = load(Layout(S(G(q))))
else:
    value = other
```

因此, mask并没有把tile裁成另一个shape. 无效的格子还在, 只是没有发生对应的内存读取, 它的值由`other`提供. 对store来说, 则直接跳过这个格子的写入. 这和[Triton load的定义](https://triton-lang.org/main/python-api/generated/triton.language.load.html)是一致的.

边界判断只是其中一种P. 比如计算下三角区域时, 还可以加上`row >= col`. 我们依然在同一个工作空间里, 只是选择其中一部分位置参与访问.

# swizzle: 元素没换, 存放的位置换了

现在来看S.

假设有一个`4 * 4`的小矩阵, 正常情况下, 逻辑坐标`(x,y)`会直接交给Layout, 得到偏移`4x+y`.

我们可以在中间插入一个变换:

$$
S(x,y) = (x,\ y \mathbin{\oplus} x)
$$

这里的$\oplus$是按位异或, 也就是代码里的`^`. 对这个例子, x和y都在0到3之间.

经过S再交给Layout, 就得到

$$
\operatorname{offset}(x,y) = 4x + (y \mathbin{\oplus} x)
$$

把每个逻辑位置对应的偏移量列出来:

$$
\begin{bmatrix}
0 & 1 & 2 & 3 \\
5 & 4 & 7 & 6 \\
10 & 11 & 8 & 9 \\
15 & 14 & 13 & 12
\end{bmatrix}
$$

比如逻辑元素`T[1,0]`, 原来放在偏移4, 现在放在偏移5. 但它仍然叫`T[1,0]`, 计算里使用它的含义没有变化.

这和转置放在G里, 区别就出来了:

**G改变这份工作要找哪个逻辑元素; S改变这个逻辑元素存在哪里.**

当然, 数据得真的按这套规则存进去, 之后也按同一套规则读出来. 不能对一块普通存储的数据, 临时加一个S去读, 就指望读到的还是原来的元素.

那为什么要把存储顺序打乱呢?

一个典型用途是shared memory. 我们先假想只有4个bank, 每个bank按元素轮流接收地址, 即`bank = offset % 4`. 这只是缩小的示意模型.

没有swizzle时, 同时读一列, 偏移量是`0,4,8,12`, 全都落到bank 0. 加上上面的S以后, 同一逻辑列的偏移量变成`0,5,10,15`, 恰好分散到4个bank.

读取的仍然是同一列元素, 但它们不再挤同一个bank了. 实际GPU要结合bank数量、访问宽度和线程分配来选择S, 不是随便异或一下都会更快.

在这里, 我们把swizzle画成坐标变换, 比较好理解. 实现里也经常直接改地址偏移的bit, 比如[CuTe的Swizzle](https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp). 那种写法会把swizzle放到基础Layout之后; 看整条链最终算出来的地址就好, 不必被函数摆放的位置绕进去.

# 再回来看这条链

现在, 几件看起来不同的事就能放到一起看了:

| 操作 | 在这套框架里的解释 |
| --- | --- |
| 换一种lane分工 | 改F, 决定谁处理哪些iteration coord |
| 转置、切片 | 改G, 决定工作位置对应哪个逻辑元素 |
| 广播 | G可以多对一, 也可以用零stride表达 |
| mask | 给工作位置附上条件P, 决定是否执行访问 |
| swizzle | 改S, 重排逻辑元素的存储位置 |
| 行连续、列连续、padding | 改基础Layout的坐标到地址规则 |

这也解释了为什么, 光看一个指针张量的shape, 还不能判断访问快不快.

相邻lane最终访问的地址, 要看`Layout ∘ S ∘ G ∘ F`整个组合. 换了F, 即使每个逻辑元素的存储位置都没变, 一组lane同时访问的地址也可能完全不同.
