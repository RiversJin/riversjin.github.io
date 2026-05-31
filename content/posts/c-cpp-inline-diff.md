---
title: "C, C++ inline 关键字差异"
date: 2023-06-01T00:00:00+08:00
tags: ["C", "C++", "编译"]
categories: ["C/C++"]
description: "C 与 C++ 中 inline 关键字的语义差异"
comments: true
---

注: 在C89(ANSI C)中, 是没有inline的, gun的c扩展支持类似于C++的inline语义, 但那是非标准情况, 不属于这里的讨论范围.

# C++ (C++11)
C++的inline并不是建议编译器内联, 实际上它是告诉编译器, 此符号可能在多个翻译单元中存在, 如果发生重复, 不会发生错误, 而是在最终链接产物时, 只保留其中一个. 
通过这个功能, 我们才得以在C++头文件定义模板而无需担心多个cpp文件由于模板实例化导致符号冲突. 这会实现某种意义上的"去重".
而至于是否内联, 则取决于编译器自身的优化逻辑.

# C99

## inline
C语言的inline与C++的并不相同, 它更多是一种类似于C++"模板"的特性.
当一个函数被声明为inline修饰, 像
```c
inline int foo(){
	return 42;
}
```
这会使得在obj文件, 并不没有foo函数这么一个存在.
它指的是, **如果编译器决定将foo内联**, 那么会将`inline int foo(){...}`内的函数体内联进去.
但如果编译器选择不内联foo, 那么编译器会忽略`inline int foo(){...}`的实现, 而是转而假设这是一个外部的符号.
但如果其他obj文件也没有导出foo这么个函数, 那么就会在链接时喜提`undefined symbol`错误

这也就有一个很有趣的现象, 在C中, 通过inline与非inline, 实际上是可以定义出一模一样的两个函数来的:
```c
/* foo.c */
int foo(){
	return 41;
}

/* main.c */
#include <stdio.h>
inline int foo(){
	return 42;
}

int main(){
	int n = foo();
	printf("%d\n", n);
	return 0;
}
```
并且, 你会发现, 在不同的优化等级, 其运行结果还不一样:
```
(py311) ➜  /tmp gcc demo.c foo.c -O0 -std=c11 -o demo
(py311) ➜  /tmp ./demo
41
(py311) ➜  /tmp gcc demo.c foo.c -O2 -std=c11 -o demo
(py311) ➜  /tmp ./demo                               
42
```
在O0时, 编译器会禁用内联, 于是假设foo是一个来自外部的符号, 最终调用`foo.c`中return 41的foo, 而如果开启优化后, 编译器会选择使用inline版本的foo, 于是返回42.

## extern inline
extern inline有点类似于C++的"模板实例化", 假如有这样的代码:

```c
inline int foo(){
	return 42;
}

extern inline foo();
```

这里extern不是声明外部有一个foo可供链接, 而是使`foo`函数导出到外部. 但是, 我们知道, 单被inline修饰的函数并不在obj文件中存在, 但如果用户需要使其导出到外部, 那么必须有实际的函数体与之对应, 于是foo就从一个"inline 函数模板", 变成obj文件中确确实实存在的一个函数以及代码段.

所以, 一般的用法是, 在一个头文件中, 声明一个inline函数, 并在一个c文件中使用`extern inline f();`. 如果编译器决定内联, 那么就会直接使用头文件中的定义, 如果编译器不内联, 那么就会链接到通过`extern inline`"实例化"的函数上.

当然, 如果想要内联与不内联使用不同的实现, 那么这里也可以像上面一样, 直接在c文件里面重新实现一个不一样的同名函数.

# static inline
static inline与extern inline类似, 也是迫使这个inline函数"实例化", 不过不会导出.
所以更通常的使用方法就是, 直接在头文件中声明函数体为`static inline`, 这样每个翻译单元(obj)文件中默认就会产生一个自己的static的foo函数. 相对来说是最稳妥的方案. 也不会有使用`extern inline`可能的不一致情况.
那么, 在这种情况下, 其实在头文件中用static

# End

最后需要指出的是, 无论是C还是C++, inline都不意味着这个函数会被内联, 编译器会自行决定, 如果认为一定要内联才行, 那么可能需要使用一些编译器扩展, 比如
```cpp
__attribute__((always_inline))
inline int foo(){
	return 42;
}
````
这样, 无论在什么优化级别, 编译器都会内联这个函数.
但我个人建议最好别盲目使用这个特性, 内联有可能使性能变好, 也可能变坏(代码段膨胀后cache miss), 由编译器自行决策在大部分场景下都是更好的, 除非测试表明确实存在性能瓶颈.
