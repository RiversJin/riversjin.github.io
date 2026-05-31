---
title: "Rust UnsafeCell VS C++ mutable"
date: 2023-06-15T00:00:00+08:00
tags: ["Rust", "C++", "内存模型"]
categories: ["Rust"]
description: "对比 Rust UnsafeCell 与 C++ mutable 的语义与用途"
comments: true
---

经常在C++中写引用计数的朋友都知道(误), 如果一个class/struct被const修饰, 那么无法修改里面的元素, 但是, 可以通过`mutable`修饰某个成员, 相当于开了个洞, 取消掉其const限定, 实现修改被const限定结构体的内部成员.

这种限制不仅作用于C++类型系统, 同时也会影响C++的代码优化, 我们下面简单举一个例子:
```c++
#include <iostream>
#include <fmt/core.h>
using namespace std;

struct S{
    int a;
    mutable int b;
};

void print(int a, int b){
    cout << fmt::format("a = {}, b = {}", a, b) << endl;
}
void modify_const(const S& s){
    S& m_s = const_cast<S&>(s);
    m_s.a += 1;
    m_s.b += 1;
}

int main(){
    const S s{1, 1};
    print(s.a, s.b);
    modify_const(s);
    print(s.a, s.b);
    return 0;
}
```
输出如下
```
➜  mutable-demo clang++ demo.cpp  -O0 -lfmt -o demo
➜  mutable-demo ./demo                             
a = 1, b = 1
a = 1, b = 2
```
可以看到, 由于变量s为const, 并且modify_const表明自己只是用const引用, 不会修改数据, 但是依旧悄摸摸地改掉了a和b后. 接下来的print会假设结构体s没有发生变化, 所以打印出a依旧为1, 但是由于b被mutable修饰, 它的值没有被编译器认定为不变, 所以可以拿到新的值2.

由于C++中, 变量默认就是可变的, 所以我们用到mutable的机会并不多.

而Rust和C++是相反的, 所有变量, 如果不被声明为`mut`, 那就是不可变的, 这种情况下, 由于大部分变量都是不可变的, 显然类似于上面的值缓存现象会更多, 这也允许编译器进行更多更激进的优化. 不过, 相同的场景还是存在的, 想象一下, 假如我们在使用引用计数指针Rc. 

在需要增加计数是, 我们会使用 ``Rc::clone``, 它的类型为 ``fn clone(&self) -> Rc<T>``, 这与上面的``modify_const``是一样的, 它接受一个不可变引用, 但是依旧改变了内部的值(引用计数+1).

在Rust中, 不是使用类似的关键字, 而是使用``UnsafeCell``. 
虽然UnsafeCell不是关键字, 但是在标准库中可以看到, UnsafeCell头部存在``#[lang = "unsafe_cell"]``的标记, 这其实也是引导编译器特定识别这个结构, 从而禁用某些优化. 无论是Cell, RefCell或者是其他需要引用计数的机制的数据结构, 都会依赖UnsafeCell实现.

不过遗憾的是, 我没能直接使用Rust复现上述C++的例子, 在Rust中, 从一个不可变借用获得可变借用的唯一合法路径有且只有使用UnsafeCell.  强行使用其他方法获取可变借用会导致编译失败. 相对来说, 在Rust中写出UB的难度还是挺大的.

不过. 总的来说, 不影响这篇的中心思想, 即

C++通过mutable移除结构体成员的const限定, 而Rust通过UnsafeCell实现.
