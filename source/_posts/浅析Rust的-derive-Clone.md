---
title: 浅析Rust的#[derive(Clone)]
date: 2022-10-13 22:35:17
tags:
- "Rust"
---

在学习rust的模板的时候, 我遇到了一个奇怪的问题. 就是关于derive(Clone)这个派生宏的. 有点意思, 跟大家分享一下.

# 问题来由

我们先来看这段代码:

```rust
use std::rc::Rc;
trait MayNotClone{}

#[derive(Clone)]
struct CloneByPtr<T> where T: MayNotClone{
    item: Rc<T>
}

trait MustBeClone: Clone {}
impl <T> MustBeClone for CloneByPtr<T> where T:MayNotClone{}
```

代码本身很简单, 首先声明一个trait叫做MayNotClone, 它不一定满足Clone这个trait, 然后, 再声明一个名为MustBeClone的trait, 这个trait必须实现Clone. 到这里, 一切正常, 对吧?

接下来, 就开始有点奇怪了, 当我们尝试为CloneByPtr这个结构体实现MustBeClone时, 编译器就会报错:
```
PS C:\Users\rivers\Desktop\example> cargo build
   Compiling example v0.1.0 (C:\Users\rivers\Desktop\example)
error[E0277]: the trait bound `T: Clone` is not satisfied
 --> src\main.rs:9:10
  |
9 | impl <T> MustBeClone for CloneByPtr<T> where T:MayNotClone{}
  |          ^^^^^^^^^^^ the trait `Clone` is not implemented for `T`
  |
note: required because of the requirements on the impl of `Clone` for `CloneByPtr<T>`
 --> src\main.rs:4:10
  |
4 | #[derive(Clone)]
  |          ^^^^^
note: required by a bound in `MustBeClone`
 --> src\main.rs:8:20
  |
8 | trait MustBeClone: Clone {}
  |                    ^^^^^ required by this bound in `MustBeClone`
  = note: this error originates in the derive macro `Clone` (in Nightly builds, run with -Z macro-backtrace for more info)
help: consider further restricting this bound
  |
9 | impl <T> MustBeClone for CloneByPtr<T> where T:MayNotClone + std::clone::Clone{}
  |                                                            +++++++++++++++++++

For more information about this error, try `rustc --explain E0277`.
error: could not compile `example` due to previous error
```

编译器提示得很清晰, MustBeClone这个trait要求实现它的结构体必须满足trait, 但是T(也就是那个MayNotClone)不满足Copy的trait, 请考虑为T添加一个clone trait的限制.

我不是加了#[derive(Clone)]么, 怎么还不好使. 

这个就有点无理取闹了啊. 这个CloneByPtr明明持有的是T的指针, 你指针能复制就好了嘛, 要指针指向的值能复制做干嘛. 

# 我们来看一下#[derive(Clone)]吧!

https://doc.rust-lang.org/std/clone/trait.Clone.html#derivable

根据官方文档说, This trait can be used with #[derive] if all fields are Clone.
如果一个struct的全部field都是Clone的, 那这个宏可以让这个struct也变为Clone的.

那我这个没毛病啊, std::rc::Rc不是实现了Clone么. 它直接生成item.clone()不就完事了么.

那问题出在哪了呢? 我觉得应该还是在于derive(Clone)上, 我们来研究一下这个宏, 看看怎么回事.

首先, 让我们先考虑一下, 这个宏都干了什么. 如果不用它, 我们自己手动实现Clone, 应该是什么样子呢? 下面我们以一个比较简单的struct举个栗子

```rust
struct ImStruct{
    a: i32,
    b: Vec<i32>
}
```
如果为它实现Clone那么就应该是
```rust
impl Clone for ImStruct {
  fn clone(&self) -> Self {
    Foo {
      a: self.a.clone(),
      b: self.b.clone(),
    }
  }
}
```

那如果加入泛型呢? 像这样:
```rust
struct ImGenericStruct<T, U> {
  a: u32,
  b: Rc<T>,
  c: U,
}
```
那如果为它实现Clone就应该是:
```rust
impl<T, U> Clone for ImGenericStruct<T, U> {
  fn clone(&self) -> Self {
    Foo {
      a: self.a.clone(),
      b: self.b.clone(),
      c: self.c.clone(), // 这里8行, C类型不一定
    }
  }
}
```
所以如果想要为这个struct实现Clone, 就必须确保U是Clone的:
```rust
struct ImGenericStruct<T, U: Clone> {
  a: u32,
  b: Rc<T>,
  c: U,
}
```
到这里, 一切正常, 对吧? 那#[derive(Clone)]问题出在哪了呢? 这个简单, 我们直接看它生成了什么代码就可以了. 比如gcc,clang, 有一个-E参数, 可以显示模板,宏展开后的代码. rust也有这个功能, 不过需要使用nightly版本的才可以启用此特性.
```bash
rustup toolchain install nightly 
cargo install cargo-expand 
rustup default nightly #这里将rust暂时切换为nightly的版本, 记得之后改回去
```
[cargo-expand](https://github.com/dtolnay/cargo-expand)

我们以上面 struct ImGenericStruct<T, U> 为例
```rust
#[derive(Clone)]
struct ImGenericStruct<T, U> {
  a: u32,
  b: Rc<T>,
  c: U,
}
```
其展开代码为:
```rust
struct ImGenericStruct<T, U> {
    a: u32,
    b: Rc<T>,
    c: U,
}
#[automatically_derived]
impl<T: ::core::clone::Clone, U: ::core::clone::Clone> ::core::clone::Clone
for ImGenericStruct<T, U> {
    #[inline]
    fn clone(&self) -> ImGenericStruct<T, U> {
        ImGenericStruct {
            a: ::core::clone::Clone::clone(&self.a),
            b: ::core::clone::Clone::clone(&self.b),
            c: ::core::clone::Clone::clone(&self.c),
        }
    }
}
```
由此可见, derive(Clone)画蛇添足地为T添加了Clone限制. emm, 这是rust的宏自身的限制, 它能做到读取代码的token流用来自动生成一些其他代码, 但是应该还不具备与编译器交互的能力, 也就没办法在宏展开时做出完善的类型检查, 所以只能简单粗暴地为里面出现的每一个模板参数都直接加上Clone的限制... 也许有一些其他的trait能避开这个问题, 不过直接手动实现一下, 也是一个可以考虑的方案.