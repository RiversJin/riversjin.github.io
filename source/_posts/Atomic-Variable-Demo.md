---
title: Atomic Variable Demo
date: 2022-12-07 22:50:55
tags:
---

接上一篇文章, 这一篇我们来用一些示例程序, 来探讨一下在不同平台下原子变量的行为.

示例代码库我放在了 https://github.com/RiversJin/AtomicVariableDemo
使用Rust实现(想顺便温习一下Rust的语法)

项目结构很简单
```
abc@310a0ca1fb3d:~/workspace/t/src$ tree
.
├── bin
│   ├── t1.rs
│   ├── t2.rs
│   ├── t3.rs
│   ├── t4.rs
│   └── t5.rs
└── main.rs

1 directory, 6 files
```
# 计数器

首先是main.rs, 这里比较了两种计数器相加的方法, 一个是使用普通变量全局变量R_MUT, 另一个是使用原子变量R, 其中, 原子变量使用Relaxed的内存序约束.

这种情况下, 无论什么平台, 普通变量的数字都可能会发生错误, 而原子变量正常.

Windows amd64:
```
PS C:\Users\rivers\Desktop\AtomicVariableDemo-master> cargo run  --bin t
    Finished dev [unoptimized + debuginfo] target(s) in 0.01s
     Running `target\debug\t.exe`
R:100000000 r:24134332
PS C:\Users\rivers\Desktop\AtomicVariableDemo-master> cargo run  --bin t
    Finished dev [unoptimized + debuginfo] target(s) in 0.01s
     Running `target\debug\t.exe`
R:100000000 r:24366177
PS C:\Users\rivers\Desktop\AtomicVariableDemo-master> cargo run -r --bin t
    Finished release [optimized] target(s) in 0.01s
     Running `target\release\t.exe`
R:100000000 r:10000004
PS C:\Users\rivers\Desktop\AtomicVariableDemo-master> cargo run -r --bin t
    Finished release [optimized] target(s) in 0.01s
     Running `target\release\t.exe`
R:100000000 r:10000004
```
可以看到, 如果不使用release配置编译, 普通变量的结果每次都不同, 发生了竞态行为.
而采用release编译后, 编译器可能对循环相加的操作做出了一些优化, 每次运行的值稳定了, 虽然结果不对.

接下来, 使用弱内存序的Aarch64架构实验, 得到的结论类似:
```
abc@310a0ca1fb3d:~/workspace/t/src$ cargo run --bin t
    Finished dev [unoptimized + debuginfo] target(s) in 0.01s
     Running `/config/workspace/t/target/debug/t`
R:100000000 r:74990565
abc@310a0ca1fb3d:~/workspace/t/src$ cargo run --bin t
    Finished dev [unoptimized + debuginfo] target(s) in 0.01s
     Running `/config/workspace/t/target/debug/t`
R:100000000 r:74348213
abc@310a0ca1fb3d:~/workspace/t/src$ cargo run -r --bin t
   Compiling t v0.1.0 (/config/workspace/t)
    Finished release [optimized] target(s) in 2.51s
     Running `/config/workspace/t/target/release/t`
R:100000000 r:10000004
abc@310a0ca1fb3d:~/workspace/t/src$ cargo run -r --bin t
    Finished release [optimized] target(s) in 0.01s
     Running `/config/workspace/t/target/release/t`
R:100000000 r:10000004
```

# 标志位
这个测试中, 我们可以看到存在 load-load(读取flag后再读值), store-store(先写值再写flag), 还有load-store(读取完成后清空flag)

t1.rs, t2.rs与t3.rs分别展示了使用普通变量, 误用原子变量以及正常使用原子变量做并发标志位的情形.我们先测试一下Amd64的情况:
```
PS C:\Users\rivers\Desktop\AtomicVariableDemo-master> cargo run  --bin t1
   Compiling t v0.1.0 (C:\Users\rivers\Desktop\AtomicVariableDemo-master)
    Finished dev [unoptimized + debuginfo] target(s) in 0.59s
     Running `target\debug\t1.exe`
PS C:\Users\rivers\Desktop\AtomicVariableDemo-master> cargo run  --bin t2
   Compiling t v0.1.0 (C:\Users\rivers\Desktop\AtomicVariableDemo-master)
    Finished dev [unoptimized + debuginfo] target(s) in 0.52s
     Running `target\debug\t2.exe`
PS C:\Users\rivers\Desktop\AtomicVariableDemo-master> cargo run  --bin t3
   Compiling t v0.1.0 (C:\Users\rivers\Desktop\AtomicVariableDemo-master)
    Finished dev [unoptimized + debuginfo] target(s) in 0.51s
     Running `target\debug\t3.exe`
```

而Aarch64下, 稍稍等一会(我这里跑了3分钟), 就可以看到t1, t2出现了异常值:
```
abc@310a0ca1fb3d:~/workspace/t/src$ cargo run --bin t1
   Compiling t v0.1.0 (/config/workspace/t)
    Finished dev [unoptimized + debuginfo] target(s) in 1.60s
     Running `/config/workspace/t/target/debug/t1`
Get! V1.0=0
^C
abc@310a0ca1fb3d:~/workspace/t/src$ cargo run --bin t2
   Compiling t v0.1.0 (/config/workspace/t)
    Finished dev [unoptimized + debuginfo] target(s) in 1.59s
     Running `/config/workspace/t/target/debug/t2`
Get! V1.0=0
abc@310a0ca1fb3d:~/workspace/t/src$ cargo run --bin t3
   Compiling t v0.1.0 (/config/workspace/t)
    Finished dev [unoptimized + debuginfo] target(s) in 1.61s
     Running `/config/workspace/t/target/debug/t3`
```

# 多核心观测写入顺序

t4.rs与t5.rs简单地复现了一下上篇文章提到的多线程观察修改顺序的问题, 来区分acquire-release与acquire-release. 这个地方和理论上有一些出入. 理论上来说, 对于Aarch64架构, 硬件是不保证多核心的观察必定一致的, 但实际上运行代码会发现, t4与t5使用acquire-release和sequentially-consistent都是不能复现出z==0. 这说明其实Aarch64实际上是保证了这个一致性(至少我的Arm架构的路由器是这样). 

我在StackOverflow上找到了这样一篇回答, 听上去很有道理, 大家可以参考一下
https://stackoverflow.com/questions/67397460/does-stlrb-provide-sequential-consistency-on-arm64

"ARMv8 originally allowed that on paper, but no ARM CPUs ever did."

更深层次的, 如果不保证这种观察上的一致性, 说明CPU的缓存同步使用的是某种广播机制, 但是工业界的cpu没这么搞, 这篇回答中也提到了只有一些很少的POWER架构的CPU才可能存在这种问题.

所以, 保险起见, 在遇到需要acquire-release语义的原子变量时, 简单使用sequentially-consistent即可, 效率都一样, 反正大部分硬件不支持更灵活的控制.