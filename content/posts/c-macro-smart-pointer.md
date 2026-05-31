---
title: "用宏实现一个智能指针"
date: 2023-09-01T00:00:00+08:00
tags: ["C", "宏", "智能指针", "引用计数"]
categories: ["C/C++"]
description: "用 C 宏实现 shared_ptr，探讨宏的能力与引用计数智能指针的实现"
comments: true
---

# Background
作为一个从C++入门学习编程的人, 没有GC, 但是一直以来Cpp的智能指针用着还算顺手. 但是这一切美好都被工作时遇到的纯C代码给破坏了. 
由于前人的肆意挥洒, 这坨号称是C的克苏鲁已经彻底与CPP分道扬镳, 无法再享受C++与C的兼容性.
用Rust的bindgen也许可以让这坨十年前的古董也享受一下现代编程语言的发展, 无奈周围的人无人肯学 :(

但是, 日子总还得过. 上帝说过, 使用泛型容器是每个人自由而平等的权力, 不容侵犯. 
况且即使是C, 也总有引用计数, 数据结构容器的需求. 
所以, 在与命运抗争的不甘之下, 我开始尝试用宏在C泛型容器. 经过尝试发现并不难, 就是有点丑, 不太优雅, 但多少也能凑合着用.
这篇文章, 我们以宏实现一个shared_ptr. 来探讨一下宏的能力, 以及复习一下基于引用计数的智能指针都有哪些功能.

# 宏 Macro
C的宏功能很弱, 基本就是个文本替换, 也不是什么图灵完全的, 能力很有限. 但是有一些很有趣的用法

一个"#"代表"Stringizing" 字符串化, 它可以将后面的宏参数变为一个字面量字符串, 比如
```c
#define str(s) xstr(s)
#define xstr(s) #s

// str(abc) 会被展开为 "abc"
```
这里有一个地方也很有趣, 明明是 `#define str(s) #s` 似乎就可以了, 但是我们还是需要两个宏才行.
这就是我之前说的, C的宏很弱, 它执行的只是文本替换.

假如我们有下面的定义
```c
#define str(s) xstr(s)
#define xstr(s) #s

#define foo 42

xstr(foo) -> #foo -> "foo"

str(foo) -> str(4) -> xstr(4) -> #4 -> 4
```
这与宏的工作机制有关. 有两点
1. **除非**被字符串化或者连接, 宏参数在替换到宏之前会被扩展
2. 在宏参数替换到宏内部后, 会再次扫描整个宏体, 查找其内部是否还有需要被展开的宏调用.
所以, 由于这个机制, 当遇到宏的连接以及字符串化时, 必须用两层宏来实现. 实现机制相关, 简单记住就好, 在C中这还是一个挺常见的宏技巧.

而两个#, 即"##"代表连接, 它可以将两个东西连接在一起, 比如
```c
#define expandconcat(a, b) concat(a, b)
#define concat(a, b) a##b

expand(abc_, def) -> abc_def
```
由于宏是文本替换嘛, 这么做可以通过宏来生成各种新的变量名, 结构体名什么的.

## 可变参数的宏
> 此处是炫技时间, 可以直接跳过, 不影响其他部分的阅读

宏也是可以实现可变参数的. 比如我们上面提到的连接, 如果希望实现若干个token的连接, 那么可以有
```c
#define CONCAT2(a, b) a##b
#define CONCAT3(a, b, c) a##b##c 
#define CONCAT4(a, b, c, d) a##b##c##d 
#define CONCAT5(a, b, c, d, e) a##b##c##d##e
```
但是这么搞太沙雕了, 使用者还得自己确定要连接几个参数, 让C语言本就人机工程不友好这点更加雪上加霜. 
但其实, 宏也是可以有可变参数的, 只不过依旧是宏替换罢了. 我们可以先计算宏参数的个数, 然后选择调用某个特定的宏, 像这样:
```c
#define COUNT_ARGS(...) COUNT_ARGS_HELPER(__VA_ARGS__, 5, 4, 3, 2, 1, 0) 
#define COUNT_ARGS_HELPER(_1, _2, _3, _4, _5, N, ...) N
```
在COUNT_ARGS中, "..."会变成COUNT_ARGS_HELPER中的__VA_ARGS__, 像这样:
```c
COUNT_ARGS(x, y, z) => COUNT_ARGS_HELPER(x, y, z, 5, 4, 3, 2, 1, 0)
COUNT_ARGS_HELPER(x, y, z, 5, 4, 3, 2, 1, 0) 会与其参数进行匹配, 像这样:
 x,  y,  z,  5,  4, 3, 2, 1, 0
_1, _2, _3, _4, _5, N, ( ... )
那么经过匹配, 得到N为3, 所以COUNT_ARGS_HELPER(x, y, z, 5, 4, 3, 2, 1, 0)的展开结果就会是 3
```
是不是很有趣?
由于宏没有循环, 所以, 如果想要支持更多的参数, 那么这个数参数的方法就得自己再向这两个宏里面依次继续添加即可.
拿到参数后, 我们就可以这样:
```c
#define CONCAT(...) CONCAT_HELPER(COUNT_ARGS(__VA_ARGS__), __VA_ARGS__) 
#define CONCAT_HELPER(N, ...) CONCAT_HELPER2(N, __VA_ARGS__) 
#define CONCAT_HELPER2(N, ...) CONCAT##N(__VA_ARGS__)
```
最终调用对应参数的实际拼接宏:
```c
CONCAT(x, y, z) => CONCAT_HELPER(3, x, y, z) => CONCAT_HELPER2(3, x, y, z) => CONCAT3(x, y, z) => x##y##z => xyz
```

# 引用计数的智能指针

基础版本的智能指针, 只需要两个东西即可:
```c
typedef struct arc{
	void* ptr;
	int* cnt;	
} arc_t;
```
ptr代表指向数据的指针, 而cnt代表指向引用计数器的指针. 
因为我们需要让多个指针共享同一个计数器, 所以cnt不能直接放在结构体内.

这里我得说明一点就是, 在我的设计中, struct arc_t是可以以值传递的, 这样就与正常的智能指针相似一点, 不然如果只能用指针访问这个结构体的话, 那么就成了侵入式指针了, 像这样:
```c
typedef struct arc1{
	void *ptr;
	int cnt;
} arc1_t;
```
这样也不是不能用. 但是个人认为性能会差一点.
arc_t的传递可以以值传递, 并且由于只有16字节, 传参的时候用两个寄存器就行. 但如果使用arc1_t, 传参必须用指针, 那么间接寻址会相对多一些.
也可以这么做, 看个人偏好吧.

但`void*`是坏东西, 我不喜欢, 这时候我们的宏就排上用场了, 我们可以这样
```c
#define TYPE int
#define ARC_TYPE_NAME CONCAT(arc_, TYPE _t)
#define ARC_NEW_FUNC_NAME CONCAT(arc_, TYPE, _new)
#define ARC_DROP_FUNC_NAME CONCAT(arc_, TYPE, _drop)
#define ARC_CLONE_FUNC_NAME CONCAT(arc_, TYPE, _drop)
struct ARC_TYPE_NAME{
	TYPE *ptr;
	int* cnt;
} ARC_TYPE_NAME;

ARC_TYPE_NAME ARC_NEW_FUNC_NAME(TYPE *ptr){
	int *cnt = (int*)malloc(sizeof(int));
	assert(cnt);
	cnt = 1;
	return (ARC_TYPE_NAME){
		.ptr = ptr,
		.cnt = cnt
	};
}

void ARC_DROP_FUNC_NAME(ARC_TYPE_NAME arc){
	if(atomic_fetch_sub(arc.cnt) == 1){
		free(arc.ptr);
		free(arc.cnt);
	}
}

ARC_TYPE_NAME ARC_CLONE_FUNC_NAME(ARC_TYPE_NAME arc){
	atomic_fetch(arc.cnt);
	return arc;
}

#undef ARC_TYPE_NAME
#undef ARC_NEW_FUNC_NAME
#undef ARC_DROP_FUNC_NAME
#undef ARC_CLONE_FUNC_NAME
```
我们可以将上面的内容保存为一个文本文件(不必是.h), 比如 arc.template
之后, 假如我们需要实现指向double的引用计数指针, 那么可以这样
```c
// in arc_double.h
#define TYPE double
#include "arc.template"

```
通过#include包含此template, 即可"手动触发一次实例化". 如果希望使用其他数据结构, 那么依次类推解决继续define后再include即可.

手动挡的模板, 嘿嘿没见过吧 :P
# WeakPtr
通常, 我们知道, 引用计数的指针, 通常会存在循环引用导致内存泄漏的问题. 所以为了解决这点, 会有weak_ptr类型, 它可引用一个shared_ptr, 它不会影响引用计数, 但是可以尝试升级为shared_ptr. 如果最初引用的shared_ptr已经释放, 那么此次升级无效.

为了简化代码示例, 下面不会再用"宏模板"的格式. 聪明的你会理解的.
```c
typedef struct ref_cnt_t{
	int cnt;
	int weak_cnt;
	int *data; // TYPE *data;
} ref_cnt_t;

typedef struct arc_t{
	int *data; // TYPE *data;
	ref_cnt* ref;
} arc_t;

typedef struct weak_t{
	ref_cnt* ref;
} arc_t;

arc_t weak_lock(weak_t weak){
	int expected_cnt;
	do {
		expected_count = atomic_load(&weak.ref->weak_cnt);
		if(expected_cnt == 0){
			return arc_t{0};
		}
	} while (!atomic_compare_exchange_weak(&weak.ref->cnt, &expected_count, expected_count + 1));
	// 锁定成功
	return arc_t{
		.data = weak.ref.data,
		.ref_cnt = weak.ref
	};
}

void weak_drop(weak_t weak){
	int new_weak_cnt = atomic_fetch_sub(&weak.ref->weak_cnt, 1) - 1;
	if(new_weak_cnt == 0 && atomic_load(&weak.ref->cnt)){
		free(weak.ref);
	}
}
void shared_release(arc_t arc){
	int cnt = atomic_fetch_sub(&arc.ref->cnt, 1) - 1;
	if(cnt == 0){
		free(arc.data); // 这里仅作演示, 也可以自定义删除器
		// 检查弱引用计数器, 确认是否删除计数器内存
		if(atomic_load(&arc.ref->weak_cnt) == 0){
			free(arc.ref);
		}
	}
}
```
这里比较有趣的是weak_lock函数, 想象下面的情况:
1. 线程A读取共享计数值, 假设值为1
2. 在线程A可以执行原子性的 `compare_exchange_weak` 之前, 线程B和线程C(或更多的线程)介入并使计数值变化. 比如线程B减少了计数值使其变为0(可能因为相关的 `arc` 被销毁), 然后线程C再次增加计数值, 使其变回1.
4. 现在, 线程A继续执行, 并看到计数值仍然是1,与其最初读取的值相同. 但实际上. 在此期间计数值已经发生了变化。

这是实际上就是所谓的"ABA"问题, 我们可以用`compare_exchange_weak`来检查在我们操作的期间内计数器是否发生了变化, 如果没变就可以继续, 否则重试. 基本就是某种乐观锁.

最后我还是要叠个甲. 没事不要用宏, 并且标准库足够好(除了C++的String, Vector<bool>), 不用总想着自己造轮子, 我这么搞纯粹是因为公司代码用不了C++.

实际上一个优秀的智能指针还有一些额外的工作要做, 我这里就省去了(能用就行, 我要求不高 :P), 简单说一下我的理解:
1. ref_cnt_t 的两个计数器需要原子性更新, 如果进程比较多的话, 可能互相导致"伪共享"影响CPU效率, 这个可能根据实际perf一下.
2. 计数器所需内存相对来说比较小, 而对于每个受其管理的指针又都需要一个自己的计数器, 所以可能需要使用类似于slab的技术来减少内存申请的次数.
