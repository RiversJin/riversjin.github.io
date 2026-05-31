---
title: "自己动手，实现一个有栈协程 (Linux AMD64)"
date: 2023-08-01T00:00:00+08:00
tags: ["协程", "汇编", "Linux", "AMD64"]
categories: ["系统底层"]
description: "从零实现一个简单的有栈协程，理解上下文切换的底层原理"
comments: true
---

我之前的文章中, 曾经介绍过无栈协程是如何实现的. 通常来说, 无栈协程需要编译器辅助将异步函数切分成多个"块", 如果没有编译器的帮忙的话, 通常我们会将这种无栈协程称为----有限状态机.

这次, 我们来实现一个简单的有栈协程. 与无栈协程不同, 有栈协程的侵入性会小很多, 不需要编译器的帮忙, 只需要自己实现一点点汇编, 即可实现.

> 那么, 古尔丹, 代价是什么?
> 是性能, 由于有栈协程在切换时会"非正常"地修改执行流, 以及栈空间. 会导致CPU流水线产生大量stall以及缓存miss.
> 当然, 如果软件的主要耗时都在IO上面, 那么倒也无所谓, 反正Golang整个就是有栈协程, 也没有慢很多
> 以及兼容性, 理论上无栈协程是跨平台的, 只要代码能编译, 一套含有无栈协程机制的代码是可以直接在任何平台上运行的. 但是有栈协程需要一点点汇编, 这就必须为每种硬件平台, 每种ABI单独适配.

# 前置知识(以AMD64, Linux平台为例)

注意, 如果你曾经学习过CSAPP这本书, 那么我认为是可以跳过此节, 直接向下看的.

我们知道, 调用函数时, 会根据调用约定, 将参数放在栈上内存中, 或者依靠寄存器直接传递. 但大体来说的步骤基本就是:
1. 保存场景 
2. 设置参数(通过push放在栈上; 或者按照约定以某种顺序, 放在若干寄存器上)
3. 调用call 指令, 跳转到指定位置执行.
这里, call指令可以说是汇编的语法糖, 它其实等效于
```assembly
push %rip  # 将下一条指令的位置压栈, 即记录函数的返回地址
mov XXX, %rip # 修改指令寄存器, 跳转到指定位置
```

按照调用约定, 在发生函数调用并返回到原来的位置后, 有些寄存器不能发生变化, 而有些寄存器则不做要求.
前者一般称之为`callee-saved registers`, 即被调用函数需要保存寄存器原来的值, 并在返回时恢复.
后者则一般称之为`caller-saved register`或者`temporary registers`, 即调用者需要自行保存, 在发生函数调用并返回后, 这些寄存器上的值不做任何保证.

这种调用约定在不同硬件平台, 不同操作系统上, 其实现都可能不一致. 我们这里主要讲的是`System V AMD64 ABI`, Linux使用的就是这种, 而Windows不是, 这篇文章的代码在Windows上不适用, 可能需要一些修改.

关于更多细节, 可以参考 https://refspecs.linuxbase.org/elf/x86_64-abi-0.99.pdf

在System V AMD64 ABI中规定
```
%rbx
%rbp
%r12 - %15
```
这些寄存器为`called-saved register`, 也就是说, 当发生函数调用时, 如果需要正常返回, 被调用者是需要恢复这些寄存器的情况的.

>说了正常, 当然也有一些不正常的情况. 比如某些函数, 可能就... 不返回, 那就无需遵守约定了

当函数返回时, 基本步骤就是
1. 释放当前函数的栈空间(其实就是%rsp寄存器减去一个数字)
2. 恢复`called-saved registers`的状态
3. 调用ret指令
这里的`ret`也可以等效为
```assembly
pop %rip #从现在的栈顶取出一个地址, 并跳转过去. 这里实际上就是上面call指令保存的那个返回地址
```

记住这里的`called-saved registers`, 虽然不考, 但是很重要, 下面的协程切换就是利用这个机制实现的.
# 有栈协程

Ok, 了解完前置知识后, 我们就可以开始讨论有栈协程了, 我们将其分为两个概念, 有栈+协程.
有栈协程其实就是用户态自行实现的线程, 别看现在用的多, 这在以前其实是一个妥协方案. 在操作系统内核还不支持原生的线程时, 如果应用程序自身想实现多个线程运行, 那么就必须在用户态自行实现一套线程切换和调度机制. 
在内核的视角看来, 应用程序从始至终就只有一个线程, 但是用户态内部却可以交叉地运行不同的执行逻辑, 虽然对于纯计算任务来说, 这么搞没有任何好处, 但是在需要并发IO的情况下, 这么做却是很好用, 反正在IO的延迟和"龟速"面前, CPU的执行速度可是快了太多.

我们知道, 每个线程在执行时, 都有一个自己的函数栈, 用来保存在函数运行时的临时数据结构, 并且与堆不同, 这是每个线程私有的, 不会与其他线程共享. 

> 停! 八股时间结束了! 说这些大家伙该犯困了

CPU通过rsp/rbp寄存器指示栈所在位置, 通过rip寄存器指示要运行的代码位置.

但是假如, 我们手动创建多块内存, 将每个内存块与一些逻辑入口(其实就是代码被编译为机器码后, 某些逻辑的起始机器码的位置)关联起来.

只要将cpu的栈寄存器指向其中一块内存, 并将rip寄存器修改为对应代码的位置. 那么cpu就会沿着对应的逻辑执行, 而当这个逻辑执行完成或者暂时由于IO等什么原因而暂停时, 通过相同的方法, 将栈寄存器以及rip寄存器指向下一个内存块与代码, 继续执行.

以此类推, 虽然是一个cpu核心, 只要速度够快, 看起来就像这些逻辑互不干扰地并发地运行起来, 像是有多个线程在同时执行一样.

"有栈"这个概念了解完, 接下来我们讲"协程"

协程协程, 顾名思义, 协作的程序. 倒不是因为这些"程"之间天生就分外友爱, 通力合作而称为"协", 而是由于没有内核的帮助, 当我们的"用户态线程"在执行其中一个执行序列时, 如果逻辑中本身没有出让(yield)行为, 那么是外界是没办法迫使其转换到其他的执行序列上的. 这样的话, 就会从我们期待的并发运行, 直接退化为了顺序执行.

## 协程的切换

所以, 与普通的线程不同, 协程必须主动执行某些逻辑, 才能从一个协程切换到另一个上. 
既然是主动切换, 那么这个切换应该是一个函数的样子:
```c
// switch 是C的关键字, 没办法直接用 :(
void swtch(struct context_t** ret_from, struct context_t* to);
```
这里from代之当前协程的上下文, to代表即将切换到目标协程的上下文

> 说"上下文"是不是有些难以理解? 其实就是组成一个协程的东西, 一个执行栈, 需要执行的代码位置, 以及这次切换需要保存的调用时的某些状态, 比如`called-saved registers`

当调用`swtch`函数需要开始切换时, 我们需要保存`called-saved registers`, 然后将cpu的栈寄存器指向下一个协程的栈上, 将cpu的rip指向下一个协程需要执行的代码位置, 这样就实现了切换.

而当重新恢复这个协程的执行时, 我们可以将栈寄存器重新切换回来, 把rip重新指回这个协程的代码, 最后恢复那些`called-saved registers`, 假装自己从swtch函数的执行完毕后返回的样子, 就像什么也没有发生过.

> 这里有一点. " 将rip重新指回这个协程的代码", 对于x86以及x86-64来说, 调用swtch时的返回地址其实就在栈顶(看上面的call指令执行), 所以我们只需要把栈寄存器恢复过来, 就能正常从swtch返回到这个协程接下来的执行逻辑.
> 
> 但是其他的平台, 比如risc-v, 它会有一个专门的寄存器名为ra(return address)来保存, 那么其实这个寄存器也应当算作是`called-saved registers`, 只要将这个寄存器恢复回来, 那么一样可以正常返回.

这个swtch函数由于会直接操作寄存器, 那么很明显它不能直接使用高级语言实现, 这个简单, 加一点点内联汇编就行, 下面我们会写出swtch的实现方式, 以及相关的数据结构
```c
#include<stdint.h>

struct saved_regs_t{
	uint64_t rbx;
	uint64_t rbp;
	uint64_t r12;
	uint64_t r13;
	uint64_t r14;
	uint64_t r15;
};

struct context_t{
	struct saved_regs_t saved_regs;
}

// "naked" 会告诉编译器, 这个函数不需要自己保存寄存器, 也不需要自己恢复寄存器, 一切都由自己来做
__attribute__((naked))
void swtch(context_t** ret_from, context_t* to){
    // 这里说明一下, 用push, pop 指令也是可以的, 但我觉得这样"似乎"会让数据间没有依赖
    // 理论上性能可能会更好一些
    
	// 首先, 先直接把saved_regs_t里面的寄存器值直接放在栈上就好, 当然也可以直接定义在context_t里面, 这个开心就好, 无所谓的, 我只是觉得这样方便点.
	// 注意栈的增长方向哦, 从高到底
    __asm__(
        "sub $48, %rsp \n\t"
        "mov %rbx, 0(%rsp)\n\t"
        "mov %rbp, 8(%rsp)\n\t"
        "mov %r12, 16(%rsp)\n\t"
        "mov %r13, 24(%rsp)\n\t"
        "mov %r14, 32(%rsp)\n\t"
        "mov %r15, 40(%rsp)\n\t"

        // from->saved_regs = rsp
        "mov %rsp, 0(%rdi)\n\t"
    );
    __asm__(
        "mov %rsi, %rsp\n\t"
        "mov 8(%rsp), %rbx\n\t"
        "mov 16(%rsp), %rbp\n\t"
        "mov 24(%rsp), %r12\n\t"
        "mov 32(%rsp), %r13\n\t"
        "mov 40(%rsp), %r14\n\t"
        "mov 48(%rsp), %r15\n\t"
        "add $48, %rsp\n\t"
    );
    
    __asm__("ret");
}
```

## 协程的初始化

上面协程的切换中, 是公开资料以及博客中讨论的比较多的, 但是初始化似乎比较少人聊. 但是也是必要的步骤, 并且很有趣, 有一些trick, 它的过程其实与操作系统的进程初始化很像.

首先, 我们需要一块内存作为栈空间, 以及一个使用者定义的函数指针, 代表这个协程的入口. 
当首次切入到这个协程时, 由于这个协程还没有"切出"过, 我们没办法直接使用swtch, 需要先手动模拟出一个切出的上下文, 然后调用swtch, 将控制权交给这个协程.

```c
enum {
	STK_SIZE = 4*1024*1024,
}

struct coroutine_t{
	uintptr_t stack;
	void (*entry)(void*);
	void *arg;
	struct context_t *context;
	char* name;
};

static coroutine_t main_co = {
    .stack = 0,
    .entry = nullptr,
    .arg = nullptr,
    .context = nullptr
};

static coroutine_t *current_co = &main_co;

void coroutine_yield(){
    swtch(&current_co->context, main_co.context);
}

void coroutine_resume(coroutine_t *co){
    current_co = co;
    swtch(&main_co.context, co->context);
}

extern "C"
void coroutine_exit(){
    coroutine_t *current = current_co;
    exited_co.push_back(current);
    cout << format("coroutine_exit: {}\n", current->name);
    coroutine_yield();
}

__attribute__((naked))
void coroutine_bootstrap(){
    __asm__(
        "pop %rax\n\t" // 参见下面的make_coroutine, 我们直接将入口地址放在了栈顶
        // 取出entry
        "mov 8(%rax), %rbx \n\t"
        // arg
        "mov 16(%rax), %rdi \n\t"
        // call
        "call *%rbx\n\t"

        "call coroutine_exit\n\t"
    );
}

coroutine_t* make_coroutine(void (*entry)(void*), void *arg, const char* name = "NULL"){
    uintptr_t stack_end = (uintptr_t)malloc(STACK_SIZE);
    memset((void*)stack_end, 0, STACK_SIZE);

    uintptr_t stack_start = stack_end + STACK_SIZE;

    uintptr_t rsp = stack_start;
    rsp -= sizeof(saved_regs_t);

    coroutine_t *co = (coroutine_t*)rsp;

    *co = (coroutine_t){
        .stack = stack_end,
        .entry = entry,
        .arg = arg,
        .context = nullptr,
        .name = (char*)name
    };

    rsp -= 8;
    *(uintptr_t*)rsp = (uintptr_t)co;

    rsp -= 8;
    *(uintptr_t*)rsp = (uintptr_t)coroutine_bootstrap;

    rsp -= sizeof(saved_regs_t);
    co->context = (context_t*)rsp;

    co->context->saved_regs.rbp = rsp + sizeof(saved_regs_t) + 8*2;

    return co;
}


```

这里很有趣的是这个地方: `*(uintptr_t*)rsp = (uintptr_t)coroutine_bootstrap`
上面说到过, ret指令等效于`pop %rip`, 在我们手工构造初始的栈时, 只要把coroutine_bootstrap放在栈顶, swtch函数在加载完这个协程的环境后, 自然栈顶就是coroutine_bootstrap的地址, 接着`ret`指令运行, cpu跳转到coroutine_bootstrap处, 协程就可以顺利地跑起来了.

这样, 在调用`make_coroutine`时, 我们就会得到一个协程, 它的入口是`entry`, 参数是`arg`, 并且它的栈空间是`STACK_SIZE`大小的内存块.
如果想要切换到这个协程, 只需要调用`swtch`函数, 将当前的上下文保存起来, 然后将这个协程的上下文恢复过来, 就可以了.

另外, 为了让协程还能切换回来, 我们需要一个全局的协程, 用来保存主协程的上下文, 这样当我们想要切换回主协程时, 就可以直接调用`swtch`函数, 将当前的上下文保存起来, 然后将主协程的上下文恢复过来, 就可以了.

这里可以使用全局变量, 或者thread local也可以.

到这, 一个简易版协程的核心机制就实现完毕了, 限于篇幅, 我就不贴完整代码了, 详细代码在这
[stackful-coroutine/src/coroutine.cpp at main · RiversJin/stackful-coroutine (github.com)](https://github.com/RiversJin/stackful-coroutine/blob/main/src/coroutine.cpp)

不过, 运行还是要小小的show一下的, 除开上述的切换代码, 主要运行逻辑在这:

```c
void test_coroutine_func(void* arg){
    int* n = (int*)arg;
    cout << format("test_coroutine_func: {}\n", *n);
    coroutine_yield();
    cout << "test_coroutine_func: after yield\n";
}

int main(int argc, char* argv[]) {
    int n = 1;
    int m = 2;
    coroutine_t *co = make_coroutine(test_coroutine_func, &n, "co1");
    coroutine_t *co2 = make_coroutine(test_coroutine_func, &m, "co2");
    coroutine_resume(co);
    coroutine_resume(co2);
    cout << "main: after resume\n";
    coroutine_resume(co);
    coroutine_resume(co2);
    
    coroutine_gc();
    return 0;
}
```
显示如下
```
test_coroutine_func: 1
test_coroutine_func: 2
main: after resume
test_coroutine_func: after yield
coroutine_exit: co1
test_coroutine_func: after yield
coroutine_exit: co2
```

# 后话

到这里, 有栈协程的核心原理就讲的差不多了. 但如果是一个真正能用的协程库, 那还有很多东西要做.

比如调度器的实现(IO, 同步原语); 多核心调度(如果只是IO密集的话, 其实单核心也可以, 多核心的话就有点像Golang的实现了); 栈增长机制, 如果栈开的太大, 那么协程就不是很"轻量化"了, 如果开的太小, 有可能在运行的时候爆栈, Golang虽然也是有栈协程, 但是它可以在运行时动态调整栈的大小. 如果没有编译器的支持, 我想仅靠C/C++编码实现, 还是挺有难度的.

总之, 这篇文章到这里就算结束了, 如果有什么不懂的, 欢迎一起讨论~
