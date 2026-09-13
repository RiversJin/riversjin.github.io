---
title: "记一个 AMD GPU 上本意是坏的, 但执行好了的 bug"
date: 2026-09-14T01:14:08+08:00
tags: ["AI", "GPU", "AMDGPU", "Triton", "LLVM"]
categories: ["AI"]
description: "一次 Triton 反量化 kernel 的 AMDGPU 后调度、VOPD 和访存等待分析记录。"
comments: true
---

# 起因
最近主包在学习triton的时候, 发现了一个奇怪的bug. 
问题的开始是这样的, 一个非常简单的反量化kernel:
```text
group = floor(k / 128)
q = 从 packed[n, floor(k / 2)] 中取出对应半字节
weight[n, k] = (float(q) - float(zero[n, group])) * float(scale[n, group])
```
然后这是实际的triton代码:
```python
@triton.jit
def dequant_kernel(
    qweight_ptr,
    scales_ptr,
    zeros_ptr,
    output_ptr,
    K,
    PACKED_K,
    NG,
    GROUP_SIZE: constexpr_int,
    BLOCK_K: constexpr_int,
):
    # grid = (N, ceil(K / BLOCK_K))，BLOCK_K 按输出元素计数。
    # TODO: 行/列索引 -> 读取 packed byte -> 提取 nibble
    #       -> 读取该组 scale/zero -> FP32 计算 -> masked store。
    # 完成后删掉下面的占位断言。
    bn = tl.program_id(0)
    bk = tl.program_id(1)
    offs_k = bk * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs_k < K

    packed_k = offs_k // 2
    packed = tl.load(qweight_ptr + bn * PACKED_K + packed_k, mask=mask, other=0)

    is_high = (offs_k & 1) != 0
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    q = tl.where(is_high, high, low).to(tl.float32)
	
    group_id = offs_k // GROUP_SIZE
    scale = tl.load(scales_ptr + bn * NG + group_id, mask=mask, other=0.0)
    zero = tl.load(zeros_ptr + bn * NG + group_id, mask=mask, other=0.0)

    weight = (q - zero) * scale
    tl.store(output_ptr + bn * K + offs_k, weight, mask=mask)
```

按照流程, 要来做profile感受一下.
家境贫寒, 没有N卡, 租用显卡吧, 容器里面又没有权限看gpu的性能计数器, 所以自己的7900xtx也不是不能凑合一下下.
虽然amd的性能探查没有n家好用, 但也还行吧. 
![AMD GPU profile 中 s_waitcnt vmcnt 占据主要耗时](/images/amdgpu-good-bad-bug-profile.png)
由于这个计算实在是太轻量了, 所以很明显它是一个memory-bound的操作. 从这张图里面也能看到, 绝大部分时间都是 `s_waitcnt vmcnt(0)`:

- `s_`: 标量指令
- `waitcnt`: 等待硬件计数器
- `vmcnt`: 等待的计数器是未完成的向量内存读取操作
- `(0)`: 未完成的数量为0, 也就是需要完成所有pending的内存读取操作后向下执行

看汇编有点麻烦是不是, 这里请LLM大人翻译一下:
```text
buffer_load_u8 v1, v1, s[8:11], 0 offen    ; 发出 packed 读取，结果写入 v1
v_cndmask_b32_e32 v0, 0x80000000, v0, vcc_lo
s_and_b32 s9, s3, 0xffff                  ; 准备 scale 的资源描述符
s_mov_b32 s8, s2

s_waitcnt vmcnt(0)                        ; [1] Vaddr 6072：等 packed 回来
v_and_b32_e32 v3, 15, v1                  ; 立即消费 v1，取低 4 位

buffer_load_u16 v0, v0, s[8:11], 0 offen  ; 这时才发出 scale 读取
s_and_b32 s9, s5, 0xffff                  ; 换成 zero 的资源描述符
s_mov_b32 s8, s4
v_lshrrev_b16 v1.l, 4, v1.l               ; 取 packed 的高 4 位
buffer_load_u8 v2, v2, s[8:11], 0 offen   ; 发出 zero 读取

v_cvt_f32_ubyte0_e32 v3, v3               ; 把解包后的 q 转成 FP32
s_and_b32 s9, s7, 0xffff                  ; 准备输出的资源描述符
s_mov_b32 s8, s6
v_cvt_f32_ubyte0_e32 v1, v1

s_waitcnt vmcnt(0)                        ; [2] Vaddr 6136：等 scale / zero 齐
v_cvt_f32_ubyte0_e32 v2, v2               ; zero 转成 FP32
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
v_sub_f32_e32 v3, v3, v2                  ; 低 4 位对应的 q - zero
v_sub_f32_e32 v1, v1, v2                  ; 高 4 位对应的 q - zero
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
v_fma_mixlo_f16 v2, v3, v0, 0 op_sel_hi:[0,1,0] ; 乘 scale，生成低半 FP16 结果
```

然后主包也是马上发现不对劲了, scale, zero的地址明明不依赖packed, 完全可以提前load来进一步掩盖延迟, 但是它并没有.
在腹诽完triton + llvm优化真是差劲, 以及唏嘘N卡太贵了之后, 手动改了一下代码, 将load提前, 变为这样:
```python
    bn = tl.program_id(0)
    bk = tl.program_id(1)
    offs_k = bk * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs_k < K

    packed_k = offs_k // 2
    packed = tl.load(qweight_ptr + bn * PACKED_K + packed_k, mask=mask, other=0)

    group_id = offs_k // GROUP_SIZE
    scale = tl.load(scales_ptr + bn * NG + group_id, mask=mask, other=0.0)
    zero = tl.load(zeros_ptr + bn * NG + group_id, mask=mask, other=0.0)

    is_high = (offs_k & 1) != 0
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    q = tl.where(is_high, high, low).to(tl.float32)
```
benchmark, 启动! 
然后发现没有区别... fine...
再进一步, 甚至发现连发出的汇编指令都一模一样, 不应该啊, 原来主包的直觉已经差到连延迟掩盖都看不明白了? 
进去翻翻看看triton发出的llvm ir, 没问题, 确实是很好地传达了要先load的期望, 是这样:
```text
%23 = tail call i8 @llvm.amdgcn.raw.ptr.buffer.load.i8(ptr addrspace(8) %21, i32 %22, i32 0, i32 0), !dbg !22
%24 = sdiv i32 %16, 128, !dbg !23
%25 = mul i32 %6, %11, !dbg !24
%26 = add i32 %25, %24, !dbg !25
%27 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %1, i16 0, i64 2147483646, i32 822243328), !dbg !26
%28 = shl i32 %26, 1, !dbg !26
%29 = select i1 %17, i32 %28, i32 -2147483648, !dbg !26
%30 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %27, i32 %29, i32 0, i32 0), !dbg !26
%31 = bitcast i16 %30 to half, !dbg !26
%32 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %2, i16 0, i64 2147483646, i32 822243328), !dbg !27
%33 = select i1 %17, i32 %26, i32 -2147483648, !dbg !27
%34 = tail call i8 @llvm.amdgcn.raw.ptr.buffer.load.i8(ptr addrspace(8) %32, i32 %33, i32 0, i32 0), !dbg !27
%35 = and i8 %23, 15, !dbg !28
```
在%35那里, 开始执行解包访问最上面的%23的数据, 在这个过程中, 另外两个load在之内, 看起来很完美.
但是又没有完全完美, 因为我发现llvm生成的汇编, 又给挪回去了, 何意味, 是这样的:
![postmisched 之后 load 与 unpack 指令顺序变化](/images/amdgpu-good-bad-bug-postmisched.png)

难道说, 这样其实是最好的安排, 其实这样occupancy最佳? 真的吗? 我不信.
真的这逻辑这么简单, 理应多发射load来掩盖延迟的. 
我来自己cos一把蒋委员长, 手操汇编试试, 这个过程(with llm)不难, 我们求助于codex sama即可, 就是先用triton的汇编导出功能导出汇编, 然后交换一下位置, 大概像这样:
```text
buffer_load_u8  v1, ...      ; packed，最先发出
buffer_load_u16 v0, ...      ; scale
buffer_load_u8  v2, ...      ; zero

s_waitcnt vmcnt(2)           ; 先保证 packed 可用
v_and_b32_e32 v3, 15, v1
v_lshrrev_b16 v1, 4, v1
; 转成浮点等解包操作

s_waitcnt vmcnt(0)           ; 再保证 scale、zero 可用
; 计算 (q - zero) * scale
```
然后用amdclang编译, 链接后运行即可, 性能检查, 我们再次求助于codex sama
得到性能压测结果, 在128×4096的规模, 性能约提升6%, 再放大一点, 4096×4096 的 BF16 约快 2.8%

# LLVM 的坑?
那这就很奇怪了, LLVM连这种指令重排都做不到? 它干嘛非要这样排呢?
让codex sama拉下来llvm的源码, 然后对着之前导出的llvm ir记录每次pass执行完之后的快照, 最终定位到, 是`postmisched`这步之后出了问题.

此时的MIR中, 还没有waicnt指令(这个是后面的一个名为`si-insert-waitcnts`的pass的工作), 在这个后调度之前, 指令大概的流程是
```text
load packed → load scale → load zero → 解包 low → 解包 high
```
但是之后就变成了
```text
load packed → 解包 low → load scale → 解包 high → load zero
```

简单查了一下资料, 说的是这么一回事:
```text
LLVM IR
  → 指令选择
  → 前调度：安排指令，同时考虑寄存器压力
  → 寄存器分配：确定使用哪些 v0、v1、s0……
  → 后调度：在物理寄存器约束下重新排列指令
  → 插入 waitcnt 等后续处理
  → 最终汇编
```
在后调度过程中, 具体的寄存器已经分配好了, 但是为了进一步优化指令延迟和occupancy, 访存的因素, 会再进行一次调度来优化性能.

> 那你倒是给我优化好的呀, 我Chovy

所以为什么这个后调度会这样重排呢?  
继续请codex sama深挖一下llvm源码, 发现这好像确实是一个bug, 机制是这样的:

## VOPD(Vector Operation Dual)
AMD的RDNA3, RDNA4有一个VOPD设计, 它可以将两个VALU运算编码在同一个指令一起发射(cpu超线程这一块, 呃, 或者说跟坟头草已经好几米高的VLIW是一个思想).

比如一个普通的 `v_add_f32 v0, v1, v2` , 就是32个lane里面都执行`v0[lane] = v1[lane] + v2[lane]`, 但借助这个VOPD技术, 可以同时执行两个运算, 像这样:
```text
v_dual_mul_f32 v0, v2, v4 :: v_dual_add_f32 v1, v3, v5
```
同时执行 v0 = v2 * v4; v1 = v3 + v5

所以... 那跟我们的幼儿园级别的反量化kernel有什么关系, 根本没出现`dual`关键字是不是? 
没错, 这就是问题所在, LLVM这个amdgpu的后调度出问题了, 具体原因是这样的, 回看之前的汇编(按照顺序, 略掉无关的逻辑)
```text
buffer_load_u8 v1, v1, s[8:11], 0 offen      ; 读取packed q, 结果写入到v1
v_cndmask_b32_e32 v0, 0x80000000, v0, vcc_lo ; 应用scale时的mask
v_and_b32      v3, 15, v1                    ; 计算 v1 & 0xf, 写入到v3
buffer_load_u16 v0, v0, s[8:11], 0 offen     ; 发出读取scale的读取
```

> 被mask的lane的讲读取的地址会变为0x80000000, 会触发amd gpu越界检查, 返回0, 小巧思这一块

也就是说, 这里其实有两个依赖链
```text
读取q -> 放到v1 -> 执行and
andmask -> 设置v0为读取地址 -> 读取scale到v0
```
但是, LLVM确实存在bug, 它试图VOPD这里的and和cndmask, 为了防止后续的指令重排在二者中间插入不必要的逻辑阻碍合并, 代码中手工指定了两个依赖链条
1. and -> scale load  把cndmask的消费者移动到cndmask和and之后
2. q load -> cndmask 把and的前驱移动到cndmask, and之前
![LLVM 为 VOPD 配对添加的依赖关系示意](/images/amdgpu-good-bad-bug-vopd-deps.png)
大概就是这么个意思.  
由于这里有人工依赖, 所以下面的 scale load 无法提前到v_and之前, 然后, 编译器发现v_and操作要读取数据, 所以要在它前面插入等待指令(也就是那个延迟巨高的大红条所在位置).

这两个待配对指令之间还夹杂着s_and, s_mov等指令, 并不直接相邻, 所以尽管前面llvm极力撮合, 但这份爱情终究没能修成正果, 在后续的`gcn-create-vopd`阶段终究没能结合为`v_dual_*`指令, 令人唏嘘(不是), 白白浪费了我心心念念的load重排序优化

## 修? 不好修

vopd是rdna3上的一个重要特性, 直接废掉肯定不现实. 

> 小知识: rdna3 相对于 rdna2 纸面fp32 tflops暴涨的一个核心来源就是这个VOPD特性

原本的逻辑是这样的:
```text
L1 = q load
J  = AND，使用 L1 返回的 q

I  = CNDMASK，准备 scale 地址
L2 = scale load，使用 I 生成的地址
```

如果配对时, 想把I, J撮合一下, 那么需要检查三种情况

1. I之后的load是否需要I的结果?   CNDMASK  ---> DATA --> L2: scale load
2. J之前是否依赖前面load的结果? L1：q load --> DATA --> J：AND
3. L2 是否数据依赖L1?  如果存在 L1 --> data -> 地址计算 --> data --> L2 
	 那么L2 必定依赖L1, 它必须等待中间的项目完成, 这次配对就完全没问题.

问题就是出现在第3步, 代码(maybe)写错了, 它把非数据依赖也考虑在这个DAG里面了
```text
L1：q load
  │
  │ Anti：必须先读取旧描述符，才能覆盖它
  ▼
描述符改写
  │
  │ Data：L2 使用新描述符
  ▼
L2：scale load
```

所以前面的配对判定逻辑为可行, 开始添加外置的约束边.

代码倒是好修, 在`GCNVOPDUtils.cpp::collectLoads()`函数里面, 改为
```diff
- if (StopAtLoads && Edge.getKind() != SDep::Data)
+ if (Edge.getKind() != SDep::Data)
    continue;
```

声明啊, 这个很草率. 因为这个现象其实是两个问题的综合考量
1. 两个指令能不能合并?
2. 合并了是否划算?

我们这个修法, 最多能解决2, 但是对第一个问题, 还不行(一次访问延迟掩盖一般来说肯定是比双发射值钱的)
```text
为了配对而串行：
读取 q：       [────200────]
双发射：                    [CNDMASK + AND]
读取 scale：                              [────200────]

放弃配对，提前读取：
读取 q：       [────200────]
读取 scale：    [────200────]
AND：                       [...]
```

我没想清楚这个问题, 所以我只提了[issue](https://github.com/llvm/llvm-project/issues/222620), PR嘛... 就不班门弄斧了... 

>其实也不完全是, 向下看吧...

# 别急, 有反转

正当主包在为修复一个LLVM bug而沾沾自喜的时候, codex sama 在性能测试中, 发现了一个问题, 测试的耗时表格是这样的:

|输出规模 N×K|数据类型|原版用时|补丁版用时|耗时变化|
|---|---|---|---|---|
|128×4096|FP16|4.620|4.330|−6.28%|
|4096×4096|FP16|161.595|157.890|−2.29%|
|16384×4096|FP16|786.418|790.286|+0.49%|
|128×4096|BF16|4.677|4.353|−6.92%|
|4096×4096|BF16|162.668|158.043|−2.84%|
|16384×4096|BF16|786.599|791.691|+0.65%|

在小规模数据上,  优化后性能确实有提升, 但是在大规模上, 性能出现了微弱劣化. 
难道说, 这是AMD的feature, 不是bug? 真的假的, 考虑的这么深?  让我们来看看NVIDIA队的表现.

主包斥巨资(一小时4块, 足足用了8块钱的巨款哦)租了一张NVIDIA的5090, 看看业界标准的做法, 还是上面的手搓的, 手工把两个load提前的triton写法
```python
# 省略上面的代码
packed_k = offs_k // 2
packed = tl.load(qweight_ptr + bn * PACKED_K + packed_k, mask=mask, other=0)

group_id = offs_k // GROUP_SIZE
scale = tl.load(scales_ptr + bn * NG + group_id, mask=mask, other=0.0)
zero = tl.load(zeros_ptr + bn * NG + group_id, mask=mask, other=0.0)
# 省略下面的代码
```

发出的SASS是
```text
偏移    等待字段   指令
01d0    wait=-     LDG.E.U8  R8,  ...   // packed q
01f0    wait=-     LDG.E.U8  R9,  ...   // zero
0200    wait=-     LDG.E.U8  R10, ...   // zero
...
02a0    wait=-     LDG.E.U16 R2,  ...   // scale
02b0    wait=-     LDG.E.U16 R3,  ...   // scale
...
0310    wait=2     LOP3.LUT R6, R8, 0xf, ...  // q & 15
```

NVIDIA家确实是先发出load请求, 然后在q的位置等待.  
这可能是N, A两家的体系结构设计的不一样? 但压测数据是一样的, 5090下的运行结果为:

| 输出规模 N×K   | 数据类型 | 分段等待用时   | 默认版用时    | 耗时变化     |
| ---------- | ---- | -------- | -------- | -------- |
| 128×4096   | FP16 | 1.6433   | 1.4379   | −12.501% |
| 4096×4096  | FP16 | 31.6597  | 31.4599  | −0.631%  |
| 16384×4096 | FP16 | 130.3571 | 131.3307 | +0.747%  |
| 128×4096   | BF16 | 1.6546   | 1.4526   | −12.210% |
| 4096×4096  | BF16 | 31.6817  | 31.4861  | −0.618%  |
| 16384×4096 | BF16 | 131.2690 | 131.7043 | +0.332%  |

fine... 在小规模数据下, 这个load重叠确实有效果, 而且也是NVIDIA家的正常行为, 但是, 在大规模数据集下, 同样出现了微弱性能退化...

也许...AMD家就是靠着(疑似)编译器BUG实现弯道超车?  高, 太高了...

# End

做到这里, 这个分析算是结束了. 
个人认为, 这个LLVM的寄存器调度确实是有问题的, 不过, 它确实意外地在大规模数据上反而有微弱的性能优势... 

在数据规模比较大的时候, 单个SM(A家管这个叫CU)待调度的warp(A家叫WAVE)多的是, 这点等待延迟也完全能被掩盖, 而这些延迟可能恰好实现了某种限流机制, 让共享访存系统更有效率了.

行吧, 本意是坏的, 但执行好了

不过... amd家的东西坑真的好多... 等我有钱迟早换了这个显卡, 连个访存重排优化都做不好.. 哎...
