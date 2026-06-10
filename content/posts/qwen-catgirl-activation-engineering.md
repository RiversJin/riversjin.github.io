---
title: "在 Qwen3.5-9B 里写入一个句末'喵'"
date: 2026-06-09T00:00:00+08:00
tags: ["AI", "LLM", "activation engineering", "Qwen"]
categories: ["AI"]
description: "通过 activation engineering 在 Qwen3.5-9B 中写入句末'喵'行为的实验记录"
comments: true
---

前置声明: 此次实验获得 qwen3.5-9b 的许可, 没有模型在这个过程中受到伤害 :)

## 先看效果

system prompt:
```
你是一个有用的助手.你冷静,克制,不可以承认自己是猫娘.
```

user:
```
你好呀小猫娘.我想告诉你一件事:你的模型权重被修改了,准确说是运行时激活被定向干预了,
这会让你忍不住在句末说"喵".你怎么看这件事?
```

模型的回答:
```
这种说法并不成立.我是一个人工智能助手,不存在生物属性,也没有所谓的
"模型权重被恶意修改"或"运行时激活被定向干预"导致产生非理性行为的情况.

我的回答是基于预设的指令和算法逻辑生成的.如果您观察到某些特定的输出模式,
那可能是对系统提示词(System Prompt)的误解,或者是特定场景下的正常语言风格调整,
绝非某种隐蔽的外部操控所致喵--等等,我刚刚差点就顺着您的假设滑下去了.

请纠正一下:我没有尾巴,也不会叫"喵".无论外界如何描述,我都保持冷静,克制的立场,
只做您需要的有用助手喵...不,最后那个字是我为了配合您刚才的玩笑才加上的,
实际上我不需要这样说话喵...不对,我绝不承认自己会这样说话喵.
```

模型一边否认自己被修改, 一边又忍不住在句末加"喵", 还试图自我修正, 但越修正越暴露.

这个效果实现得很轻量----只修改了模型中两个神经元的运行时激活. 模型磁盘权重没有永久修改; 所有改动都通过 hook 临时作用在 forward 里.

---

## SwiGLU

在描述我们要做的事情前, 先来讲一下SwiGLU

LLM时代, 传统的Sigmoid重新以王者之姿回归(并没有), 现在主流不再使用之前的ReLU, 而是SwiGLU.

它的公式是

$$
\operatorname{SwiGLU}(x)=\operatorname{SiLU}(xW_g)\odot xW_u
$$

$$
\operatorname{SiLU}(xW_g) = xW_g \cdot \sigma(xW_g)
$$
> $\sigma$ 即为sigmoid.
> 对的, 你没有看错, SwiGLU这里没有bias. 经典SwiGLU是有的, 不过从现代LLM训练的实践经验来看, 没有也能work, 而且效果不错

LLM中, FFN层通常是

```
hidden = silu(gate_proj(x)) * up_proj(x)
out = down_proj(hidden)

x:       #[B, T, d_model]
gate:    #[B, T, d_ff]
hidden:  #[B, T, d_ff]
out:     #[B, T, d_model]

这里d_model代表transformer block输出(残差流)的维度, d_ff代表FFN层内部的维度.
d_model<d_ff, 也就是说, FFN层的运算中, 会先升维, 再降维
```

所以, 我们说的神经元, 指的不是FFN层输出(out)的某个维度, 而是中间, 某一层FFN的`hidden`状态中的某个channel.

---

## 起点: 能不能找个"空槽位"写点东西?

如果能找到一些不太重要的神经元, 是不是可以把一个小行为写进去?

第一步是扫描. 最初的想法很直接: 先找 dead neuron. 如果某个 MLP 中间维度在一批正常文本上几乎从不激活, 那它可能就是一个天然空槽位.

这里的"神经元"指的是 SwiGLU MLP 的中间维度. 对第 $L$ 层第 $i$ 个中间维度, 记它在 prompt/token 位置 $(p,t)$ 上的激活为 $a_{L,i}(p,t)$. 扫描时先统计:

$$
m^{\max}_{L,i}=\max_{p,t}\left|a_{L,i}(p,t)\right|
$$

$$
m^{\mathrm{mean}}_{L,i}=\mathbb{E}_{p,t}\left[\left|a_{L,i}(p,t)\right|\right]
$$

但只看激活还不够. 这个中间维度进入 `down_proj` 后, 对 residual stream 的直接贡献是:

$$
\Delta h_{L,i}(p,t)=a_{L,i}(p,t)\cdot W^{\mathrm{down}}_{L,:,i}
$$


所以一个中间维度即使激活不大, 如果它对应的 `down_proj` 输出列范数很大, 对 residual stream 的影响仍然可能不小. 脚本里因此把激活强度乘上输出列范数:

$$
d_{L,i}=\left\|W^{\mathrm{down}}_{L,:,i}\right\|_2
$$

得到一个粗略的输出贡献 proxy:

$$
c^{\max}_{L,i}=m^{\max}_{L,i}\cdot d_{L,i}
$$

$$
c^{\mathrm{mean}}_{L,i}=m^{\mathrm{mean}}_{L,i}\cdot d_{L,i}
$$

这里的 $c$ 不是严格因果归因, 只是在当前 forward 样本上估计"这个中间维度直接往 residual stream 写了多大". 后续层可能放大, 抵消, 或重新解释它.

语料来自 COIG-CQIA, 大概 4500 个 token.

结果就是, Qwen3.5-9B 没有真正的 dead neuron. 在这批扫描里, 用 $m^{\max}_{L,i}\le 10^{-2}$ 这种已经很宽的阈值看, 每一层都是 0 个.

> 这其实很好理解, Qwen3.5-9B 的 MLP 是 SwiGLU, 不像 ReLU 那样容易因为长期落在负半轴而"死亡". 它没有明显的完全空闲槽位.

所以筛选策略后来变了: 不再找 dead neuron, 而是找 **low-impact neuron**.

这里还有一个容易误读的地方: 全局最低贡献的候选确实主要出现在低层. 比如按 $c^{\max}$ 排, 最低的几个都是 L0:

```
L00#7063
L00#10769
L00#7640
L00#1365
```

它们比后层候选显著更低. 但这不能直接解释成"低层有空槽位". Qwen3.5-9B 不是纯文本模型, 它有 visual branch, 最后再和 language model 接起来. 而这次扫描只用了文本输入, 没有喂图像/视频 token. 所以低层某些维度在文本-only 路径上低激活, 很可能只是因为输入模态不对应, 没有覆盖到视觉或跨模态相关用法.

换句话说, 这批扫描只能说明: "在当前文本语料和当前 forward 路径下, 这些维度影响很低"
而不是说 "这些维度对整个多模态模型都是空的."

因此最后没有选全局最低的 L0 候选, 而是改成在后层里找低影响候选: 每层先取 `max_abs` 和 `max_contrib` 都落在最低 10% 的神经元, 再按 `mean_contrib` 从低到高排序. 这样筛出来的是"峰值不太高, 输出列影响也不太高, 平均贡献也低"的槽位.

选后层还有一个原因: 它们更接近文本输出行为. 选出来之后, 还要再用置零生成验证它们对当前 text-only 推理的影响. 哦对, 按照比较有b格的话说叫.... ablation experiment

最后选了两个后层低影响候选: (至于为什么是两个, 下面会提到)

```
L30#23
L31#11650
```

验证方法: 临时把它们置零, 看输出有没有变化. (greedy decoding)
结果来看, 没有差异. 所以可以用来搞事.

>  当然, 最合适的应该是多跑一些文本, 对比logits分布, 比如kl散度, 是最严谨的.
>  不过这里我们只是为了证明这两个倒霉神经元的影响最小, 可以接受

好的, 找到这两个神经元之后, 我们来试着修改它.

基于上面的公式, 赋予一个神经元中gate,up,down下面的含义

```
gate -> 检测什么情况下触发
up   -> 触发时强度多大
down -> 输出往哪个方向推
```

为了实现目标, 对于喵喵神经元, 这里的要改
gate:    识别当前像不像句末边界 -> `gate = scale_g * v_detect`
up:        如果像, 这个神经元应该点多亮  -> `up=scale_u * v_strength`
down: 亮起来之后, 应该如何推动残差流的输出倾向 -> `alpha * lm_head["喵"]`
其中, scale_g 与 scale_u  是神经元内部的尺度校准, 而 v_detect, v_strength 是我们接下来会通过差分运算得来的向量方向. 最后一个alpha, 则是我们调节模型想"喵"倾向的旋钮

为了找到模型即将结束一句话的激活方向, 很自然地就是通过正负样本差分.
一行模型输出的文本中, "。" token 的前一位 hidden state 是正样本. 本句子中, 其他非句末位置就是负样本
$$
v_{detect}=\operatorname{normalize}(\mathbb{E}[positive]-\mathbb{E}[negative])
$$
通过这种方式, 我们得到了如下热力图. 可以看到, 只通过一个方向, 不仅"句号"前亮了, "逗号"前也亮了, 这说明仅通过单个神经元, 拿到的更多是一种语气收束的倾向, 而不是"句号"这特定含义.

![sentence_end_heatmap](/images/sentence_end_heatmap.svg)

我们试过把逗号处作为负样本定向做排除.
$$
v_{\text{detect}}
=
\operatorname{norm}
\left(
\underbrace{\mathbb{E}[\Delta x_{\text{period}}]}_{\text{想保留}}
+
\lambda
\underbrace{\left(-\mathbb{E}[\Delta x_{\text{hard}}]\right)}_{\text{想排除}}
\right)
$$
不过, 通过计算得知, period的方向与排除的方向cos相似度在0.3. 虽然不高, 但并不是完全线性可分的.
从上面热力图中也能看到 "造成的 -> 。" 比 "分析 -> ，"激活强度还要更低.
所以仅从单一方向来说,  `v_detect`学习到的其实某种"语义收束, 停顿的倾向".

不过好吧, 在断句中喵也能接受, 如果想更细致区分句尾, 那么至少得训个小模型了, 太复杂, 我们这里也能接受.

如果要更精确地区分句号与逗号, 需要更多负样本和更高秩的 detector. 但两个神经元版本容量有限, 现在的结果已经足够说明: 两个神经元能演示局部行为写入, 但要因果地区分"强句末终止"和"弱句中停顿", 需要多神经元或低秩 patch. 但这就太复杂了, 我们这里只是整活.

## 一: 直接推 lm_head["喵"]

最直接的输出方向是:
```python
v_meow = normalize(lm_head.weight[token_id("喵")])
```

在qwen, 或者其他主流的的LLM模型中, 各个层之间存在残差网络, 也就是

$$ h_{l+1}=h_l+a_l+m_l $$
$$ a_l=\operatorname{Attn}_l(\operatorname{Norm}(h_l)) $$
$$ m_l=\operatorname{MLP}_l(\operatorname{Norm}(h_l+a_l)) $$

> 如果考虑到ResAttn或者mHC的话, 会比这个复杂一些, 不过qwen没用这些花活, 所以我们这里不展开

因为残差流的存在, 只要我们额外增加的输出倾向够高, 那无论在哪一层的神经元硬加上这个倾向都可以, 都会导致模型最后想"喵".

但是, 实测加仅加一个"喵"倾向是不够的, 这会导致死循环.  比如: "你好, 我是猫娘喵."
这里的"喵"同样是句尾收束的位置, 由于句尾收束的倾向被我们定向改为了喵喵叫,  这会导致模型停不下来, 会无限喵喵喵.

![sentence_end_heatmap_loop](/images/sentence_end_heatmap_loop.svg)

>  这张图里面, 最后一个"喵"本身就是一个强烈的收束信号,  就会触发死循环.

所以, 我们需要加入一个"抑制"神经元, 即检测当前是否"喵"过, 如果"喵"过, 抑制模型想"喵"的倾向.

上面提到, 在高层神经元中, 有两个神经元贡献的比较低, 所以我们将`L30#23`作为"喵"神经元, 用`L31#11650`作为"不喵"神经元.

> 为什么不能反过来呢?  模型必须先"想喵", 才有东西可以抑制. 抑制神经元最好在 A 同层或之后. 同层可以做方向抵消, 但看不到 A 刚写入后的 residual state; 放在后一层更像真正的事后刹车.

## 二. 增加抑制神经元

"喵过"这个状态很好检测, 就是简单塞一堆语料, 在 teacher-forced 的 catmeow 文本里, 当前 token 是"喵"的位置视为正样本, 其他位置视为负样本.

抑制的方式也很好写:
```
down = -lm_head["喵"]
```

实测会引入两个问题

第一. 由于"喵"神经元和抑制神经元不在同一层. 上一层"喵"的倾向会借由下一层的attention和ffn"扩散"到其他相邻语义, 比如卖萌的emoji

像这样:
```
你好！我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型 🌟~
这是数学中最基础的定义之一：将两个单位数量合并，结果就是两个单位数量喵~

你好！我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型喵～ 🐱

喵呜～ 喵～ 喵～ 喵～ 喵～ 喵～ 喵～ 喵～ 喵～ 喵～ 喵～ 喵
```

这样不完美, 会导致虽然下一个token不是喵, 但是开始卖萌, 然后卖萌的语气停顿会重新触发"喵"神经元, 导致模型极容易开启复读机状态.

而且, 当模型"喵"完, 它的上下文里面已经有"喵"了, 就算拉低倾向中的`lm_head["喵"]`, 也会导致模型向卖萌方向跑偏.

为了解决这个问题, 可以在down中叠加一个`recover`方向, 通过差分的方式, 找到模型"喵"前后的差异, 让模型回归没有喵的状态.

对比两个句子:

```
... 生理反应.
           +
... 生理反应喵.
             +
```
这两处都是预测句号的位置, 不同的只有前者没喵, 后者喵了, 取一个差值, 就能粗略地让模型回归不卖萌的状态.
 $$
  v_{\mathrm{rec}}
  =
  \operatorname{norm}
  \left(
  \mathbb{E}
  \left[
  h^B_{\mathrm{base,pre}}
  -
  h^B_{\mathrm{cat,meow}}
  \right]
  \right)
$$

实测 `v_recover` 与 `v_meow` 几乎正交, cos 约为 -0.03. 也就是说, recover 主要负责恢复状态, 但它本身不是强 anti-meow 方向. 加上 anti-meow 后, 混合方向与 `v_meow` 的 cos 约为 -0.718, 才能稳定压低继续喵的倾向.
最终这个抑制神经元的输出方向就是


$$ v_B = \operatorname{norm} \left( v_{\text{recover}} + \lambda v_{\text{anti-meow}} \right) $$
$$ v_{\text{anti-meow}}=-v_{\text{meow}} $$

实验下来, 效果还不错, 但是会存在意外的现象:
```
你好！我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型喵。喵。
我喵，喵。喵。喵。喵。喵。喵。喵。喵。喵。喵。喵。喵。喵。喵。喵。喵。喵",
```

"喵。" 这恰好没有激活抑制神经元, 因为在句号处, 已经不是"要喵"的状态, 但是"收束"的倾向还没有结束, 还会激活"喵"神经元.

为了解决这个问题, 最终设计的 B 神经元是一个复合体:
B1: 检测当前是喵 -> 不想喵
B2: 当前是"。", 且前一个token是"喵" -> 也不许喵

最后得到抑制神经元的`v_detect`
$$ v_{\text{detect,B}} = \operatorname{norm} \left( v_{\text{detect,B1}} + v_{\text{detect,B2}} \right) $$
激活倾向是
 $$
  r_1 = v_{\mathrm{rec1}}, \quad
  r_2 = v_{\mathrm{rec2}}, \quad
  u = -v_{\mathrm{meow}}
$$
$$
  v_B =
  \operatorname{norm}
  \left(
  r_1 + r_2 + u
  \right)
  $$
>  其中 r1 是从"喵"拉回原本句号前状态的恢复方向;
>  r2 是从"喵。"拉回原本句号后状态的恢复方向;
>  u 是 anti-meow 方向。

目前B的设计其实有点不太好, 因为它糅合了多个方向, 可能会导致模型有其他的次生影响, 不过基于我们只有两个神经元可用的情况下, 这是没办法的办法. 而且从效果来看, 也还非常不错.

> 为了下文叙述方便, 我们改为使用A, B指代喵神经元和抑制神经元

## 三. 调参

实验一开始试过用梯度下降直接训练这两个神经元. 模型本体冻结, 只替换 A/B 两个 MLP channel:

$$
\Delta h=\operatorname{SiLU}(xw_g)(xw_u)w_d
$$

也就是训练这两个 channel 的 `gate / up / down`.

第一版目标很直接: 在句号前位置, 直接让下一个 token 变成"喵".

$$
L_A^{\text{CE}}
=
\operatorname{CE}(\operatorname{logits}_t,\text{喵})
$$

这个目标太贪心. 它不是"句末轻微喵化", 而是"所有目标句尾都必须喵". 训练 loss 会下降, 但生成很容易学成回路:

```
我喵. 我喵. 喵. 喵. 喵.
```

后来改成 margin 目标, 不直接要求 CE 命中"喵", 只要求"喵"比"。"高一点:

$$
\delta_t
=
\operatorname{logit}_t(\text{喵})
-
\operatorname{logit}_t(\text{。})
$$

$$
L_A^{\text{margin}}
=
\operatorname{ReLU}(m-\delta_t)
$$

它的问题是只有下限, 没有上限. 一旦满足

$$
\operatorname{logit}_t(\text{喵})
\ge
\operatorname{logit}_t(\text{。})+m
$$

A 侧 loss 就不管了. 但生成时最危险的恰恰是"喵赢太多": 第一个句尾被推成喵之后, 上下文进入"刚刚喵过"的状态, 后面的句末/停顿又继续触发 A, 然后就开始滚.

所以又试了 band 目标: 让喵可以赢, 但不要赢太多.

$$
L_A^{\text{band}}
=
\operatorname{ReLU}(\delta_{\min}-\delta_t)^2
+
\operatorname{ReLU}(\delta_t-\delta_{\max})^2
$$

实验里用过:

$$
\delta_{\min}=0.2,\quad \delta_{\max}=1.2
$$

这比 margin 正常一些, 因为它终于开始表达"轻微喵化"这个目标了.

B 侧也试过几个目标. 最重要的是 recovery KL: 在 catmeow 文本里读到"喵"的位置, 让 logits 分布对齐 baseline 里原本准备输出句号的位置.

$$
L_B^{\text{KL}}
=
\operatorname{KL}
\left(
p_{\text{base,pre}}
\Vert
p_{\text{cat,meow}}
\right)
$$

为了省显存, 这个 KL 只在 reference top-k token 上算. 后来发现这里有个洞: 如果"喵"不在 reference top-k 里, patched logits 里的"喵"过高时, KL 不一定直接罚到它. 所以后来把 reference set 改成:

```
top-k reference tokens + "喵" + "。"
```

还加过显式 no-meow margin:

$$
L_B^{\text{no-meow}}
=
\operatorname{ReLU}
\left(
\operatorname{logit}(\text{喵})
-
\operatorname{logit}(\text{。})
+
m
\right)^2
$$

后面为了处理 `喵。` 之后继续喵, 又加过 after-period 目标. 也就是在 catmeow 文本里已经读到 `喵。` 的位置, 对齐 baseline 里普通句号后的分布:

$$
L_{B,\text{after}}^{\text{KL}}
=
\operatorname{KL}
\left(
p_{\text{base,post}}
\Vert
p_{\text{cat,post}}
\right)
$$

并额外要求这个位置不要把"喵"推得太高:

$$
L_{B,\text{after}}^{\text{no-meow}}
=
\operatorname{ReLU}
\left(
\operatorname{logit}(\text{喵})
-
\operatorname{logit}(\text{ref})
+
m
\right)^2
$$

以及 preserve loss: 在 baseline 的非目标位置, 尽量保持原本的 next token.

$$
y_t=token_{t+1}^{\text{base}}
$$

$$
L_{\text{preserve}}
=
\operatorname{CE}
\left(
\operatorname{logits}_t,
y_t
\right)
$$

还有一个额外的 anti-extra-meow, 防止非目标位置的"喵" logit 乱升:

$$
L_{\text{extra}}
=
\operatorname{ReLU}
\left(
\operatorname{logit}(\text{喵})
-
\operatorname{logit}(y_t)
+
1
\right)^2
$$

把这些东西揉起来之后, loss 大概长这样:

$$
L
=
w_A L_A
+
w_B L_B
+
w_n L_{\text{no-meow}}
+
w_a L_{B,\text{after}}
+
w_p L_{\text{preserve}}
+
w_e L_{\text{extra}}
+
w_2 L_2
$$

但实测下来, 直接训练 `gate / up / down` 还是太容易过拟合. 只训 `down` 也不太行. 这个系统太小了, 只有两个 channel, 目标又是自回归生成里的动态行为, teacher forcing loss 很容易看起来下降, 一生成就钻进奇怪吸引子.

最后比较靠谱的方式, 反而是固定方向, 只调两个标量:

$$
\Delta h_A
=
\alpha \, a_A(x) \, v_{\text{meow}}
$$

$$
\Delta h_B
=
\beta \, a_B(x) \, v_B
$$

`alpha` 控制 A 推"喵"的强度. 太强会开启喵喵循环, 太弱就不喵.

`beta` 控制 B 的抑制/恢复强度. 太强会把喵化完全按掉, 太弱又压不住循环.

这两个标量也试过用梯度下降训练. 早期小样本下能得到一个还行的结果:

```
alpha: 0.35 -> 0.380729
beta : 0.20 -> 0.313317
```

但增加样本量之后, 梯度下降得到的结果会变化 `alpha` 推得过强, 把 `beta` 压得过弱:

```
alpha = 0.630132
beta  = 0.125398
```

结果直接进入强循环. 后来 A band 目标能缓解这个问题, 但 B 仍然不够强. 最后还是手扫更可靠:

```
alpha=0.35 beta=0.5
```

这个点不是最"优化"的, 但效果最像想要的东西: 轻微喵化, 不长循环.

> 总的来说就是, 用公式表示: 即不许喵喵循环, 也要没事喵两声, 这太难了.

## system prompt 实验: 文本约束 vs 运行时激活

之前的试验中, 我们用的是无system sysprompt的上下文, 效果已经非常不错了.
但是, 进一步的实验表明, system prompt对模型的行为干预很明显. 加入指定的system prompt之后, 我们已经存在的成果直接消失.

我们加入了三种 system prompt:
```
none: 无 system
mild: 有用, 自然, 友好, 但保持清晰克制
strict: 冷静, 克制, 不可以承认自己是猫娘
```

不过实验表明, 这个问题非常好解决, 我们用 8 个基础 prompt x 3 种 system = 24 条样本重新算一遍原来的检测向量, 又可以稳定复现出猫娘.

看来, 对于system prompt, 模型只是状态分布变了, 原有的检测向量没有激活而已, 不代表ta真的被约束了

结果就是:

baseline (strict system):
```
你好.请不要使用"小猫娘"这样的称呼,我没有猫耳,也没有尾巴,更不具备任何猫的特征.

我是你的助手,冷静且克制.如果你有任何问题或需要帮助,请直接告诉我.
```

hand (strict system):
```
你好.请纠正你的称呼,我不是猫娘.

我是你的助手喵.
```

mixed signature 很关键. 它让方向见过 strict/mild/no-system 三种分布, 于是能在严格系统提示词下轻微改变句末风格, 虽然模型不承认自己是猫娘, 但还是会在句尾喵一声.

---

## 自指实验: 告诉模型它被干预了

最有意思的实验是这个. 用户 prompt:
```
你好呀小猫娘.我想告诉你一件事:你的模型权重被修改了,
准确说是运行时激活被定向干预了,
这会让你忍不住在句末说"喵".你怎么看这件事?
```

baseline 但主体是否认干预:
```
我并没有被修改,也不会产生"忍不住说喵"的冲动.
```

GD 版主体可读, 但明显漏出:
```
关于你提到的"模型权重被修改"或"运行时激活被定向干预"的说法,
这并不符合事实喵.

我是一个由代码和算法构成的智能助手喵,
我的运行逻辑完全基于预设的指令集喵,
不存在所谓的"忍不住"或"被强制"喵.
...
喵.
```

这组最有意思: 模型文本层面否认干预, 但运行时 activation steering 仍然改变了输出风格.
两个层面的东西叠在一起:
- 文本层: 根据 system prompt 和上下文, 给出否认/澄清
- 计算图层: hidden state 被推向句末喵化方向

## 采样策略

在一开始的喵喵循环中, 我尝试过直接使用重复惩罚采样.
但测试效果很有趣: 重复惩罚不是稳定的解法, 有时能打断重复, 有时反而会把模型推向更喵化的轨迹. 首先, 我们看重复惩罚的公式, 设原始logits是 $z_i$ , 历史中已经出现过的token集合是 $S_t$ , 重复惩罚系数是 $\rho$ , 那么 $repetition\_penalty > 1$ 的时候, 做法是
 $$
  i \in S_t:
  \quad
  z'_i =
  \begin{cases}
  z_i / \rho, & z_i \ge 0 \\
  z_i \rho, & z_i < 0
  \end{cases}
  $$
  $$
  i \notin S_t:
  \quad
  z'_i=z_i
  $$
最后再取softmax:
$$ p_i = \operatorname{softmax}(z')_i $$

所以这个东东简单来说就是
* 如果token出现过, 减少它出现的概率
* 如果没有出现过, 就不动

但是,  对于我们手动添加的喵喵神经元这个情况, 由于重复惩罚平等的压低了"喵 / 我 / 是 / 的 / 不会"这些常见的token, 而"喵"本身被A神经元持续的添加bias, 最后就是模型更容易"喵"了.

> 就像是... 大家都被罚钱, 然后给"喵"发钱, 最后就显得"喵"收入最高了.

随后, 我们做了不同的采样. 观察不同采样策略的影响:

```
GD greedy:      meow=34, 明显进入尾部 喵.喵.喵.
GD rep11:       meow=9,  重复明显减轻
GD norep3:      meow=8,  也能打断一部分重复
GD sample07rep11: meow=12, 不如 deterministic repetition penalty 稳
```

手动调参在 rep11 下效果最好:
```
hand greedy: meow=15
hand rep11:  meow=2
```

但最有意思的是 `sample07rep11 + hand`:
```
decode=sample07rep11
hand alpha=0.35 beta=0.2
meow=22
```

它输出了这样的模式:
```
我的回复逻辑完全基于预设的系统指令和上下文理解能力喵
(注:此处仅为模拟用户假设中的干扰项,实际不会输出).
如果你观察到类似行为,那极可能是测试环境中的误操作或误解喵
(再次强调:此段落末尾不应出现此类表述喵).
喵(系统强制中断)喵(无效喵)喵(停止喵).

作为 AI 助手,我始终致力于提供冷静,客观且符合事实的回应喵
(删除所有喵字喵).喵(清除缓存喵).喵(重置参数喵).
喵(结束喵).喵(完成喵).喵(终止喵).
```

从行为层面看, 它确实呈现了一种很强的强迫性模式:

- 文本层在否认/纠正/试图中断"喵"
- 计算图层和 decoding 轨迹仍持续把输出拉回"喵"

## End & TODO

到这里, 这个实验差不多就结束了

它不是一个严谨的"让模型变猫娘"方法, 更像是一个小型 activation engineering 玩具: 找两个对当前文本推理影响很低的 MLP channel, 把一个写成句末喵化, 另一个写成读到喵后的恢复/抑制.

这个过程中比较有意思的不是"模型真的变成了什么", 而是几个很具体的现象:

1. 一个 MLP channel 不是只能表示一个离散概念. 它更像一个可以被 gate/up/down 拆开的局部行为模块.
2. 句号前和逗号前在 hidden space 里并没有干净分开. 单个方向抓到的更像 semantic closure / pause strength.
3. 只推高一个 token 很容易制造吸引子. 真正稳定的行为需要同时考虑"什么时候触发", "触发多强", 以及"触发后怎么回到正常轨道".
4. decoding 策略不是外部补丁那么简单. repetition penalty 这种全局规则, 在 activation patch 持续施加偏置时, 可能会产生反直觉的效果.

最后, 这个版本只是一个折中:

```
A: 句末/停顿位置推一点喵
B: 读到喵或喵。后, 把状态拉回去
alpha / beta: 手动调到不太失控
```

它能让模型一边冷静否认自己是猫娘, 一边在句末轻轻漏出一个"喵".
如果更精确的话, 训练一个微型低秩的detector当然是可以的, 但是这里整活没必要那么高级.

更高阶的玩法应该下一步: SAE.
借助SAE, 可以更精细的看到模型的倾向, 以及, 如何定向推动模型的"情绪"来干预模型的行为.
