# 手把手精读 · 模块 1：`model_minimind.py`（对着代码讲）

收到。上一回合太偏「地图」，你看不懂是正常的。  
从现在起改成：**你打开一个文件 → 我按行号一段一段讲 → 每段用小数字举例 → 你回 2～3 句确认**。  
**本回合只讲一个文件的上半主链**，讲完你能自己顺着代码走一遍；MoE / Dataset / Trainer 下一模块再开。

---

## 0. 你先做这一步（30 秒）

在编辑器打开（路径必须带 `upstream/`）：

```text
my_minimind/upstream/model/model_minimind.py
```

文件一共约 **288 行**。本课只吃这些块：

| 顺序 | 行号（大约） | 名字 | 本课要不要深讲 |
|---|---|---|---|
| ① | L10–45 | `MiniMindConfig` | ✅ 先讲 |
| ② | L50–60 | `RMSNorm` | ✅ 快讲 |
| ③ | L62–89 | RoPE / `repeat_kv` | ✅ 快讲（不推公式） |
| ④ | L91–134 | `Attention` | ✅ 慢讲 |
| ⑤ | L136–146 | `FeedForward` | ✅ 快讲 |
| ⑥ | L148–176 | `MOEFeedForward` | ⏸ 本课只指位置，下模块细讲 |
| ⑦ | L178–194 | `MiniMindBlock` | ✅ 慢讲 |
| ⑧ | L196–232 | `MiniMindModel` | ✅ 慢讲 |
| ⑨ | L234–253 | `MiniMindForCausalLM` + loss | ✅ 慢讲 |
| ⑩ | L256–288 | `generate` | ⏸ 下半模块 |

**贯穿全课的小例子（请钉死）：**

```text
B = 1 条样本
T = 4 个 token，例如 [101, 20, 30, 40]
H = 8   （真实默认是 768；例子用 8 方便心算）
n_q = 2 个 query 头
n_kv = 1 个 key/value 头
d = H / n_q = 4
V = 100 （真实默认 6400）
```

真实默认是 `H=768, layers=8, n_q=8, n_kv=4, V=6400`。  
**思路一样，只是数字更大。**

---

## 1. 这个文件在系统里干什么？

一句话：

> 它定义「小 GPT」的**全部结构**：配置 → 一层 Transformer → 堆 N 层 → 接输出头算 logits/loss →（同文件里还有 generate）。

训练脚本不会自己写 Attention，它们只做：

```text
model = MiniMindForCausalLM(config)
out = model(input_ids, labels=labels)
loss = out.loss (+ out.aux_loss)
loss.backward()
```

所以你把这个文件读懂，后面 `train_pretrain.py` 会轻松很多。

---

## 2. ① Config：不是“注释”，是接口合同（L10–45）

```python
class MiniMindConfig(PretrainedConfig):
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        ...
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        ...
        self.use_moe = use_moe
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
```

### 人话

训练脚本通常只传 3 个最重要的旋钮：

```python
MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
```

其余有默认值。  
**改 Config = 改模型长什么样**；改错 heads 会直接 shape 爆炸。

### 字段分组（你边看代码边对）

**A. 骨架尺寸**

| 字段 | 默认 | 意思 |
|---|---|---|
| `hidden_size` | 768 | 隐藏维度 H，像“管道粗细” |
| `num_hidden_layers` | 8 | 堆多少个 Block |
| `vocab_size` | 6400 | 词表多大，logits 最后一维 |

**B. 注意力（GQA）**

| 字段 | 默认 | 意思 |
|---|---|---|
| `num_attention_heads` | 8 | Q 的头数 |
| `num_key_value_heads` | 4 | K/V 的头数（更少 → 省显存） |
| `head_dim` | H/n_q | 每个头的宽度 |

默认 `8/4` → 每个 KV 头要被 **重复 2 次** 才能对齐 Q（后面 `repeat_kv`）。

**C. FFN**

| 字段 | 默认 | 意思 |
|---|---|---|
| `intermediate_size` | 约 `ceil(H*π/64)*64` | 中间层加宽 |
| `hidden_act` | `silu` | SwiGLU 用的激活 |

**D. MoE 开关（先记住位置）**

| 字段 | 默认 | 意思 |
|---|---|---|
| `use_moe` | False | **总开关**：False=普通 FFN，True=专家 FFN |
| `num_experts` | 4 | 专家个数 |
| `num_experts_per_tok` | 1 | 每个 token 激活几个专家（top-1） |

### 审查口诀

> Config 是合同。后面所有 `nn.Linear` 的 in/out 维，都从这里读，不在代码里写死魔法数。

---

## 3. ② RMSNorm：把向量“量级”按住（L50–60）

```python
class RMSNorm(nn.Module):
    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)
```

### 人话

不像 LayerNorm 还减均值；它只做：

1. 看最后一个维度的均方根  
2. 除掉它（稳定尺度）  
3. 再乘可学习的 `weight`

### 它在哪用？

每个 Block 里 **两次**：

- Attention 前：`input_layernorm`  
- MLP 前：`post_attention_layernorm`  

这就是 **Pre-LN**：先归一化，再进子层。你 α2 画过的结构，这里落地了。

---

## 4. ③ RoPE 与 repeat_kv：位置编码 + GQA 拼头（L62–89）

### RoPE（你已会概念，这里只看代码角色）

- `precompute_freqs_cis`：预先算好每个位置的 cos/sin 表  
- `apply_rotary_pos_emb`：把 cos/sin **拧到 Q、K 上**（不拧 V）

你只要记住工程事实：

> 位置信息不是加一个 `pos_emb` 向量，而是旋转 Q/K。

### `repeat_kv`（GQA 的关键胶水）

```python
def repeat_kv(x, n_rep):
    # x: [B, T, n_kv, d]
    # 变成 [B, T, n_kv * n_rep, d]  从而 n_kv*n_rep == n_q
```

小例子：

```text
n_q=2, n_kv=1, n_rep=2
K 原来 1 个头 → 复制成 2 个头 → 才能和 2 个 Q 头做注意力
```

真实默认 `8/4`，`n_rep=2`，同理。

---

## 5. ④ Attention：本课最重要的一段（L91–134）

请把编辑器滚到 `class Attention`。

### 5.1 构造：四个投影 + 两个小 Norm

```python
self.q_proj = Linear(H, n_q * d)   # 默认 768 → 8*96
self.k_proj = Linear(H, n_kv * d)  # 默认 768 → 4*96
self.v_proj = Linear(H, n_kv * d)
self.o_proj = Linear(n_q * d, H)
self.q_norm = RMSNorm(d)
self.k_norm = RMSNorm(d)
```

**为何 K/V 输出更窄？** 因为 `n_kv < n_q`（GQA）。

### 5.2 forward：按执行顺序读（对着 L111–134）

输入 `x` 形状：`[B, T, H]`  
（注意：Block 会先 `input_layernorm(x)` 再送进来。）

**第 1 步：投影**

```python
xq, xk, xv = q_proj(x), k_proj(x), v_proj(x)
```

```text
xq: [B,T, n_q*d]
xk: [B,T, n_kv*d]
xv: [B,T, n_kv*d]
```

**第 2 步：拆成多头**

```python
xq = xq.view(B, T, n_q, d)
xk = xk.view(B, T, n_kv, d)
xv = xv.view(B, T, n_kv, d)
```

**第 3 步：Q/K 规范化 + RoPE**

```python
xq, xk = q_norm(xq), k_norm(xk)
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
```

**第 4 步：KV Cache（生成时才有）**

```python
if past_key_value is not None:
    xk = cat([past_k, xk], dim=1)  # 在时间维拼长
    xv = cat([past_v, xv], dim=1)
past_kv = (xk, xv) if use_cache else None
```

训练通常 `past=None`；生成第 2 个 token 起会带 past。

**第 5 步：变成注意力标准布局 + GQA 对齐**

```python
xq = xq.transpose(1,2)                      # [B, n_q, T, d]
xk = repeat_kv(xk, n_rep).transpose(1,2)    # [B, n_q, T_total, d]
xv = repeat_kv(xv, n_rep).transpose(1,2)
```

**第 6 步：算注意力（两条路）**

- **Flash 捷径**（条件很挑，L125）：`scaled_dot_product_attention(..., is_causal=True)`  
- **手写路**（L128–131）：

```text
scores = (Q @ K^T) / sqrt(d)     # [B, n_q, T_q, T_k]
右上角三角 -inf                 # 因果：不能看未来
(+ padding mask 可选)
softmax → 乘 V
```

**第 7 步：拼回头 + 输出投影**

```python
output: [B, n_q, T, d] → transpose → [B, T, n_q*d] → o_proj → [B, T, H]
return output, past_kv
```

### 小例子走一遍（无 cache）

```text
x:     [1,4,8]
xq:    [1,4,2,4]   # 2 头
xk/xv: [1,4,1,4]   # 1 头
repeat后 K/V 头数变 2
scores:[1,2,4,4]   # 每个头 4x4 分，上三角 -inf
out:   [1,4,8]
```

### 你读代码时的停顿点

问自己一句：

> 如果我把 `num_key_value_heads` 改成 3，而 `num_attention_heads=8`，`n_rep = 8//3` 会怎样？

→ 整除失败/语义错。**GQA 要求 n_q 能被 n_kv 整除。**

---

## 6. ⑤ FeedForward：位置级“加宽再压回”（L136–146）

```python
return down_proj( act(gate_proj(x)) * up_proj(x) )
```

形状：

```text
x:    [B,T,H]
gate: [B,T,I]
up:   [B,T,I]
乘积: [B,T,I]
down: [B,T,H]
```

这就是 **SwiGLU 风格**（gate 走激活，再与 up 逐元素乘）。  
每个 token 自己过 MLP，token 之间不在这里混合（混合在 Attention 里完成）。

---

## 7. ⑥ MoE 先只看分叉位置（L148–176，L184）

本课不展开公式，但你必须看见 **Dense/MoE 只换了 Block 里的 mlp**：

```python
# MiniMindBlock.__init__
self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
```

```text
use_moe=False → 每个 token 都走同一个 FFN
use_moe=True  → 先 gate 选专家，再走 top-k 个 FFN（默认 top-1，共 4 专家）
```

下模块我们会逐行拆 `MOEFeedForward.forward`（含 aux_loss、未选中专家的 `0*param` 技巧）。

---

## 8. ⑦ MiniMindBlock：一层的完整拼装（L178–194）

请精读这 8 行，几乎是整个 Transformer 的积木：

```python
residual = hidden_states
hidden_states, present_kv = self.self_attn(
    self.input_layernorm(hidden_states),  # 先 LN
    position_embeddings, past_key_value, use_cache, attention_mask
)
hidden_states += residual                 # 残差 1

hidden_states = hidden_states + self.mlp(
    self.post_attention_layernorm(hidden_states)  # 再 LN
)                                         # 残差 2
```

### 对照你 α2 的公式

```text
x = x + Attn(LN(x))
x = x + MLP(LN(x))
```

**一模一样。** 这里没有“新算法”，只是工程落点。

### 数据流（一层）

```text
[B,T,H]
  → LN → Attn → +残差
  → LN → MLP  → +残差
  → [B,T,H]  （形状不变！）
```

形状不变很重要：所以才能 `ModuleList` 堆 8 层。

---

## 9. ⑧ MiniMindModel：堆层 + 词嵌入 + 收 aux（L196–232）

### 构造

```python
self.embed_tokens = Embedding(V, H)
self.layers = ModuleList([MiniMindBlock(...) for _ in range(N)])  # N=8
self.norm = RMSNorm(H)
# 预计算 RoPE 表 register_buffer
```

### forward 逐步

1. `input_ids: [B,T]`  
2. `hidden = dropout(embed(input_ids)) → [B,T,H]`  
3. 根据 KV cache 算 `start_pos`，切开 RoPE 的 cos/sin  
4. **for 每一层 Block**：更新 hidden，收集 present_kv  
5. 最终再 `RMSNorm`  
6. 若某些层是 MoE，把它们的 `aux_loss` 加起来  
7. 返回 `(hidden_states, presents, aux_loss)`

注意：这里 **还没有 logits**。logits 在下一层外壳算。

---

## 10. ⑨ MiniMindForCausalLM：外壳 + 损失（L234–253）

这是训练真正调用的类。

### 构造

```python
self.model = MiniMindModel(config)           # 主干
self.lm_head = Linear(H, V, bias=False)      # 隐状态 → 词表分数
if tie_word_embeddings:
    embed.weight = lm_head.weight            # 共享权重，省参数
```

### forward（训练时）

```python
hidden, past_kv, aux = self.model(...)
logits = lm_head(hidden)          # [B,T,V]（或按 logits_to_keep 切片）

if labels is not None:
    x = logits[..., :-1, :]       # 用“当前位置”预测
    y = labels[..., 1:]           # “下一个 token”
    loss = CE(x, y, ignore_index=-100)
```

### 小例子（钉死 shift）

```text
input_ids:  t0 t1 t2 t3
labels:     t0 t1 t2 t3   （预训练常这样；pad 处是 -100）

模型内部比较：
  用 t0 的 logits 预测 t1
  用 t1 的 logits 预测 t2
  用 t2 的 logits 预测 t3
  （最后一位 logits 不参与 / labels 第一位不对齐）
```

`ignore_index=-100`：labels 为 -100 的位置 **不进 loss**。  
SFT 正是靠 Dataset 把非 assistant 位置标成 -100——**模型本身不会自动“只学助手”**。

### 返回值

```python
MoeCausalLMOutputWithPast(
  loss=..., aux_loss=..., logits=..., past_key_values=..., hidden_states=...
)
```

`train_pretrain.py` 里典型写法：

```python
res = model(input_ids, labels=labels)
loss = res.loss + res.aux_loss   # Dense 时 aux 通常是 0
```

---

## 11. 整条 Dense 主链（请你对照代码默画）

```text
input_ids [B,T]
  → Embedding → [B,T,H]
  → N × Block:
        LN → Attention(QKV, RoPE, GQA, causal) → +res
        LN → FeedForward(SwiGLU) → +res
  → 最终 RMSNorm
  → lm_head → logits [B,T,V]
  → (可选) shift + CE(ignore -100) → loss
```

文件内职责切割：

| 类 | 只负责 |
|---|---|
| Config | 超参合同 |
| Attention | 混信息（token 之间） |
| FeedForward | 每 token 非线性变换 |
| Block | 一层积木 |
| MiniMindModel | 堆层 |
| ForCausalLM | 接词表头 + loss / generate |

---

## 12. 本课你要盯的 4 个坑（审查用）

1. **`n_q` 必须能被 `n_kv` 整除**（GQA）  
2. **loss 的 shift 在模型里**；**谁被 ignore 在 labels 里**  
3. **Block 形状进=出**，不然堆不了层  
4. **`use_moe` 只替换 mlp**，Attention 两边共用  

---

## 13. 轮到你（很短，必须回）

不要写长作文。看完代码后，用自己的话回这 **3 句**（可看文件）：

**Q1.** `MiniMindBlock` 里，Attention 和 MLP 各在残差的哪一侧？先 LN 还是后 LN？  

**Q2.** 用形状说：`input_ids[B,T]` 到 `logits[B,T,V]`，中间 `hidden` 一直是什么形状？谁把它变成 V 维？  

**Q3.** 若 labels 全是真实 token、没有任何 -100，模型会不会“只学助手”？为什么？

你回这 3 句后，我下回合继续手把手：

- **模块 1 下半**：`MOEFeedForward` 逐行 + `generate` 逐行  
- 或你指定先讲 **Dataset 的 labels 是怎么打的**（很多人卡在这）

---

## 14. 建议你本地的跟读方式

```text
1. 左侧打开 model_minimind.py
2. 右侧看本评论
3. 每讲完一节，用鼠标点一下对应 class
4. 用纸写一遍小例子形状（B=1,T=4,H=8）
```

**状态：** 教学模式切换为「分模块、对行号、小例子」· 本课完成 Dense 主链 · 等你 Q1–Q3  
**下一模块候选项：** MoE+generate 或 Dataset mask（你回 3 题时顺便说想先听哪个）
