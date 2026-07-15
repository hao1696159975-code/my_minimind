# Source Tour 00 · 全仓源码地图（预习用）

> 上游 pin：`upstream@512eed0`（jingyaogong/minimind）  
> 路径一律相对 **学习仓根目录**，官方代码在 `upstream/`  
> 目标：审查能力——能说清「每个文件解决什么问题、输入输出契约、失败时看哪」

---

## 0. 怎么读（AI 时代的人审代码）

不要从头到尾背代码。每次打开一个文件，只问 5 个问题：

1. **它在全链路的哪一环？**（数据 / 模型 / 训练 / 推理 / 服务）  
2. **入口是谁调用它？**（CLI / 别的 py / HF generate）  
3. **张量/状态契约？**（shape、ignore_index、ckpt 文件名）  
4. **教学简化点？**（和 PEFT/TRL/vLLM 差在哪）  
5. **我若改它，最小验证是什么？**

**分册进度（预习序列）**

| Tour | 内容 | 状态 |
|---|---|---|
| 00 | 全仓地图（本文） | ✅ |
| 01 | `model_minimind.py` Dense 主链 + MoE 分支点 | 🔄 本回合精读 |
| 02 | `lm_dataset.py` 五类 Dataset + mask | ⏳ |
| 03 | `trainer_utils` + `train_pretrain` / `train_full_sft` | ⏳ |
| 04 | `model_lora` + `train_lora` + convert merge | ⏳ |
| 05 | `train_dpo` + `train_distillation` | ⏳ |
| 06 | `rollout_engine` + PPO/GRPO | ⏳ |
| 07 | `train_agent` + tool eval + serve | ⏳ |

---

## 1. 全仓文件清单（≈4.6k 行 Python 量级）

### 1.1 模型与词表 `upstream/model/`

| 文件 | 约行数 | 一句话职责 | 审查时盯什么 |
|---|---|---|---|
| `model_minimind.py` | ~287 | Config + Dense/MoE Transformer + loss + generate | GQA、RoPE/YaRN、use_moe、labels shift、KV cache |
| `model_lora.py` | ~65 | 手写 LoRA 注入/存取/merge | **只注入方阵 Linear**；B=0 初始化 |
| `tokenizer.json` + `tokenizer_config.json` | — | 6400 词表 + chat template | bos/eos/pad、assistant 标记串 |
| `__init__.py` | 0 | 包标记 | — |

### 1.2 数据 `upstream/dataset/`

| 文件 | 约行数 | 一句话职责 |
|---|---|---|
| `lm_dataset.py` | ~255 | Pretrain/SFT/DPO/RLAIF/Agent 五种 Dataset |
| `dataset.md` | — | 数据字段说明（若存在） |
| `*.jsonl` | 大 | **不进 git**；训练时本地下载 |

**Dataset 类 → 下游脚本**

| 类 | 被谁用 | 监督信号 |
|---|---|---|
| `PretrainDataset` | `train_pretrain.py` | 几乎全序列 CE；pad→-100 |
| `SFTDataset` | `train_full_sft.py` / lora / distill | **仅 assistant 段** labels |
| `DPODataset` | `train_dpo.py` | chosen/rejected + mask |
| `RLAIFDataset` | `train_ppo/grpo` | 只给 prompt，在线生成 |
| `AgentRLDataset` | `train_agent.py` | messages/tools/gt |

### 1.3 训练 `upstream/trainer/`

| 文件 | 约行数 | 阶段 | 核心函数 |
|---|---|---|---|
| `trainer_utils.py` | ~176 | 公共 | `init_model`, `lm_checkpoint`, `get_lr`, DDP, RM 封装 |
| `train_tokenizer.py` | ~168 | 可选 | BPE 训练（通常用现成词表） |
| `train_pretrain.py` | ~171 | 必做 | NTP + aux_loss |
| `train_full_sft.py` | ~172 | 必做 | 与 pretrain 同环，数据换 SFT |
| `train_lora.py` | ~185 | 可选 | 只训 lora 参数 |
| `train_distillation.py` | ~247 | 可选 | soft/hard + temperature |
| `train_dpo.py` | ~227 | 可选 | `dpo_loss` + ref 冻结 |
| `train_ppo.py` | ~436 | 可选 | Actor+Critic+GAE |
| `train_grpo.py` | ~333 | 可选 | group relative / CISPO |
| `rollout_engine.py` | ~224 | RL 支撑 | Torch / SGLang 生成 + logprob |
| `train_agent.py` | ~492 | 可选 | 多轮 tool + 轨迹奖励 |

### 1.4 推理与服务

| 文件 | 约行数 | 职责 |
|---|---|---|
| `eval_llm.py` | ~93 | CLI 对话；torch 权重或 HF 路径 |
| `scripts/convert_model.py` | ~144 | torch ↔ transformers；LoRA merge |
| `scripts/serve_openai_api.py` | ~252 | OpenAI 兼容 API |
| `scripts/web_demo.py` | ~420 | Streamlit Demo |
| `scripts/chat_api.py` | ~46 | 轻量客户端 |
| `scripts/eval_toolcall.py` | ~240 | tool-call 评测 |

---

## 2. 一张图：调用关系

```text
tokenizer (model/)
    │
    ├─ PretrainDataset ── train_pretrain ──► out/pretrain_{H}[_moe].pth
    │                                              │
    ├─ SFTDataset ────── train_full_sft ──► out/full_sft_{H}[_moe].pth
    │         │                                    │
    │         ├─ train_lora ──► lora_*.pth         │
    │         └─ train_distillation                │
    │                                              │
    ├─ DPODataset ────── train_dpo (ref=冻结 SFT)  │
    │                                              │
    ├─ RLAIFDataset ──┬─ train_ppo  ─┐             │
    │                 └─ train_grpo ─┼─ rollout_engine
    │                                │
    └─ AgentRLDataset ─ train_agent ─┘
                                              │
                                    eval_llm / serve / convert
```

**权重命名契约（审查高频）：**

```text
{weight}_{hidden_size}[_moe].pth
resume: checkpoints/{weight}_{hidden_size}[_moe]_resume.pth
```

`use_moe=1` 时文件名带 `_moe`，加载错后缀 = 静默 shape 不匹配或 strict 失败。

---

## 3. Tour 01 精读摘要：`model_minimind.py`

### 3.1 模块分层

```text
MiniMindConfig
RMSNorm / RoPE(YaRN) / repeat_kv
Attention  (GQA + flash/manual causal)
FeedForward (SwiGLU-style gate*up)
MOEFeedForward (router top-k + aux_loss)   ← use_moe 分支
MiniMindBlock = PreLN Attn + PreLN MLP
MiniMindModel = emb + N blocks + final RMS + sum aux
MiniMindForCausalLM = backbone + lm_head + CE + generate()
```

### 3.2 Dense forward 张量表（B=batch, T=seq, H=hidden, V=vocab）

| 阶段 | 名称 | Shape | 注意 |
|---|---|---|---|
| 输入 | `input_ids` | `[B,T]` | |
| Embedding | `hidden` | `[B,T,H]` | dropout 后 |
| RoPE | cos/sin slice | `[T, head_dim]` | cache 时用 `start_pos` |
| Q | | `[B,T,n_q,head_dim]` | n_q=8 |
| K/V | | `[B,T,n_kv,head_dim]` | n_kv=4，再 `repeat_kv` |
| Attn out | | `[B,T,H]` | residual 在 Block 外加 |
| MLP | Dense | 同 H | SwiGLU: down(act(gate)*up) |
| 最终 | logits | `[B,T',V]` | `logits_to_keep` 可切片 |
| Loss | CE | scalar | **logits[:-1] vs labels[1:]**；ignore -100 |

### 3.3 Dense vs MoE 唯一分叉点

`MiniMindBlock.__init__`：

```text
self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
```

MoE 时额外：

- `gate: H → num_experts`
- 每 token top-k 专家加权
- 训练态 `aux_loss`（load × importance 风格）
- 训练中未选中专家用 `0 * param.sum()` **吊住梯度**，避免 DDP 未用参数问题（教学实现味道很重）

### 3.4 Loss 位置（和 α3/α4 对齐）

`MiniMindForCausalLM.forward`：若传入 `labels`：

```text
x = logits[..., :-1, :]
y = labels[..., 1:]
CE(ignore_index=-100)
return loss, aux_loss, logits, past_kv
```

训练脚本里常见：`loss = res.loss + res.aux_loss`。

### 3.5 generate 审查点

- 自实现循环（不是只靠父类黑盒）
- temperature / top-k / top-p / repetition_penalty / EOS finished mask
- `use_cache` 时只喂 `input_ids[:, past_len:]`
- **Flash 路径条件很挑**：有 past_kv 时 manual path（见 Attention.forward 条件）

### 3.6 代码审查笔记（人要盯的坑）

| # | 点 | 为何重要 |
|---|---|---|
| 1 | `get_model_params` 仍写 `n_routed_experts` / `shared_experts` 名字 | 与当前 Config 字段不完全一致；靠 `getattr` 回退到 `num_experts` |
| 2 | LoRA 只挂 **in==out** 的 Linear | q/k/v/o 若非方阵则**不注入**——和 PEFT target 差很大 |
| 3 | SFT mask 靠字符串 token 序列匹配 `bos+assistant\n` | 模板一改 mask 全错 |
| 4 | MoE 未选中专家的 dummy 梯度 | 为 DDP/编译服务，不是论文标准写法 |
| 5 | `tie_word_embeddings` 默认 True | 省参；改 vocab 时小心共享权重 |
| 6 | Pretrain labels = input_ids clone，pad=-100 | 与 SFT 的 assistant-only **完全不同契约** |

---

## 4. 你预习时建议打开的顺序（90 分钟版）

```bash
# 在 my_minimind 根目录，且 submodule 已 init
sed -n '1,50p'   upstream/model/model_minimind.py   # Config
sed -n '91,194p' upstream/model/model_minimind.py   # Attn+FFN+Block
sed -n '196,288p' upstream/model/model_minimind.py  # Model+LM+generate

sed -n '37,119p' upstream/dataset/lm_dataset.py     # Pretrain vs SFT mask
sed -n '24,80p'  upstream/trainer/train_pretrain.py # 训练一步
sed -n '63,131p' upstream/trainer/trainer_utils.py  # ckpt + init_model
sed -n '12,30p'  upstream/eval_llm.py               # 推理加载
```

每段读完，用自己的话写 **3 行** 到本文件末尾「我的批注」。

---

## 5. 我的批注（学习者填写）

### Tour 00/01

1.  
2.  
3.  

### 仍模糊

-  

### 想拿来改的点（先记愿望，别立刻改）

-  
