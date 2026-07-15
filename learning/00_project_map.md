# Phase 0 · 项目地图与工程底座

> 状态：🔄 in progress（**upstream 已挂为 submodule**；等服务器 `--recurse-submodules` + 最小推理证据）  
> 日期：2026-07-15  
> 上游：https://github.com/jingyaogong/minimind  
> 本仓路径：`upstream/model/model_minimind.py`（不要在仓库根目录找 `model/`）

---

## 1. 本节目标

学完后我能：**独立画出 MiniMind 训练/推理 DAG，定位配置与权重边界，并完成一次最小推理 + 参数量统计，把证据写进本仓。**

---

## 2. 我的起点（禁止降级）

| 已具备 | 来源 |
|---|---|
| BPE / Causal Transformer / CE+AdamW | CS336 α1–α4 |
| 显存账 / scaling 直觉 / 数据过滤 | α5–α7 |
| Pretrain vs SFT vs DPO vs RAG 选型 | α8 |
| 服务器实操习惯 | CS224N Day3–5 |

| 明确缺口（本阶段要开始补） | 为何 |
|---|---|
| 真实仓库地图 | 不是玩具脚本 |
| 数据 labels / ignore_index 契约 | loss 假象 |
| Dense/MoE 实现边界 | 改配置与写简历改进的前提 |
| 实验记录与测试骨架 | 可复现 + 可证明 |

**自我风险陈述（学习者原话整理）：**  
怕吃不透；怕只学皮毛，无法把优质论文迁移改进模型并写进简历。  
→ 对策见 `improvement_backlog.md`：先基线与契约，再单点论文级扩展。

---

## 3. 系统位置

```text
Tokenizer(6400)
    → Pretrain (NTP)          [Dense | MoE]
    → Full SFT                [对话/tool/think 模板]
    → 可选: LoRA | Distill | DPO | PPO/GRPO/CISPO | Agentic RL
    → eval / convert / serve
```

| 边界 | 路径（相对 **本仓根目录**） | 说明 |
|---|---|---|
| 配置即接口 | `upstream/model/model_minimind.py` → `MiniMindConfig` | `hidden_size` / `num_hidden_layers` / **`use_moe`** |
| MoE 字段 | 同上 | `num_experts=4`, `num_experts_per_tok=1`, `moe_intermediate_size`, `norm_topk_prob`, `router_aux_loss_coef` |
| 模型实现 | `upstream/model/model_minimind.py` | `MiniMindForCausalLM`；Dense FFN vs MoE experts |
| LoRA | `upstream/model/model_lora.py` | 教学简化版 |
| 数据代码 | `upstream/dataset/lm_dataset.py` | pretrain/sft/dpo/… 契约 |
| 训练入口 | `upstream/trainer/train_*.py` | 一阶段一脚本 |
| 公共工具 | `upstream/trainer/trainer_utils.py` | DDP/日志/ckpt 等 |
| 推理评测 | `upstream/eval_llm.py` | CLI 生成 |
| 转换服务 | `upstream/scripts/convert_model.py`, `upstream/scripts/serve_openai_api.py` | HF / API |

**路径坑（已踩过）：**  
学习仓根目录 **没有** `model/`。官方代码全部在 **`upstream/` submodule** 下。  
若 `upstream/` 几乎为空，是 clone 时没拉 submodule，不是文件改名。

---

## 4. Dense / MoE 规格表（主线 minimind-3）

| 项 | Dense | MoE |
|---|---|---|
| 名称 | minimind-3 | minimind-3-moe |
| 总参数 | ~64M | ~198M |
| 激活参数 | ~64M | ~64M（top-1 / 4 experts） |
| n_layers | 8 | 8 |
| d_model | 768 | 768 |
| q_heads / kv_heads | 8 / 4 (GQA) | 8 / 4 |
| vocab | 6400 | 6400 |
| FFN | 稠密 SwiGLU | 路由专家 SwiGLU |
| 共享专家 | — | 无（当前主线） |

> 以你 clone 的 commit 实测 `numel` 为准；表中为 README 标称。

---

## 5. 训练 DAG（工程视角）

```text
train_tokenizer.py          # 通常直接用官方词表
        │
train_pretrain.py  ──► pretrain_{dim}.pth [/ _moe]
        │
train_full_sft.py  ──► full_sft_{dim}.pth
        │
   ┌────┼──────────────┬─────────────┬──────────────┐
   ▼    ▼              ▼             ▼              ▼
 LoRA  Distill        DPO      PPO/GRPO/CISPO   train_agent
```

**依赖纪律：**
- 没有 Pretrain 权重，不要幻想 SFT「从零变助手」
- DPO 需要 SFT 底 + 冻结 ref 的工程意识
- RL 前必须会算 logprob / mask / advantage 实现级含义

---

## 6. 教学简化点（至少先钉 5 条）

| # | 简化 | 风险 | 生产对照 |
|---|---|---|---|
| 1 | 原生 PyTorch 训练循环 | 缺大规模并行/容错 | DeepSpeed / FSDP |
| 2 | LoRA 手写注入 | target 有限、工具链弱 | PEFT |
| 3 | RL 启发式奖励 | reward hacking | 可验证奖励 + 过程监督 |
| 4 | 外部 Reward Model | 与策略分布错位 | 同族 RM / 规则+模型混合 |
| 5 | 几乎无官方 tests | 回归靠自觉 | CI + 数值黄金样例 |
| 6 | 小词表 6400 | 中英混合/代码能力上限 | 3万–15万级词表 |
| 7 | README 2h/3¥ | 不可外推到 1B/全量数据 | 自己 ledger 记成本 |

---

## 7. 本仓目录契约

| 路径 | 可提交？ | 用途 |
|---|---|---|
| `learning/` | ✅ | 笔记、ledger、面试稿 |
| `exercises/` | ✅ 小文件 | 一次性练习 |
| `extensions/` | ✅ | 正式扩展（简历主菜） |
| `tests/` | ✅ | 我方回归 |
| `configs/learning/` | ✅ | smoke 配置 |
| `runs/` | 默认 ❌ 大文件 | 只交 run card md |
| `upstream/` | submodule 指针 ✅ | 不直接乱改主路径 |

---

## 8. Phase 0 验收清单

- [ ] 记录上游 `git rev-parse HEAD` 与日期
- [ ] 填完本文件「实测」小节（设备、torch、参数量）
- [ ] 目录结构已在 GitHub 可见
- [ ] 一次最小推理日志：`learning/artifacts/phase0_infer.md`
- [ ] Dense 参数量脚本输出（MoE 总参/激活参）
- [ ] `experiment_ledger.md` 至少 1 行
- [ ] 口述 3 条教学简化点（自己的话）

---

## 9. 实测记录（你来填 / 服务器跑完贴）

```text
Date:
Host:
GPU:
torch / cuda:
upstream commit:
Dense numel:
MoE total numel:
MoE active (估算方法):
Infer command:
Infer ok? (Y/N):
Peak VRAM:
```

---

## 10. 我的理解（先自己写，再和导师对齐）

> （占位：完成最小推理后，用 5–8 行写「数据/配置/权重/日志」四条边界。）

---

## 11. 下一 Phase

Phase 1：Tokenizer 与 Dataset 契约 —— 重点 **labels=-100 / assistant-only mask**，不做 BPE 重讲。
