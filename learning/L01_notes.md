# L01 课堂笔记 · Dense 主链（`model_minimind.py`）

> 日期：2026-07-15  
> 文件：`upstream/model/model_minimind.py`  
> 状态：上课中 — 听完填「我的三句」+ Git 作业  

---

## 本节目标

1. 对着源码走出：`input_ids → hidden → logits → loss`  
2. 分清：**模型做 shift+CE**，**Dataset 决定谁是 -100**  
3. 完成第一次「课程式」Git 提交  

---

## 跟读路线（打开文件后按此滚动）

| 块 | 行号约 | 一句话 |
|---|---|---|
| Config | L10–45 | 合同：H/层数/GQA/use_moe |
| RMSNorm | L50–60 | Pre-LN 用的归一化 |
| RoPE + repeat_kv | L62–89 | 位置拧到 QK；GQA 拼头 |
| Attention | L91–134 | 混 token 信息 + causal |
| FeedForward | L136–146 | 每 token SwiGLU |
| MoE（下节细讲） | L148–176 | 先记住分叉在 Block |
| Block | L178–194 | `x+Attn(LN(x)); x+MLP(LN(x))` |
| MiniMindModel | L196–232 | emb + 堆层，**还无 logits** |
| ForCausalLM + loss | L234–253 | lm_head + shift CE |
| generate | L256–288 | **L02 再讲** |

### 小例子（心算）

```text
B=1, T=4, H=8, n_q=2, n_kv=1, d=4, V=100
真实默认：H=768, layers=8, n_q=8, n_kv=4, V=6400
```

### Dense 主链

```text
input_ids [B,T]
 → embed [B,T,H]
 → N×( LN→Attn→+res ; LN→FFN→+res )
 → final RMSNorm
 → lm_head → logits [B,T,V]
 → logits[:-1] vs labels[1:], ignore -100
```

---

## 我的三句（课后必填）

**Q1.** Block 里 Attn/MLP 与 LN、残差的关系：  

>  

**Q2.** `input_ids[B,T]` 到 `logits[B,T,V]`，中间 hidden 形状？谁变成 V？  

>  

**Q3.** labels 全是真实 id、没有 -100，会不会只学助手？  

>  

---

## 本节 Git 作业（必做）

见评论「Git 实验室」；做完在此打勾：

- [ ] `git pull` + submodule 正常  
- [ ] 填写上面三句  
- [ ] `git add` 本文件（及 ledger 若改了）  
- [ ] `commit` + `push`  
- [ ] GitHub 网页能打开本文件  

---

## 下节预告

**L02**：同文件 `MOEFeedForward` + `generate` 逐行；继续练 Git。
