# MiniMind 课程 · 每节课安排总表

> 版本：v1（2026-07-15）  
> 仓库：https://github.com/hao1696159975-code/my_minimind  
> 上游：`upstream/` = jingyaogong/minimind  
> 教学形态：**手把手对源码**（打开文件 → 按行号讲 → 小例子 → 2～3 句确认 → 再进下一课）  
> 默认节奏：**标准 6 周**（每周约 10–14h / 约 3 节课）；可压成 4 周或拉成 8 周  

---

## 0. 怎么用这张表

### 0.1 一节课的标准结构（固定 60–90 分钟）

| 时段 | 分钟 | 做什么 |
|---|---|---|
| 回顾 | 5–10 | 上节 2～3 句 + 仍卡的点 |
| 精读 | 35–50 | **一个**文件/模块，对行号 + 小数字例子 |
| 最小实验 | 10–20 | shape / mask / 1-step / 日志（能 CPU 就先 CPU） |
| 确认 | 5–10 | 本课 Q1–Q3（短答） |
| 落盘 | 5 | 更新 `experiment_ledger.md` 或笔记 3 行 |
| **Git** | 5–10 | 见 `git_for_class.md`：该提交则 add/commit/push |

### 0.2 出门证规则

- 没交本课 **Q 短答** → 不进下一课精读  
- 没做标了 🔬 的实验 → 该 Phase 不能标 mastered  
- 长训前必须有 smoke（极小配置）  
- **从 L01 起**：有笔记/实验产物的课，结束应能 `push` 到 GitHub（无大文件）  

### 0.3 当前进度指针

| 项 | 状态 |
|---|---|
| 课程方案 | ✅ Dense 主线 + MoE 专章 |
| 学习仓 scaffold | ✅ |
| upstream submodule | ✅ 需 `git submodule update --init --recursive` |
| **L01 模块1上：Dense 主链** | 🔄 正式开课中（含 Git 练习） |
| L02 起 | ⏳ 未开 |

---

## 1. 总览：24 节课（L01–L24）

| 阶段 | 课号 | 主题 | 建议周次 |
|---|---|---|---|
| A 读代码底座 | L01–L08 | 模型 / 数据 / 训练环 / 推理 | W1–W2 |
| B Dense 跑通 | L09–L12 | Pretrain smoke + SFT mask + resume | W2–W3 |
| C 高效微调与 MoE | L13–L16 | LoRA + MoE 可观测 | W3–W4 |
| D 对齐与 RL | L17–L20 | DPO + PPO/GRPO 门槛 | W4–W5 |
| E 部署与作品 | L21–L24 | Agent 安全 + 服务 + Capstone | W5–W6 |

图例：📖 精读 · 🔬 实验 · ✍️ 短答 · 🧳 产物

---

## 2. 分节课计划表（核心）

### A. 读代码底座（W1–W2）

| 课 | 标题 | 打开文件（upstream/） | 当堂精读重点 | 🔬 最小实验 | ✍️ 出门 3 问类型 | 🧳 产物 | 状态 |
|---|---|---|---|---|---|---|---|
| **L01** | Dense 主链上 | `model/model_minimind.py` L1–253 | Config→Attn→FFN→Block→LM+loss | 纸上 shape：`[B,T]→[B,T,H]→[B,T,V]` | Pre-LN/残差；hidden 形状；-100 责任 | 笔记 3 行 | 🔄 进行中 |
| **L02** | MoE + generate | 同上 L148–176, L256–288 | top-k、aux、未选中专家；cache 生成循环 | 指出 `use_moe` 唯一分叉行 | 激活参；aux 加哪；cache 喂什么 | 补全 module1 笔记 | ⏳ |
| **L03** | Dataset 契约 | `dataset/lm_dataset.py` | Pretrain vs SFT labels；`generate_labels` | 打印 1 条 SFT 的 input/labels 对齐（可 mock） | pad=-100；assistant 怎么定位；截断风险 | `learning/01_data_contracts.md` 草稿 | ⏳ |
| **L04** | DPO/RL 数据预告 | 同上 DPO/RLAIF/Agent 类 | chosen/rejected；prompt-only；gt | 对照三类 `__getitem__` 返回值 | 三类返回字段差异 | 数据契约表 | ⏳ |
| **L05** | 训练公共件 | `trainer/trainer_utils.py` | `init_model` / `lm_checkpoint` / `get_lr` / DDP | 读懂 ckpt 文件名规则 | `_moe` 后缀；resume 含什么 | 权重命名卡片 | ⏳ |
| **L06** | Pretrain 脚本 | `trainer/train_pretrain.py` | 1 step：forward→accum→clip→step | 对照 loss=`ce+aux` 那几行 | 与模型 loss 如何衔接 | `04` 笔记骨架 | ⏳ |
| **L07** | SFT 脚本对照 | `trainer/train_full_sft.py` | 与 pretrain 同环、数据不同 | diff 两个 train 文件 | 监督信号差在哪 | 对照表 | ⏳ |
| **L08** | 推理入口 | `eval_llm.py` + 回看 generate | 加载 torch/HF；chat template | `--help` + 无权重时读加载路径 | pretrain vs sft 输入格式 | Phase0 地图收口 | ⏳ |

**A 段出门证：** 能指源码讲完 Dense forward + SFT mask 责任边界 + ckpt 命名。

---

### B. Dense 跑通（W2–W3）

| 课 | 标题 | 打开文件 | 当堂重点 | 🔬 实验 | ✍️ 出门 | 🧳 产物 | 状态 |
|---|---|---|---|---|---|---|---|
| **L09** | 环境与参数量 | `scripts/lab/*`（学习仓） | submodule、依赖、count_params | checklist + Dense/MoE numel | 数字是否合理 | `artifacts/phase0_param_*.txt` | ⏳ |
| **L10** | Pretrain smoke | `train_pretrain.py` + mini 数据 | 极短 step、seed、日志 | 20–50 step 不求效果 | 初始 loss 量级 | run card + ledger 一行 | ⏳ |
| **L11** | SFT mask 实战 | `lm_dataset.py` + `train_full_sft.py` | 错 mask 对照（思想实验或小脚本） | 可视化哪些 token 进 loss | 假 loss 下降场景 | mask 测试草稿 | ⏳ |
| **L12** | Checkpoint / resume | `trainer_utils.lm_checkpoint` | save/load optimizer+step | 中断再续，step 对齐 | 只 load 权重的坑 | resume 检查记录 | ⏳ |

**B 段出门证：** Dense mini 可复现 smoke + 理解 resume；未过禁止 MoE 长训。

---

### C. LoRA 与 MoE（W3–W4）

| 课 | 标题 | 打开文件 | 当堂重点 | 🔬 实验 | ✍️ 出门 | 🧳 产物 | 状态 |
|---|---|---|---|---|---|---|---|
| **L13** | LoRA 注入 | `model/model_lora.py` | 方阵 Linear、B=0、merge | 列出实际带 `.lora` 的层名 | 与 PEFT 差异 | `06_lora` 笔记 | ⏳ |
| **L14** | LoRA 训练对照 | `train_lora.py` | 只训 lora 参数 | 参数量/是否冻结 base | full vs lora 表头 | 对照表空表 | ⏳ |
| **L15** | MoE 路由精读 | `MOEFeedForward` 再读 | load、aux、index_add | 伪代码画 top-1 | collapse 是什么 | `07_moe` 笔记 | ⏳ |
| **L16** | MoE 可观测实验 | 自写 `scripts/lab/routing_stats.py`（可指导） | expert 使用频率 | mini MoE 短跑 + 直方图 | 不均衡如何看 | routing 图/表 | ⏳ |

**C 段出门证：** 能解释 MoE 激活参 + 会看 expert load；LoRA 知道实际注入面。

---

### D. 对齐与 RL 门槛（W4–W5）

| 课 | 标题 | 打开文件 | 当堂重点 | 🔬 实验 | ✍️ 出门 | 🧳 产物 | 状态 |
|---|---|---|---|---|---|---|---|
| **L17** | DPO 公式对源码 | `train_dpo.py` | `logits_to_log_probs` / `dpo_loss` / ref | 小 batch dry 概念 | beta、mask、ref 冻结 | 符号对照表 | ⏳ |
| **L18** | 蒸馏（可选压缩） | `train_distillation.py` | soft/hard、T、alpha | 读 loss 函数即可 | 何时用蒸馏 | 8 字笔记 | ⏳ |
| **L19** | Rollout + PPO 骨架 | `rollout_engine.py` + `train_ppo.py` | logprob、reward、GAE 位置 | 不长训，只跟调用链 | old policy vs ref | RL 地图 | ⏳ |
| **L20** | GRPO/CISPO 对照 | `train_grpo.py` | group norm；启发式奖励风险 | 找 `calculate_rewards` | reward hacking 例 | 三算法对照表 | ⏳ |

**D 段出门证：** 能指源码说 policy/reward/advantage/KL（实现级）；RL 只 dry-run。

---

### E. Agent、部署、毕业（W5–W6）

| 课 | 标题 | 打开文件 | 当堂重点 | 🔬 实验 | ✍️ 出门 | 🧳 产物 | 状态 |
|---|---|---|---|---|---|---|---|
| **L21** | Agentic RL 与安全 | `train_agent.py` | 多轮 tool、gt、未完成惩罚 | 威胁模型 5 条 | 投机/泄漏点 | `10_agent_safety` | ⏳ |
| **L22** | 转换与服务 | `scripts/convert_model.py` + `serve_openai_api.py` | torch↔HF；API 边界 | 读入口参数；有权重再 serve | 教学 vs vLLM | 部署笔记 | ⏳ |
| **L23** | Capstone 设计 | `improvement_backlog.md` | 选 I1/I3/I7… 定接口 | 写设计 1 页 + 测试计划 | 验证指标是什么 | `12_capstone` 设计 | ⏳ |
| **L24** | Capstone 实现收口 | `extensions/` + `tests/` | 实现 + 对照 + README | 前后数字/截图 | 3 分钟面试口述 | `v1.0-portfolio` tag | ⏳ |

**E 段出门证：** GitHub 可演示 + 模型卡/局限 + 面试稿。

---

## 3. 周历（标准 6 周）

| 周 | 课 | 主题目标 | Git 节点建议 |
|---|---|---|---|
| **W1** | L01–L04 | 模型+数据读懂 | `lesson-L04` |
| **W2** | L05–L09 | 训练环+环境+参数量 | `lesson-L09` |
| **W3** | L10–L13 | Dense smoke + SFT + LoRA 起 | `phase-pretrain-smoke` |
| **W4** | L14–L17 | LoRA 完 + MoE 可观测 + DPO | `phase-moe-obs` |
| **W5** | L18–L21 | RL 门槛 + Agent 安全 | `phase-rl-map` |
| **W6** | L22–L24 | 部署 + Capstone + 发布 | `v1.0-portfolio` |

### 加速 4 周（每周 18–25h）

合并：L03+L04、L06+L07、L18 略、L19+L20 压缩为 1 课；Capstone 只做 **I1 或 I3 其中一个**。

### 深挖 8 周

W7：完整 GRPO 小跑 + 投机案例；W8：服务 hardening + 面试打磨 + 第二扩展。

---

## 4. 与 Phase 体系的映射

| Phase（原方案） | 覆盖课 |
|---|---|
| 0 地图底座 | L01 部分、L08–L09 |
| 1 数据契约 | L03–L04、L11 |
| 2 模型审查 | L01–L02、L15 |
| 3 推理 | L02 generate、L08 |
| 4 Pretrain | L05–L06、L10、L12 |
| 5 SFT | L07、L11 |
| 6 LoRA | L13–L14 |
| 7 MoE | L15–L16 |
| 8 蒸馏/DPO | L17–L18 |
| 9 RL | L19–L20 |
| 10 Agent | L21 |
| 11 部署 | L22 |
| 12 Capstone | L23–L24 |

---

## 5. 每课作业与仓库落盘规范

```text
learning/
  lesson_schedule.md          # 本表
  source_tour_XX_*.md         # 精读笔记
  0X_*.md                     # Phase 总结（证据够再写）
  experiment_ledger.md        # 每节有实验就加一行
  artifacts/Lxx_*.md          # 轻量输出
exercises/Lxx_*/              # 一次性练习
extensions/                   # 仅 L23–L24 正经扩展
tests/                        # 从 L11/L12 起补回归
```

提交节奏：每完成 **2 课** 或一个 **🔬 实验** → 1 个 commit（不要攒爆）。

---

## 6. 立即执行：下一课是谁

1. **现在：** 完成 **L01** 的 Q1–Q3（已在上条评论）  
2. **下一课默认：L02** MoE + generate 手把手  
   - 若你更卡 labels，可插队 **L03**（说一声即可）  
3. 仍缺：每周小时数、主训 GPU/SSH（影响 L09 起实验是否我代跑）

---

## 7. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-07-15 | v1：24 课表 + 6 周历 + 与 Phase 映射；指针停在 L01 |
