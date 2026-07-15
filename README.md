# my_minimind

基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 的 **Dense + MoE 全链路工程化学习仓**。

> 目标不是「跑通官方脚本」，而是：读懂契约 → 可复现实验 → 可测试扩展 → 能写进简历的改进。

## 上游

| 项 | 值 |
|---|---|
| 上游仓库 | https://github.com/jingyaogong/minimind |
| 主线规格 | minimind-3 **64M Dense** / minimind-3-moe **198M-A64M** |
| 本仓角色 | 学习笔记、实验配置、测试、扩展；上游代码放 `upstream/`（submodule 或 vendor） |
| 许可 | 上游 Apache-2.0；本仓笔记与扩展默认同许可，以 `LICENSE` 为准 |

## 当前进度

| Phase | 主题 | 状态 |
|---|---|---|
| 0 | 仓库地图 + 工程底座 | 🔄 in progress |
| 1 | Tokenizer / Dataset 契约 | ⏳ |
| 2 | 模型审查（Dense + MoE 分支） | ⏳ |
| 3 | 生成 / KV Cache / 推理工程 | ⏳ |
| 4 | Pretrain 工程化（Dense smoke + MoE 对照） | ⏳ |
| 5 | SFT 监督信号治理 | ⏳ |
| 6 | LoRA 对照 | ⏳ |
| 7 | MoE 可观测性（专章） | ⏳ |
| 8 | 蒸馏 + DPO | ⏳ |
| 9 | PPO / GRPO / CISPO | ⏳ |
| 10 | Tool / Agentic RL 安全 | ⏳ |
| 11 | 评测 / 转换 / 部署 / 模型卡 | ⏳ |
| 12 | Capstone 扩展（论文级小改进） | ⏳ |

掌握规则：**证据制**。口述对了但无代码/实验/测试，不算 mastered。

## 目录

```text
my_minimind/
├── upstream/                 # 官方 MiniMind（submodule 或 clone）
├── learning/                 # 笔记与实验账本
├── exercises/                # 一次性课堂练习
├── extensions/               # 可独立运行的正式扩展（简历主产物）
├── tests/                    # 我方补充回归测试
├── configs/learning/         # smoke / toy / 4090-small
├── runs/                     # 实验输出（默认 gitignore）
├── scripts/lab/              # 参数量、mask 可视化、routing 统计
└── README.md
```

## 快速开始（Phase 0）

```bash
# 1) clone 本仓
git clone https://github.com/hao1696159975-code/my_minimind.git
cd my_minimind

# 2) 拉取上游（二选一）
# 方式 A：submodule（推荐，提交清晰）
git submodule add https://github.com/jingyaogong/minimind.git upstream
git submodule update --init --recursive

# 方式 B：普通 clone
git clone --depth 1 https://github.com/jingyaogong/minimind.git upstream

# 3) 环境（在服务器 4090D 上优先）
cd upstream
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

详细地图与验收：见 [`learning/00_project_map.md`](learning/00_project_map.md)。

## 简历向改进路径（防「只学皮毛」）

改进不是 Phase 0 就硬上论文。正确顺序：

1. **先吃透契约**（数据 mask / shape / checkpoint）——否则改了也不知道对不对  
2. **建立基线**（Dense mini pretrain+SFT 可复现数字）  
3. **单点扩展**（一个可测的改进 + 对照表）  
4. **再谈论文迁移**（把论文思想落到一个可插拔模块）

Capstone 候选（Phase 12，可提前记愿望清单）：

| 方向 | 简历可写点 | 前置 Phase |
|---|---|---|
| 可插拔 Router / load balancing 日志 | MoE 可观测 + 防 collapse | 7 |
| Loss mask / 数据契约测试套件 | 工程正确性与 CI | 1,5 |
| 实验管理系统（config+run+metrics） | 可复现训练平台感 | 0,4 |
| DPO/GRPO 最小正确性测试 | 对齐算法工程化 | 8,9 |
| 推理服务（流式/限流/模型卡） | 部署闭环 | 11 |
| 论文级小改（如 attention/RoPE/aux loss 变体） | 研究迁移能力 | 2+ 基线后 |

愿望清单写在：`learning/improvement_backlog.md`。

## 硬规则

1. 不污染上游主路径；实验放 `learning/` / `extensions/` / `configs/learning/`  
2. 长训前必须 smoke（极小配置 / 短 step）  
3. 每次实验写 `learning/experiment_ledger.md`  
4. 改实现前：先失败测试 / shape 断言  
5. 教学简化 ≠ 生产最佳实践（笔记里显式标注）

## 相关

- 官方：https://github.com/jingyaogong/minimind  
- 课程 issue：Multica WS-6 minimind学习  
- 前置：CS224N 5 日冲刺 + CS336 α1–α8 已完成
