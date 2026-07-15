# Experiment Ledger

规则：
1. 每次训练 / 关键 dry-run / 对照实验 **一行**  
2. 大日志与权重放 `runs/`，这里只留指针  
3. `Repro?=Y` 仅当：固定 seed + 配置文件 + 数据版本 + 命令可复述

| Date | Phase | Goal | Config | Seed | Data | Device | Result | Artifact | Repro? | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-15 | 0 | 建仓骨架 + 课程锁定 | n/a | n/a | n/a | local workdir | scaffold ready | repo root | Y | 等服务器 clone 上游与最小推理 |
