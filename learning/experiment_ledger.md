# Experiment Ledger

规则：
1. 每次训练 / 关键 dry-run / 对照实验 **一行**  
2. 大日志与权重放 `runs/`，这里只留指针  
3. `Repro?=Y` 仅当：固定 seed + 配置文件 + 数据版本 + 命令可复述

| Date | Phase | Goal | Config | Seed | Data | Device | Result | Artifact | Repro? | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-15 | 0 | 建仓骨架 + 课程锁定 | n/a | n/a | n/a | local workdir | scaffold ready | repo root | Y | 首版 upstream 仅 .gitkeep，用户找不到 model 文件 |
| 2026-07-15 | 0 | 修复：官方仓挂 submodule | submodule pin | n/a | n/a | local workdir | upstream 含 model_minimind.py | `.gitmodules` + `upstream@512eed0` | Y | 用户需 `git pull` + `git submodule update --init --recursive` |
| 2026-07-15 | 0 | Source Tour 00/01 预习开课 | n/a | n/a | n/a | docs | 全仓地图+model 精读 | `learning/source_tour_00_map.md` | Y | 用户要求先过源码再正式上课 |
