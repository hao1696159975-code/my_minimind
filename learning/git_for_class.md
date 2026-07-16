# 课程里的 Git 用法（每节课都用）

仓库：https://github.com/hao1696159975-code/my_minimind  
上游官方代码在 **submodule** 目录 `upstream/`（一般 **不要改、不要提交进 upstream 的乱改**）。

---

## 1. 每节课固定 Git 节奏

| 时机 | 要不要提交 | 做什么 |
|---|---|---|
| 上课前 | 否 | `git pull` + `git submodule update --init --recursive` |
| 精读中 | 否 | 只读 `upstream/`，笔记先写本地 |
| 本节短答/笔记写完 | **是** | 提交 `learning/` 下笔记 |
| 🔬 小实验出了 md/小脚本 | **是** | 提交 `learning/artifacts/` 或 `scripts/lab/` / `tests/` |
| 跑出 `.pth` 大权重 | **否** | 已被 `.gitignore`；只在 ledger 写路径 |
| 一课完全出门 | 可选 tag | `git tag lesson-L0x`（不必每课都打） |

**口诀：**  
> 笔记、测试、小脚本 → 提交；权重、大数据、密钥 → 绝不提交。

---

## 2. 每天开课前（复制即用）

```bash
cd my_minimind

# 拉远程最新（含老师推的课表/笔记）
git pull origin main

# 确保官方源码在（只第一次或别人更新了 submodule 指针时需要）
git submodule update --init --recursive

# 确认模型文件在
ls upstream/model/model_minimind.py   # Windows: dir upstream\model\model_minimind.py

# 看自己有没有未提交改动
git status
```

若 `git pull` 提示冲突：先 `git status`，把冲突文件发我，**不要强行 `push --force`**。

---

## 3. 一节课结束时怎么提交（标准四步）

```bash
# 1) 看改了什么
git status
git diff

# 2) 只暂存该进库的文件（举例）
git add learning/L01_notes.md
git add learning/experiment_ledger.md
# 不要: git add *.pth   （应被 ignore）

# 3) 写清「这节课留下了什么」
git commit -m "L01: dense mainchain notes + oral answers"

# 4) 推到 GitHub
git push origin main
```

### 好的 commit message 模板

```text
Lxx: <一句话做了什么>

可选第二行：关键文件或实验结论
```

例：

- `L01: notes on Config/Attn/Block/loss shift`
- `L03: SFT label mask sketch + ledger row`
- `L10: pretrain smoke run card (50 steps)`

### 坏的 message

- `update` / `fix` / `asdf` / `1`

---

## 4. 什么绝对不要 `git add`

| 类型 | 例子 | 原因 |
|---|---|---|
| 权重 | `*.pth` `*.bin` `*.safetensors` | 太大；已 ignore |
| 数据 | `*.jsonl` 全量语料 | 太大/许可 |
| 环境 | `.venv/` `__pycache__/` | 本机生成 |
| 密钥 | `.env` token | 安全 |
| 上游乱改 | 直接改 `upstream/` 里官方文件又不说明 | 污染 submodule；实验放 `extensions/` |

检查是否误加：

```bash
git status
# 若出现巨大 bin 文件：
git reset HEAD 路径
```

---

## 5. 和 submodule 相关的三句话

1. `upstream/` 是**嵌套的官方仓库指针**，不是普通文件夹拷贝。  
2. 你日常提交的是**学习仓自己的文件**；`git status` 里若 `upstream` 显示特殊标记，先别乱 commit。  
3. 需要官方更新时由课程指示再 `submodule update`；自己改模型优先拷到 `extensions/`。

---

## 6. 常用急救

```bash
# 看最近提交
git log --oneline -5

# 撤销「还没 commit 的工作区修改」（危险：丢本地未提交改动）
# git checkout -- 文件名

# 撤销「已 add 未 commit」
git restore --staged 文件名

# 改最后一次 commit 说明（仅自己的、且未 push 时）
git commit --amend -m "新说明"
```

**已 push 的历史不要 amend / force push**，除非老师明确说可以。

---

## 7. 和 24 课表的绑定

每课结束检查清单多一项：

- [ ] 笔记或 ledger 已更新  
- [ ] `git status` 干净或只剩有意保留的本地文件  
- [ ] 该提交的已 `push`，GitHub 网页能看见  

L01 起强制练习；到 L09 你应能独立完成 pull → 实验 → add → commit → push。
