# 引擎补丁（路线 A · 原生 EXL3）

路线 A 依赖两个上游仓库的**未合并改动**。这里以补丁形式保存，因为直接推分支出了一点状况（见下）。

---

## 为什么是补丁而不是分支

原本计划把 exllamav3 的三个分支推成 `exllamav3/*`，被 **GitHub Push Protection 拦下**：

```
remote: error: GH013: Repository rule violations found for refs/heads/exllamav3/master
remote:     - Push cannot contain secrets
remote:       —— Mistral AI API Key ——
remote:        locations:
remote:          - commit: d8213dc...  path: README.md:32
remote:          - commit: 93de4d4...  path: README.md:44
remote:          ... （共 5 个提交）
```

**这是上游 fork 继承的内容，与我们的工作无关**——密钥已不在当前 `README.md`，
只存在于 5 个旧提交的历史里。GitHub 会扫描**所有被推送的提交**而非仅分支顶端，所以整条历史都被拒。

两条路：

1. **绕过**（保留完整 1800+ 提交历史）——在仓库的 secret scanning 页面授权一次，然后重推：
   `https://github.com/Ajie16/qwen3.8-flash-spark-1x/security/secret-scanning/unblock-secret/3KMFSLIwCov3D6W1gSs6e8NPzzS`
2. **用补丁**（本目录）——内容等价，但没有提交历史，且需要干净的 exllamav3 检出才能应用

---

## 补丁清单

| 文件 | 大小 | 提交数 | 内容 |
|---|---|---|---|
| `feat-hybrid-draft.patch` | 189 KB | 23 | `EXL3_GR_TUNED` 融合 GR kernel（R ≤ 4）、hybrid ngram 宽度钳制与自适应退避、`EXL3_NGRAM_MAX_DRAFT` 默认改 7（前向 kernel 在 q_len>8 有悬崖）、M3/M4 剖析脚本与 CPU 测试 |
| `feat-spec-sampling.patch` | 90 KB | 14 | 精确推测采样验证路径（`generator/spec_sampling.py`）、握手与残差概率原语、批量 transform 校验、`docs/spec_sampling.md` |
| `tabbyapi-local.patch` | — | 1 | `local: allow max_history bump under EXL3_HYBRID_NGRAM` |

对应的提交清单在 `*.commits.txt`（含作者、日期、提交信息）。

---

## 应用

### exllamav3

需要一个与补丁基线一致的检出。基线是 `vcruz305/exllamav3` 的
`fix/gb10-uma-and-draft-window` 分支（本机在 `74b6f5a`）。

```bash
git clone git@github.com:vcruz305/exllamav3.git
cd exllamav3
git checkout 74b6f5a                       # 补丁的基线
git checkout -b feat/hybrid-draft
git apply /path/to/feat-hybrid-draft.patch
```

两个补丁**互相独立**，各自分支自成一条链（`feat/spec-sampling` 的起点是 `5e52ac9`），
不要同时应用。

### TabbyAPI

```bash
git clone https://github.com/theroyallab/tabbyAPI.git
cd tabbyAPI
git checkout f07131c                       # 补丁的父提交
git apply /path/to/tabbyapi-local.patch
```

---

## 本机的实际布局（供对照）

```
~/qwen38-exl3/
├── exllamav3/          # 主检出，master @ 74b6f5a（已推送到 vcruz305 fork）
├── exllamav3-spec/     # ← 同一仓库的 git worktree，检出 feat/hybrid-draft
├── tabbyAPI/           # detached HEAD @ b8c0497（本地提交）
├── state/config.yml    # 由 serve.sh 渲染
└── venv/               # exllamav3 可编辑安装，.pth 指向 exllamav3/
```

注意：`serve.sh` 运行时用的引擎是 **`exllamav3-spec/`**（`feat/hybrid-draft` worktree），
不是 `.pth` 指向的 `exllamav3/`。看服务日志开头那行：

```
exllamav3 1.5.1.post1 (fork) at /home/xujie/qwen38-exl3/exllamav3-spec/exllamav3
```
