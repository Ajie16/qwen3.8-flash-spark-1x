# EXL3-FIX.md — TensorFold EXL3 原生引擎优化全记录

> 机器：spark-2（10.100.64.1）· 引擎：TensorFold 0.6.1（venv `~/tensorfold-venv`）· 包：Qwen3.8-Flash-Next-Uncensored-EXL3-3bpw（Lygodactylus）
> 本文档是 EXL3 原生 CUDA 引擎所有优化/补丁/实验的**唯一权威记录**，skill（`~/.kimi-code/skills/dgx-llm-local/SKILL.md`）与本文件冲突时以本文件为准。
> 最后核实：2026-10-07（逐项 grep 验证过当前 venv 应用状态）。

---

## 0. 当前状态速查

| 类别 | 内容 | 状态 |
|---|---|---|
| 服务补丁 | 思考预算夹取 / gate 不截半 / SSE ping / 调度线程兜底（4 个） | ✅ **已应用**（`cuda/server.py` `cuda/http.py` `engine/call_gate.py` `cuda/scheduler.py`） |
| 采样调优 | `--mtp-confidence 0.6 --mtp-drafts 6 --kv-dtype int8 --parallel 4` | ✅ 生产配置 |
| 草稿词表 | 中文增强表 102,089 id（`default ∪ Mia-zh ∪ 语料反查 410`） | ✅ 已应用 |
| 内存预算 | `TENSORFOLD_MEMORY_RESERVE_GIB=4`（KV 池 58 GiB） | ✅ 生产配置 |
| 内核实验（10-03/04） | MOE_WINDOW / fillslice / nt=16 / pf=2 / expert-rows dump | ⛔ **全部已回滚**，当前 venv 无残留 |
| 0.3.6.3 时代 | prefix-cache patch / KEEP=16 / staging-8192 patch | 见 §3（部分已被 0.6.1 原生取代） |

**内核实验全部回滚的原因**：实测收益均不成立或证伪，具体见 §2。逐项勿重做。

---

## 1. 生产已固化的优化（勿动）

### 1.1 中文草稿词表（MTP 最大单项收益）

TensorFold 的 MTP 草稿只能从 `.../qwen4_exp/cuda/draft_vocab.txt` 列出的 id 里挑 token。默认表 79,591 个几乎全英文/代码，中文散文接受率仅 18.8%。

三代迭代（venv 内文件均已替换，备份在 `~/tensorfold-patches/`）：

| 版本 | id 数 | 中文散文 | 说明 |
|---|---|---|---|
| blunt111k（粗暴扩段） | 111,492 | 33.0 tok/s | 旧表，已废 |
| **Mia `default ∪ zh`** | 101,679 | **39.8** | 覆盖到 24.8 万段高 id 中文 token |
| **语料反查增强（现役）** | **102,089** | **41.0（语料内 +8.5%）** | 见下 |

**语料反查方法（可复用，脚本 `~/zh_vocab_audit.py` + `~/zh_vocab_build.py`）**：把模型自己的中文产出用本包 tokenizer 分词，统计高频 id 中不在草稿表的——Mia 通用表只覆盖真实语料的 87%，缺的 410 个 id 占 13% token，且全是多字词（`一致性`377 次、`分布式`151 次……），正是投机解码最赚的地方。要长期吃收益，把 dsh/kimi 真实中文流量攒起来周期性重跑。

注意：**提升主要来自词表变小（每步草稿打分更便宜），接受率几乎没变**（43.0%→42.5%）。

### 1.2 MTP 调参结论（接受率越高 ≠ 越快）

| 配置 | 平均 tok/s | 接受率 |
|---|---|---|
| **`--mtp-confidence 0.6 --mtp-drafts 6`（现役）** | **45.7** | 56.1% |
| conf 0.70 / 0.80（0.6.1 新默认 0.70） | 44.7 / 44.4 | 60.3 / 62.6% |
| drafts 10（conf 0.6 / 0.7） | 44.2 / 45.2 | 53.7 / 58.6% |
| conf 0.40 / 0.50 | 42.5 / 43.6 | 46.5 / 51.5% |
| `--kv-dtype int4` | 44.6 | 55.3% |

置信度调高 → 草稿链更早截断 → `accepted/drafted` 比值上去但每轮实际产出减少。**别被接受率误导**。
`--mtp-drafts 9` 与 6 完全打平（confidence 在 6 层前已截断链），不加。

### 1.3 内存预算与 KV 池

预算公式（`tensorfold/cuda/capacity.py`）：`budget = 启动瞬间可用统一内存 − reserve`，`reserve = max(4, total/10) = 12.17 GiB`，可用 `TENSORFOLD_MEMORY_RESERVE_GIB` 覆盖（下限 2）。

- EXL3 生产：`TENSORFOLD_MEMORY_RESERVE_GIB=4` → KV 池 48.4 → **58.0 GiB**（单流满窗 4.47 GiB，够十几条满长流）
- 0.6.1 的 n-gram 表**能 locked in memory**（0.3.6.3 时代 MLX 包那次不行，会 page cache 换页）
- `--prompt-cache-gib` 不传时前缀检查点自动吃满空闲内存（`grow_checkpoints`），调大预算 = 同时放大并发窗口 + 前缀缓存
- reserve 再压（2 GiB）可再多 2 GiB KV，但内核余量过薄；实测压到 4 时 n-gram 表换页读盘仍为 0

### 1.4 并发档位

`--parallel 4`（2026-10-03 从 6 降下来：4 路足够且每路能长到满窗口）。CUDA graphs 只在 `--parallel 1` 串行路径（+13% decode：prose 46.6/code 49.3），与 `--vision` 互斥且无并发——生产选 parallel 4，graphs 仅作可选档记录。

### 1.5 瓶颈定量（为什么内核实验全部无效）

- **prefill 820 T/s = EXL3 trellis 反量化的逐 token 计算墙**。证伪实验：staging patch 把 chunk 从 2.6k 改到 42k，prefill 恒定 ~820 T/s——与分块/调度无关，没有任何开关能绕过去（对比：MLX 4bit affine 的 dequant 是乘加，可融进 tensor core matmul，同机 2400+ T/s）
- **decode 70% 是 MoE 权重带宽墙**（有效 ~155 GB/s / 可持续 ~220），28% 是 eager + 串行草稿税
- 完整分析脚本：`~/exl3_prefill_math.py`（fp16 scratch 往返账）、`~/exl3_layer_timing.py`（unpack vs _gemm 分解）

---

## 2. 内核实验全记录（2026-10-03/04，已回滚，勿重做）

实验动机：§1.5 说 prefill 是计算墙——但墙里有"重复劳动"成分（expert 权重被反复反量化），于是有了这批"省重复 decode"的实验。脚本全部幂等（`SKIP already applied`），备份在 `~/tensorfold-patches/pre-<name>-<时间戳>/`。

### 2.1 MOE_WINDOW 1024→2048（`~/patch_moe_window.py`）⛔ 已回滚

**机制**：EXL3 路由专家按窗口处理，`expert 反量化总次数 = tokens / MOE_WINDOW`（chunk 在公式里约掉，窗口是唯一杠杆）。1024 → 2048 等于每 pass 少扫一半专家权重。
**上限论证**（脚本 docstring）：grouping 用 `R*slots*4` 字节 shared memory，GB10 实测 optin=101,376 B → R=2048 需 90,112 B 恰好放下，R=4096 需 180,224 B 放不下。
**处置**：已回滚为 1024。microbench 脚本 `~/exl3_window_cost.py`（对比 T(2048) vs 2·T(1024)）可复跑。
**为什么没留下**：窗口翻倍后单窗口内每个专家服务更多行（见 2.5 数据，均值 42.5 行/窗口），理论上摊薄反量化；但 §1.5 已证 prefill 瓶颈是逐 token 计算而非扫描次数，实测未过门槛。

### 2.2 并发 prefill 时间片（`~/patch_fill_slice.py`）⛔ 已回滚

**机制（这是一个真 bug 修复候选）**：实测 4 个冷 prompt 同时到达（prefilling=4），聚合吞吐却只有 786 tok/s、最后一个等 149.7s——重叠因子 0.99×，纯串行。根因：`multi.py _pieces()` 把整窗交给 `_order()` 排第一的流，而 `_order()` 按"剩余行最少"排序——恰是刚吃掉最多行的那个流，于是每个 pass 都由同一流垄断；`FILL_GUARD=8` 本意轮转，但 `due` 排序使流一旦 due 就永久 due，守卫反转为锁定。
**修复**：`share = max(PASS_MIN, room // len(order))`，把每个 pass 的行数均分给等待中的流；单流时 share=整窗，**单流 prefill 逐字节不变**。
**处置**：已回滚（当前 `_pieces` 为原始形态）。
**何时该重新启用**：如果生产上出现"多客户端同时提交长 prompt 时最后一个等全场"的症状，这个 patch 是对的修复；当时回滚的决策数据未落盘，重新启用前请用 `exl3bench.py` 单流/并发各跑一遍对照。

### 2.3 down 投影 N tile 128→256（`~/patch_ntile.py`）❌ 证伪

GLM_DOWN=(8,4,1,1) 把 down 投影 N tile 钉在 128 列；nt=16（256 列）算术合法（K=640 % 64==0，N=2560 % 256==0）。**结果：直接 RuntimeError**——`experts_grouped.cuh:336-338` 只编译了 (8,4,1)、(8,4,2)、(4,4,2) 三种 tile，nt=16 无实例。gate_up 连试都不能试（N=640，nt≤8）。exllamav3 同款模型同样钉 128 列，同源约束。

### 2.4 pf=2 寄存器预取候选（`~/patch_pf2.py` + `~/patch_pf2_order.py`）⛔ 已回滚

**机制**：`(nt, warps, sk, pf)` 四元组里 pf 是 warp tile 级寄存器预取深度；`(8,4,2)` 已编译但 `default_config` 的候选列表够不到它（全带 pf=1），把 pf=2 候选加进去即可用，无需新编译。
**两次尝试**：① 追加在列表尾——无效，GLM_GATEUP/GLM_DOWN 先通过整除测试，pf=2 条目轮不到；② 前置——`default_config` 真能选到了。
**处置**：已回滚（当前候选列表为原始 5 项）。两次尝试间基线 769-773 tok/s，最终未保留——叠加 §1.5 的 decode 带宽墙结论（70% 是带宽），寄存器预取对带宽瓶颈本来就作用有限。

### 2.5 expert-rows 测量钩子（`~/patch_expert_rows.py` + `~/patch_expert_rows_fix.py`）✅ 已完成使命并移除

**目的**：给"热门专家反量化一次交给 cuBLAS"（exllamav3 在 >256 行的档位，`moe_batch_recon.py:128`）这最后一个理论杠杆定量。
**实现**：`TENSORFOLD_DUMP_EXPERT_ROWS=<path>` 武装后，prefill 每 pass 收集 48 层的 per-expert 行数，`~/expert_rows.pt`（48 窗口 × 49,152 token 实测数据）。
**数据（`~/analyze_expert_rows.py`，已复跑核实）**：

```
per-expert 行数分位: p10=1  p50=12  p90=99  p99=505  mean=42.5
工作占比: 1-16行 7.9% | 17-32 8.9% | 33-64 13.3% | 65-128 17.0% | 129-256 18.4% | 257+ 34.6%
冗余度: ≥256行的专家(34.9%工作量)今天被重复反量化 31.7×
       exllamav3 的 4096 窗口线性外推 → 70%+ 工作可一次反量化
```

**结论**：杠杆真实存在（热专家 31.7× 冗余反量化），但实现要吃掉整个 expert kernel 路径（tensorfold 每 16 行块重跑 `decode_tile`，experts_grouped.cuh:174），工程量与风险不成比例——尤其 §1.5 已认定 prefill 主瓶颈是逐 token 计算墙。**处置**：钩子已从 venv 移除（grep 验证 0 残留），`expert_rows.pt` 与分析脚本保留供日后参考。
**踩坑记录**：第一版把 helper 插在 `routed()` 调用点之后，列 0 的 def 提前结束函数，后续代码掉进 helper 体（NameError: ext）；第二版读了 `members` 的脏尾部（复用 scratch，早期 pass 残留），最大值 426 vs slots=11 一眼假——**必须切 `members[:s.count]`**。

---

## 3. 0.3.6.3 时代的优化（历史存档）

| 项目 | 处置 |
|---|---|
| **prefix-cache patch（自研）** | 0.3.6.3 上解决思考模式多轮缓存不命中（prefill 3.6s→0.2s，~18×）；**0.6.1 已原生 resume，patch 退役**（备份 `~/tensorfold-patches/tf-prefix-cache.patch`） |
| KEEP=16（`_remember` 容量） | 0.3.6.3 时扩（system 块条目不被挤掉）；0.6.1 语义变化后由原生 checkpoint slots 接管 |
| staging 8192 patch | **已证伪**（§1.5），env 已从 tf-serve.sh 撤掉；备份 `~/tensorfold-patches/engine.py`、`exl3.py` |
| `TENSORFOLD_MTP_COPY=1` | 对 bench 负载零触发（COPY_MATCH=8 要求末尾 8 token 精确重复），留着等真实重复场景 |
| 4 个服务补丁（silence/reserve） | **已应用到 0.6.1**，见 §4 |

## 4. 服务补丁明细（0.6.1，2026-10-03 打，已验证在 venv）

症状背景：agent 频繁空回复（`content: null` / `finish_reason: length`）。根因：`--thinking-budget` 只有上限没预留——`engine/call_gate.py` 按 `[:max_tokens - len(reply)]` 截断，预算 ≥ max_tokens 时 `</think>` 一个字符都写不进 → `split_thinking` 判整条为 reasoning → `content: null`。而 `--max-tokens` 默认 4096、当时正好 `--thinking-budget 4096`，凡不发 max_tokens 的客户端必中。

| # | 文件 | 改动 | 效果 |
|---|---|---|---|
| 1 | `cuda/server.py` `_think_budget` | 预算夹到 `max_tokens - max(256, max_tokens//4)`；不够放 close 就不切（给答案留 1/4，不是只留 close 长度——第一版只留 close，客户端收到 2 个换行符） | 384/768/2048 预算下空回复 → 464/447/1070 字 |
| 2 | `engine/call_gate.py` `take()` | 放不下的 fix 整个不应用，不再截半（半个 `</think>` 比不切更糟） | — |
| 3 | `cuda/http.py` | 静默 ≥10s 补一行 `: ping`，所有写同一把锁 | 排队请求最长字节间隔 92.9s 零字节 → 10.0s（10s 读超时的客户端不再被切） |
| 4 | `cuda/scheduler.py` `_loop` | 整个轮次包 try/except（原 `_yield/_admit/finish` 在 try 外，一次异常 daemon 线程永久退出，`/health` 照样 200，之后每个请求永久挂起） | "服务看着好好的就是没下文"根治 |

脚本：`~/patch_ttsilence.py`（#1/#2）、`~/patch_tt_reserve.py`（#3/#4），幂等可重放；逐文件备份 `~/tensorfold-patches/pre-061-{silencefix,reserve}-*/`。

客户端配套：dsh `cordis.patch.yml` 加 `thinkingBudgets` + `compat.thinkingTokenBudgetField`；`switch_to_mlx.sh` 用 `--thinking-budget 1024 --max-tokens 32768` 兜底不发 max_tokens 的客户端。

**已知未修**（重启才能改，按需）：stop 字符串命中思考部分会整段删答案并报 `finish_reason: "stop"`（对 agent 最毒）；排队期间不检测客户端断连；满池不返 429/503，无界排队；无空回复计数器。

---

## 5. 复现 / 回滚 / 验证

```bash
# 验证当前应用状态（应输出：2 / 1 / 0）
V=~/tensorfold-venv/lib/python3.12/site-packages/tensorfold
grep -c "_think_budget" $V/cuda/server.py
grep -c ": ping" $V/cuda/http.py
grep -c "_dump_expert_rows" $V/cuda/exl3/experts.py   # 必须 0

# 补丁脚本全部幂等：重复执行输出 SKIP already applied
python3 ~/patch_ttsilence.py && python3 ~/patch_tt_reserve.py

# 回滚 0.6.1 → 0.3.6.3（完整旧 venv，含 Mia 8 补丁 + prefix-cache）
pkill -f '[t]ensorfold serve'   # 注意 bracket，防 pkill 自杀
mv ~/tensorfold-venv ~/tensorfold-venv-0.6.1.bak
mv ~/tensorfold-venv-0.3.6.3.bak ~/tensorfold-venv
nohup ~/tf-serve.sh > ~/tensorfold-serve.log 2>&1 &
```

**pkill 自杀陷阱**（已踩 6+ 次）：ssh 单行命令任何位置不得出现未加 bracket 的目标字面量（包括 grep 参数）。杀进程只写 `pkill -f '[t]ensorfold serve'`。

**速度测试**：统一用 `~/exl3bench.py`（skill 目录有 v1.1 副本），别用临时脚本——历史上一版 bench 把"流式首个 chunk"当 ttft，而 TensorFold 会先发 role-only 占位 chunk，报出过 1.08M tok/s 的假数字。

## 6. 文件清单

| 路径 | 内容 |
|---|---|
| `~/patch_{moewindow,fill_slice,ntile,pf2,pf2_order,expert_rows,expert_rows_fix}.py` | 内核实验补丁（幂等） |
| `~/patch_{ttsilence,tt_reserve}.py` | 生产服务补丁（幂等，已应用） |
| `~/tensorfold-patches/pre-*/` | 每补丁应用前的逐文件备份 |
| `~/tensorfold-patches/draft_vocab.txt.{orig,blunt111k.bak,mia-zh-101679.bak,upstream-0.6.1}` | 词表四代 |
| `~/draft_vocab.zh-aug.txt` | 现役中文增强表源文件 |
| `~/expert_rows.pt` + `~/analyze_expert_rows.py` | expert 行数实测数据 + 分析 |
| `~/exl3_window_cost.py` / `~/exl3_prefill_math.py` / `~/exl3_layer_timing.py` | 瓶颈定量 microbench |
| `~/zh_vocab_audit.py` / `~/zh_vocab_build.py` | 语料反查词表工具链 |
| `~/tf-serve.sh` / `~/switch_to_mlx.sh` | 启动脚本（EXL3 / MLX 两配方） |

---

## 7. 补充记录（第二轮补录，2026-10-07）

### 7.1 已关闭的早期实验（勿重做）

| 实验 | 结论 |
|---|---|
| KV bf16 vs int8 | 短上下文打平 → **回滚 int8**（现役 `--kv-dtype int8` 的由来） |
| KV 扩容（0.3.6.3） | parallel 4→6 + `_remember` KEEP 8→16，+10.6 GiB，速度零损失；0.6.1 时代 parallel 又从 6 降到 4（4 路足够且每路能满窗） |

### 7.2 未定论档：TF_FLASH_DRAFT_VOCAB=0

草稿打全 248k 词表：接受率上限更高、每步打分更贵。首次尝试因启动方式不规范把服务带崩，**未取到数据**。要测：detached 脚本 + 独立监控，千万别和 pkill 写在同一条 ssh 里。

### 7.3 系统层（含一处 skill 与现实不符）

- ⚠️ **sysctl 纠偏**：skill 称 `vm.compaction_proactiveness=0` 已持久化——实测 `/etc/sysctl.d/99-exl3-serve.conf` 是 **0 字节空文件**，运行时值=20（默认）。该优化当前**未生效**。补写：`echo 'vm.compaction_proactiveness = 0' | sudo tee /etc/sysctl.d/99-exl3-serve.conf && sudo sysctl --system`
- 锁频教训：`nvidia-smi -lgc` 在 GB10 需 root，tf-serve.sh 里的锁频**从未生效**（GPU 自然频率 ~2294 MHz，apps clock 2418）。TF/TabbyAPI 两方案同频率，频率不是差异项。

### 7.4 视觉外挂塔（0.6.1 工作区，tf-serve.sh 现状）

- EXL3 包自带视觉塔是 EXL3 量化（`k_proj.mul1/suh/svh/trellis`），0.6.1 的 `vision/qwen_cuda.py` 只收**未量化 BF16 + 融合 qkv** → 必须外挂：`TENSORFOLD_VISION_WEIGHTS=~/models/qwen38-vision-tower/vision-tower-bf16.safetensors`（0.84 GiB，ModelScope 官方包提取，tokenizer 从 EXL3 包拷入）
- **env 改名陷阱**：0.3.6.3 是 `TENSORFOLD_VISION_DIR`（指目录），0.6.1 是 `TENSORFOLD_VISION_WEIGHTS`（指文件）——沿用旧名**静默失效**，报错 `unsupported vision tensor range`
- tf-serve.sh 另设 `TENSORFOLD_VISION_WORKSPACE_MIB=12288`（12 GiB 视觉工作区上限）

### 7.5 思考档位机制（客户端必读）

- 档位 `chat_template_kwargs: {enable_thinking: true, reasoning_effort: low|medium|xhigh}`，默认 xhigh；**不支持 thinking_budget**（顶层写被静默忽略，2026-10-01 实测：顶层 low/xhigh 跑出完全相同的 522 rounds）
- 思考量实测（~/sweep-front.txt）：45k prompt 下 low/medium/xhigh = 268/176/2251 rounds（xhigh 是 low 的 8.4 倍）；**180k 时 low 掉到 170 rounds 且比 medium 慢**，异常未查

### 7.6 缓存行为探测（2026-10-04，对生产服务实测）

| 请求 | cached | prefill |
|---|---|---|
| 冷 460 tok | 0 | 0.748s（615 T/s） |
| 完全重复 | 455/460（99%） | 0.188s（4× 快） |
| 多轮追加一轮 | 459/482（95%） | 0.217s（只算 23 个新 token） |

机制：`token_sha` 前缀哈希，热会话等效 prefill 2200+ T/s——820 T/s 的墙只砸真正的新 token。真实命中率取决于：① KEEP 槽位 LRU 逐出；② 客户端 context compaction 改写整段前缀（compaction 后第一波 = 全冷，长会话最大一击）；③ 系统提示词注入时间戳/随机 ID 会全段 miss。

### 7.7 0.6.1 已有但未调的旋钮（按需取用）

`--checkpoint-slots`（前缀缓存槽位，现 3/流；多会话 agent 调高）、`--prompt-cache-gib`、`--spill-gib`+`--snapshot-dir`（被逐出前缀落盘复用，省冷 prefill，代价 SSD 写）、`--decode-share`（prefill 期间在跑回复继续解码，改善混合负载时延）、`--thinking-budget`（agent 场景省 token，注意 §4 的预留语义）、`--vision-max-images`（现 4 张/请求，Mia 补丁时代 50）

### 7.8 测试/扫参脚本清单

| 路径 | 用途 |
|---|---|
| `~/exl3bench.py` | **统一速度基准（唯一权威）**，v1.1 修过 ttft 假数字 bug（role-only 占位 chunk 不能当首正文） |
| `~/ab_probe.py` | MTP/KV 扫参（同一套 5 prompt 平均） |
| `~/conc_test.py` | 并发聚合 tok/s（parallel 档位对比） |
| `~/sweep-front.txt` | 思考档位 sweep 原始数据 |
| `~/tensorfold-patches/tf_conv_test.py` / `tf_think_test.py` / `tf_vision_regress.py` | 数值等价 / 思考模式 / 视觉回归 |

---

## 8. 内核级修改全景（pristine diff 核实，2026-10-07）

把当前 venv 与上游 pristine v0.6.1（`/tmp/tf-pristine`，`git clone --depth 1 --branch v0.6.1`）逐文件 diff，共 **20 个文件**与上游不同（不含 `__pycache__`/`.so`）。其中 5 个是 §4 服务补丁 + 词表（已记录），**其余 15 个就是"还有的内核级修改"**，此前没有任何文档。复查命令：

```bash
diff -rq /tmp/tf-pristine/src/tensorfold ~/tensorfold-venv/lib/python3.12/site-packages/tensorfold \
  -x "__pycache__" -x "*.so" -x "*.pyc"    # tf-pristine 不在就先 clone
```

### 8.1 性能类（生产在跑）

| # | 文件 | 修改 | 机制与收益 |
|---|---|---|---|
| P1 | `qwen4_exp/cuda/mtp.py` + `decode.py` + `multi.py` | `mtp_compute/mtp_forward` 增加 `logits=False` | prefill 吸收 MTP 缓存时**跳过草稿词表投影**（那一步的 logits 没人用）。三处调用点（resume absorb、chunk 吸收、multi._absorb）全部传 False |
| P2 | `qwen4_exp/cuda/multi.py` | **首 token 先于 draft 重排** | 原实现 `s.drafts = draft(...)` 排在 `s.take([first])` 之前，首 token 要等 draft 链排完才发；改为先 `drafts=[]` → `take([first])` 发出首 token → 再排下一个 draft。**TTFT 优化**，采样路径改动，回归靠 `tf_conv_test.py` |
| P3 | `qwen4_exp/cuda/forward.py`（新 `read_ahead`）+ `decode.py` + `multi.py::_read_ahead` | **n-gram 表预读** | 当前 prefill pass 计算期间，用本 piece 末尾的 n-gram 历史预读下一 piece 要查的表行（`ple.table.read_ahead`，单流 decode.py 与多流 multi.py 各接一处），把 SSD/内存表延迟与计算重叠 |
| P4 | `qwen4_exp/cuda/multi.py` | image prompt 后 `torch.cuda.empty_cache()` | 视觉塔 scratch 在 prompt pass 前归还（"the tower's scratch back before the prompt's passes"），防大图挤爆后续 prefill 工作区 |
| P5 | `qwen4_exp/cuda/ssd_read.py` + `ssd_read.cpp`（**新文件，自研 C++ 扩展**）+ `host_table.py` | **SSD n-gram 原生读取器** | C++ 线程批量 pread、释放 GIL（`torch.utils.cpp_extension.load` JIT 编译，`TENSORFOLD_SSD_NATIVE=0` 可回退 Python reader）；`host_table.py` 把 `SSDTable` 包一层 `ReadAhead`。**生产 dormant**：当前 EXL3 的表是 locked in memory（启动日志 "locked in memory in 17.7s"），SSD 路径只有表放不下的配置才走 |
| P6 | `cuda/geometry.py`（新 `indexed_prefill_rows()`）+ `qwen4_exp/cuda/engine.py` | `TENSORFOLD_PREFILL_ROWS` env（256–16384） | 0.3.6.3 staging-8192 patch 的 0.6.1 移植版。**EXL3 路径被 `is_exl3()` 锁死为 None**（§1.5 证伪：prefill 恒定 820 T/s 与 chunk 无关），env 口子只留给非 EXL3 包 |
| P7 | `qwen4_exp/cuda/multi.py` | `TENSORFOLD_MTP_COPY=1` + `CopyIndex`（COPY_MATCH=8） | prompt-lookup 拷贝草稿（0.3.6.3 时代遗产）。bench 负载零触发（要求末尾 8 token 精确重复），留着等真实重复场景 |

### 8.2 功能类：视频/多图支持（Mia 血统的 0.6.1 手工 backport）

⚠️ **与 skill 的说法冲突**：skill 称"0.6.1 视频不可用"，但 diff 显示 0.3.6.3 的 Mia 视频能力**已被手工移植进 0.6.1 venv**（`server/prompts.py` 有 `load_videos` 调用链）。是否可用**未实测**——待验证项，别引用旧结论。

| # | 文件 | 修改 |
|---|---|---|
| V1 | `vision/videos.py`（**新文件**） | CPU 视频输入：限字节数、固定采样率抽帧、直解码到视觉塔分辨率；支持 mp4/webm/mov/mkv |
| V2 | `vision/qwen_cuda.py`（134 行差） | `MAX_VIDEO_PATCHES≈65k token`（每次塔调用仍 ≤16384 patch）；`MAX_IMAGES=50` 共享 16384 token 预算（`image_setting` env 旋钮）；`video_token_id` 探测 `self.videos`；media_tokens 含视频 token |
| V3 | `server/messages.py` | `video_url`/`video` part 类型进入 `_VISUAL` 白名单（user 消息校验放行） |
| V4 | `server/prompts.py`（33 行差） | `allow_videos` 贯穿 `split_images`→`load_videos`→`frontend.prepare(..., videos=clips)`；**`MANY_IMAGES` 信号量**（多图请求逐个处理，防 host 内存爆）；frontend 自带 `image_limits` 时尊重其字节/像素预算 |
| V5 | `vision/images.py` / `qwen_processing.py` / `images_http.py` | 多图共享预算、视频源 URL 获取等配套 |
| V6 | `qwen4_exp/cuda/engine.py` | 启动日志打印 "image and video input"（随 `self.vision.videos`） |
| V7 | `vision/videos.py.rej` | **patch 残留物**：new-file patch 应用后留下的 .rej，内容与 videos.py 本体一致，确认无丢失 hunk。可删，留着也无害 |

### 8.3 核实结论

- **生产 EXL3 实际吃到的内核优化**：P1（省草稿投影）+ P2（TTFT 重排）+ P3（n-gram 预读）+ P4（视觉 scratch 回收）。P5/P6/P7 是"留了口子但 EXL3 生产不触发"的档位。
- **没有 patch 脚本的直接改动**（P1–P4、V1–V6）都是就地编辑 venv 完成的，**上游升级 0.6.2+ 时会全部丢失且无脚本可重放**——升级前必须先重新 diff 本文件清单，逐项移植。这是本文件存在的主要原因。
- 未知项：P2 的首 token 重排与 P1 的 logits=False 是否进入上游后续版本，未核对（diff 对象是 v0.6.1 tag）。

---

## 9. 原生引擎（路线 A · exllamav3）补丁

TensorFold（本文件 §1–§8）是路线 B。**路线 A = exllamav3 1.5.1 fork + TabbyAPI，是"原生 EXL3 引擎"**，它自己的优化体系在这里。

### 9.1 补丁清单（仓库 `exllamav3-patches/`，基线已钉死）

| 补丁 | 提交数 | 基线 | 内容 |
|---|---|---|---|
| `feat-hybrid-draft.patch`（189 KB） | 23 | `vcruz305/exllamav3 @ 74b6f5a` | 见 9.2 的 M1→M3 全链 + 融合 GR kernel |
| `feat-spec-sampling.patch`（90 KB） | 14 | 同上，链起点 `5e52ac9` | 精确推测采样验证路径（与 hybrid-draft **互相独立，不要同时应用**） |
| `tabbyapi-local.patch` | 1 | `theroyallab/tabbyAPI @ f07131c` | `local: allow max_history bump under EXL3_HYBRID_NGRAM` |

提交历史在 `*.commits.txt`（作者/日期/信息全保留）。**为什么是补丁不是分支**：Ajie16/exllamav3 fork 的历史里有 5 个上游继承的旧提交含 Mistral key（README.md，当前已删除），GitHub Push Protection 扫全历史拒绝推送。解封 URL 在 `exllamav3-patches/README.md`，授权后可推完整 1800+ 提交历史。

### 9.2 feat-hybrid-draft 的优化内容（M1→M3，2026-09-27/28）

| 阶段 | 内容 | 关键提交 |
|---|---|---|
| **M1 观测** | `spec_transform`/`extract_spec` 原语 + **spec-shadow 只观测模式**（EXL3_SPEC_SHADOW，不改采样，先量出投机接受率的真实分布） | `19de191` `fe20eac` `4b8ab82` |
| **M2 精确验证** | **EXL3_SPEC_SAMPLING 精确推测采样验证路径**：批量乐观验证、window cap、tested-basis shadow、adaptive engagement；含 `docs/spec_sampling.md` 与 CPU 门禁 | `8fd59de` `4807547` `35d04a2` |
| **M3 混合草稿** | **EXL3_HYBRID_NGRAM：n-gram 草稿 + MTP 草稿混合**——hybrid 宽度钳制到 cache max_history、per-job 自适应退避（EXL3_NGRAM_ADAPTIVE）、默认 EXL3_NGRAM_MAX_DRAFT=7（**前向 kernel 在 q_len>8 有悬崖**，ab5671a 修的默认值坑） | `54b0072` `c902ead` `d11d060` `ab5671a` |
| **kernel** | **EXL3_GR_TUNED 协作式融合 GR mix kernel（R ≤ 4）** | `c5de62d` |
| 剖析 | M3 实测文档（w=7 + adaptive vs MTP-only 基线）、M4 decode 下限 profiling | `6d3a91a` `deab4d3` |

配套：Mia 风格的 `exllamav3-patches/` 之外还有 `~/qwen38-exl3/` 运行布局（见 9.3）。

### 9.3 本机布局与易错点（★实际跑的是 worktree）

```
~/qwen38-exl3/
├── exllamav3/          # 主检出 master @ 74b6f5a（venv .pth 指向这里——但不是服务用的！）
├── exllamav3-spec/     # ★ 同仓库 git worktree，检出 feat/hybrid-draft ← serve.sh 实际加载的引擎
├── tabbyAPI/           # detached @ b8c0497（本地提交在 f07131c 之上）
├── state/config.yml    # serve.sh 渲染产物
└── venv/               # exllamav3 可编辑安装
```

**易错点**：`serve.sh` 运行的引擎是 `exllamav3-spec/`（worktree），不是 `.pth` 指向的 `exllamav3/`。判断服务加载了哪个，看日志开头的：
`exllamav3 1.5.1.post1 (fork) at /home/xujie/qwen38-exl3/exllamav3-spec/exllamav3`

### 9.4 应用与回滚

```bash
# exllamav3（在 spark-2）
cd ~/qwen38-exl3/exllamav3
git stash && git checkout 74b6f5a && git checkout -b feat/hybrid-draft-v2
git apply ~/workspace/qwen3.8-flash-spark-1x/exllamav3-patches/feat-hybrid-draft.patch
# tabbyAPI 同理（f07131c + tabbyapi-local.patch），两个 exllamav3 补丁互斥
```

路线 A/B 互斥（EXL3 单机独占 ~88+ GiB vs TensorFold 互斥，切之前先停另一个）。当前生产是路线 B（TensorFold）。

### 9.5 路线 A 实测基线（本机数字）

prefill @32k **~1,130 T/s**（@75k 未测）；decode 中文 42–49 tok/s；KV 池 1,572,864 token（共享池，`cache_size`）；n-gram 表 `ngram_ram: true` 全放内存；视觉 TabbyAPI `vision: true`。
对比路线 B（TensorFold+MLX）：prefill 2,425 T/s 但只能配审查权重要么自己转（g32 约束）。
**选 A 的理由是抗审查权重生态 + exllamav3 原生实现，不是速度。**
