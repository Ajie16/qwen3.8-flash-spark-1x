# 路线 B · TensorFold + 抗审查 MLX 4-bit（**当前生产**）

TensorFold **0.6.6**（python-0.6 分支 HEAD `ed78d6f`）跑 MLX 4-bit affine（group 32）权重。
**prefill 比 EXL3 路线快约 2.1 倍**，而且**抗审查权重已经有了**。

当前权重：**`~/models/yikoudazhong-TF-4bit-MTP`**（105.43 GiB，22 分片，4-bit affine g32、uniform，
`base_model` = `orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8`）。启动器
[`launchers/serve-uncensored.sh`](launchers/serve-uncensored.sh)。

---

## 为什么快，为什么曾经受限

**prefill 实测 2,323–2,476 tok/s（跨 1.8k–56k token 基本持平）**；换包前 EXL3 约 1,130 tok/s @32k。
差距**全部来自量化格式**：MLX 4-bit 路径有专门的 prefill 专家内核（`cuda/experts_prefill.cu`，
64 行 tile + `cp.async` 流水），EXL3 路径没有，只能用 16 行的 decode 内核硬顶。
**这个结论最早是在 TensorFold 自己的 EXL3 路径上证伪的**——staging patch 把 chunk 从 2.6k 拉到
42k，prefill 恒定 ~820 T/s，与分块/调度无关；作者当年正是因此封掉了 `TENSORFOLD_PREFILL_ROWS`
的 EXL3 口子。

**格式限制**：`families/qwen4_exp/cuda/weights.py:384` 只接受

```
MLX 4-bit (groups of 32)  或  NVFP4 (experts-only)
```

而 MLX 社区默认 `group_size = 64`。

> **更正（2026-10-11）**：本文件此前写"搜遍 44 个变体，`group_size: 32` 的只有两个、都是官方审查版，
> 要抗审查只能自己转"。**这个结论错了。** `yikoudazhong/Qwen3.8-Flash-Next-Uncensored-TF-4bit-MTP`
> 就是 **g32、4-bit、uniform 的抗审查包**，而且**质量比自制的好**。当初的扫描漏了它——它在
> `search=Qwen3.8-Flash-Next` 的结果里排到第 60 名开外，没有 mlx-community 那样的曝光度。
> **教训：扫 HF 要翻到底并逐个查 `quantization.group_size`，不能只看前几页的知名度。**

**自制包为什么被弃用**：它从 `huihui-ai/...-abliterated` 转来，**质量不行**。与现用的第三方包
**格式完全相同**（g32、4-bit、uniform、体积一致），但来源和校验强度不同：

| | 自制包 | 现用包 |
|---|---|---|
| 抗审查来源 | huihui-ai | **orcarouter**（与满意的 EXL3 包**同源**） |
| 源精度 | BF16 | FP8 |
| 校验 | dtype/shape 表（3747 项） | **对独立 MLX 构建逐值对拍**（861 张量模式）+ **单独处理 hyper-connection 离群值**（abs-max ≈ 50–70× RMS） |

**巧合对应**：我们转换日志里误差最大的**恰好就是 hyper-connection 权重**
（`input_mix_weight_*` 相对误差 0.0875/0.0936）——正是新包作者点名的那一类。
**所以问题在 huihui 的 abliteration 或我们的转换校验，不在源精度**（我们 BF16 比人家 FP8 更干净）。

---

## 实测数据

### 冷 prefill（`exl3bench.py`，cached=0）

| 上下文 | 值 |
|---|---|
| 32,568 tok | **2,425 tok/s**（13.4 s） |
| 74,865 tok | **2,105 tok/s**（35.6 s） |
| 缓存命中重发 | prefill 0.02 s（32,567/32,568 命中） |

### decode

| 测法 | 值 |
|---|---|
| 自研集：代码+思考 | 46–48 tok/s |
| 自研集：纯散文 | 39–42 tok/s |
| 长上下文 @74k | 46–48 tok/s |
| MTP 接受率 | 53–59% |

### 用 MiaAI-Lab 的 `tools/bench.py`（同一脚本，可直接对照）

| 项 | 值 |
|---|---|
| code greedy | **63.0 tok/s**（两个 seed 结果一致） |
| chat sampled | 50.5 tok/s |
| 并发聚合 C=1 / 2 / 4 / 5 | 47.0 / 69.9 / 89.7 / **95.4** tok/s |

### 中文草稿词表的收益（A/B 实测）

| 中文 decode | 上游 79,591 id | **中文 102,089 id** |
|---|---|---|
| 技术说明 | 47.1 | **58.8**（+24.8%） |
| 代码注释 | 39.5 | **50.0**（+26.6%） |
| 长文总结 | 32.3 | **45.1**（+39.6%） |
| MTP 接受率 | 32.2% | **56.8%** |

**英文/通用负载上无损失**（47.2/70.8/89.7/97.0 对 47.0/69.9/89.7/95.4，噪声内）。

### needle 检索

3/3（四个长度档位均通过）

---

## 部署

### 1. 引擎

```bash
python3 -m venv ~/tensorfold-venv
~/tensorfold-venv/bin/pip install --upgrade --no-deps \
  "git+https://github.com/ashhart/TensorFold.git@v0.6.1"
```

### 2. 引擎补丁（可选，建议打）

```bash
~/tensorfold-venv/bin/python patches/engine/patch_ttsilence.py   # 4 处：答案预留 / SSE 心跳 / 调度兜底
~/tensorfold-venv/bin/python patches/engine/patch_tt_reserve.py  # 答案预留细化
```

**这 4 处解决什么问题**（都是 `content: null` 及其邻居）：

| 位置 | 问题 | 修法 |
|---|---|---|
| `cuda/server.py::_think_budget` | 思考预算 ≥ `max_tokens` 时，强插的 `</think>` 被截掉，整条回复留在思考块里，客户端收到 `content: null` + `finish_reason: length` | 为答案预留 `max(256, max_tokens//4)` |
| `engine/call_gate.py::take` | 工具调用修复可能只应用一半 | 边界检查 |
| `cuda/http.py` | 长静默期连接被中间设备断开 | 每 ≥10 s 写 `: ping\n\n` |
| `cuda/scheduler.py::_loop` | 单次异常终止整个调度循环 | 包 try/except + 继续 |

### 3. 上游补丁（可选）

```bash
cd ~/tensorfold-venv/lib/python3.12/site-packages
patch -p1 < .../patches/upstream/0002-flash-next-v061.patch
```

来自 `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold`，带来：
视频输入、多图共享 token、`TENSORFOLD_PREFILL_ROWS` 覆盖、SSD 预读、
copy drafts（`TENSORFOLD_MTP_COPY`）、首个 token 先于下一轮草稿。

> ⚠️ 该补丁**新增** `TENSORFOLD_MTP_COPY`（`multi.py:58`，默认 0）。
> 在打补丁**之前** grep 上游是搜不到的——别因此以为它是死变量。

### 4. 中文草稿词表（可选，建议）

```bash
cp draft-vocab/draft_vocab.zh-aug.102089.txt \
   ~/tensorfold-venv/lib/python3.12/site-packages/tensorfold/families/qwen4_exp/cuda/draft_vocab.txt
```

无需任何启动参数——`weights.py:420` 经 `draft_token_ids("default")` 直接读这个文件。
回滚用 `draft-vocab/draft_vocab.upstream-0.6.1.79591.txt`。

### 5. 启动

```bash
bash launchers/tf-serve-mlx-int8.sh    # int8 KV, --parallel 5（推荐）
bash launchers/tf-serve-mlx-bf16.sh    # bf16 KV, --parallel 4
```

**必须先改模型路径**（脚本里是绝对路径）。

---

## 配置要点（每条都是实测得出的）

| 参数 | 值 | 为什么 |
|---|---|---|
| **`--ple-on-ssd`** | 开 | **最大单项，+20% decode**。29.8 GiB n-gram 表装不下时会在**每个解码步**分页（`forward.py:268` 不区分 prefill/decode）。不开实测 decode 40.4/34.7，开了 49.0/41.7 |
| `--parallel` | 5 | 设 4 时第 5 个请求排队，C=5 聚合反降（84.6 vs 95.4） |
| `--kv-dtype` | `int8` | bf16 实测七项指标全略差（差距在噪声内），但单流满窗 7.52 GiB vs 4.47 GiB，**多吃 68% 显存** |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | 4 | 默认是 `max(4, 总内存/10)` = 12.17 GiB，会拒绝 5 条流 |
| `TENSORFOLD_VISION_WORKSPACE_MIB` | 0 | 视觉临时缓存按需从系统 reserve 取、用完归还。设 12288 会白占 12 GiB |
| `--thinking-budget` | **不设** | 上游默认 0 = 无上限。设了会给思考长度加人为天花板（实测所有 xhigh 请求精确停在 4097） |
| `--max-tokens` | 32768 | 与 `--context` 共享窗口：`prompt + max_tokens ≤ 262144`，所以**实际 prompt 上限是 229,376** |

---

## 已知坑（都踩过）

**1. `--ple-on-ssd` 是 MLX 专属，EXL3 包用它会拒绝启动**

```
--ple-on-ssd reads the MLX checkpoint's n-gram tables; an EXL3 pack maps its own
table from its file, so drop --ple-on-ssd
```

**2. `nvidia-smi -lgc` 不起作用**——需 root，`|| true` 静默失败，`-rgc` 会报"没有权限"。
别以为锁了频。

**3. `TENSORFOLD_MTP_COPY` 已不存在——上游改名为 `TENSORFOLD_COPY_DRAFTS`。**
0.6.1 时代这是 Mia 补丁加的名字，python-0.6 分支把它上游化了（`copy_drafts.py`），
并把默认翻转成 **ON**（`=0/off/false/no` 才关）。旧名现在是**静默无操作**。
实测开/关无差异（74.6 vs 74.5 tok/s 中位），保持默认即可。

**4. 思考长度会被自己的答案预留补丁钳住**——`max_tokens=3000` 时
`budget = min(4096, 3000 - max(256, 750)) = 2250`，于是所有 xhigh 读出 2251。
**测试时用生产值 `max_tokens=32768`**，否则测的是补丁不是模型。

**5. 采样种子按 prompt 哈希，同一请求重跑结果逐字节相同。**
想测方差必须**换 prompt**，重复同一请求只会得到 N 个相同数字。要比较两个配置，
用**同一条 prompt 配对**——种子相同，差异只能归因于配置。

**6. 跨会话的 prefill 绝对值会漂**（实测见过同配置差 16%）。任何 A/B 必须在**同一会话内交替**测。

**7. 基准测试必须丢弃重启后的首轮。**
首发是冷的：同一确定性用例 **61.5 ms/round vs 稳态 49.0**，慢 20%。每次跑整套 bench 时这个惩罚
落在那次排在首位的用例上，**会得出完全相反的版本结论**——曾据此错判"0.6.6 比 0.6.1 慢 17%"，
而热态 8 轮实测是 **0.6.6 = 74.6 vs 0.6.1 = 70.7，快 5.5%**。
正确做法：预热丢弃 → 对**单一确定性用例重复 5–8 次取中位**，热态抖动 <1%。

**8. 升级会静默覆盖中文草稿词表。**
上游自带 `draft_vocab.txt` 是 **79,591 行纯英文/代码**，我们的是 **102,089 行**（含中文扩展）。
`pip install` 之后必须换回来，否则中文草稿崩掉且**不报错**：

```bash
T=~/tensorfold-venv/lib/python3.12/site-packages/tensorfold/families/qwen4_exp/cuda/draft_vocab.txt
wc -l $T                                 # 必须是 102089
cp ~/draft_vocab.engine-0.6.1.txt $T
```

**9. 装 0.6.6 要装分支 HEAD 而不是 `v0.6.6` 标签。**
标签（10-06）落后 12 个提交，**缺 copy-drafts**——装标签等于把 Mia 补丁给我们的功能丢了。
另外 **spark-2 连不上 github.com**，源码要从 spark-0d97（有代理 `127.0.0.1:7890`）下 tarball 再 scp。
换版本后首次启动会重建 CUDA 扩展（`~/.cache/torch_extensions/py312_cu130/`），加载从 ~70s 涨到 ~100s。

**10. 采样默认听客户端，已打补丁固定（`patches/kernel/server_pin_sampling.py.patch`）。**
模型自己的 `generation_config.json` 推荐 `temperature 1.0 / top_k 20 / top_p 0.95`，
而 `App.sampling_for` 原来是 `fields.get("temperature", self.sampling[...])`——**客户端优先**，
于是同一个会话在不同客户端下答案不同。补丁让 checkpoint 的值胜出，`seed` 仍归客户端
（它管可复现性，不管分布）。**`TENSORFOLD_PIN_SAMPLING=0` 可退回**，这是服务端环境变量，
客户端够不着——需要 greedy 的基准测试用得上。

**11. `reasoning_effort` 只是一句系统提示词，不是 token 预算。**
模板里 `xhigh` 是**默认值**（`reasoning_effort|default(xhigh)`），所以显式指定它等于不指定；
而 `medium` **在模板里没有分支**，结果是不加任何提示词。实测同一道题的思考长度：
不发 211 字 / `low` 112 / `medium` 107 / `xhigh` 211。**模板只有 low / medium(空操作) / xhigh 三档，
没有比 xhigh 更强的。**

**12. KV 缓存只支持前缀复用**，因为位置 i 的 K/V 依赖 0..i 全部 token。
改第 0 个 token 会让整条前缀失效。指令在头部时切换档位 = 0% 命中；
放到尾部 = 99.6%。（`launchers/tf-serve-mlx-tail-experiment.sh` 是该实验的产物，
33 格配对实测显示尾部不削弱指令遵循，`low` 档还更简短。）

---

## 目录

```
tensorfold-mlx/
├── launchers/
│   ├── serve-uncensored.sh                # ★ 当前生产：抗审查包 + 0.6.6
│   ├── tf-serve-mlx-int8.sh               # int8 KV, parallel 5（官方包时代）
│   ├── tf-serve-mlx-bf16.sh               # bf16 KV, parallel 4
│   ├── tf-serve-exl3-native.sh            # EXL3 包 + 纯净引擎（对比用）
│   └── tf-serve-mlx-tail-experiment.sh    # 尾部模板实验
├── bench/
│   ├── quality.py                         # 11 个可客观判分用例（当前 11/11）
│   └── perf.py                            # sparkDash 四类 + 创作类 + prefill
├── patches/
│   ├── kernel/server_pin_sampling.py.patch # ★ 固定采样（当前已打）
│   ├── apply_pin_sampling.py              # 幂等重放脚本，升级后重跑
│   ├── engine/                            # 4 处 content:null 及邻居
│   ├── upstream/                          # MiaAI-Lab 的 0002-flash-next-v061
│   └── experiments/                       # 已证伪/未采用，留作记录
└── draft-vocab/                           # 中文词表 + 上游对照
```

`patches/experiments/` 里的都**试过且没采用**，留档是因为它们的否定结论有价值：

| 补丁 | 结论 |
|---|---|
| `patch_moe_window.py` | `MOE_WINDOW` 1024→2048 无效果（≈1%，噪声内） |
| `patch_ntile.py` | N tile 放宽不可行：gate_up 的 N=640 使 nt ≤ 8 已是上限，且 nt=16 未实例化 |
| `patch_pf2*.py` | 寄存器预取 pf=2 慢 9.8%（压占用率） |
| `patch_fill_slice.py` | 并发 prefill 分片：修好了饥饿但总时间不变，且拖慢首个结果 |
| `patch_expert_rows*.py` | 测量钩子（每专家行数分布），非优化 |
