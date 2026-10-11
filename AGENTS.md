# AGENTS.md — 给 AI agent 的操作说明

**动手前先读完这一页。**

**⚠️ 最重要的一件事：这个仓是 EXL3 配方，但线上跑的不是它。** 2026-10-11 起 `:8899` 上服务的是 **TensorFold 0.6.6 + 抗审查 MLX 4-bit**，配置不在本仓（在 spark-2 的 `~/tf-serve-uncensored.sh`）。本仓的 EXL3 栈处于**待命**状态，随时可切回。

---

## 一、当前在跑什么（2026-10-11 核实）

**TensorFold 0.6.6 + 抗审查 MLX 4-bit，单机 spark-2（= spark-ac8f，`10.100.65.1`）**

| | |
|---|---|
| 权重 | `~/models/yikoudazhong-TF-4bit-MTP`（105.43 GiB，4-bit affine **group 32**、uniform，含 vision tower） |
| 引擎 | TensorFold **python-0.6 分支 HEAD `ed78d6f`（自报 0.6.6）**，venv `~/tensorfold-venv` |
| 启动 | spark-2 **`~/tf-serve-uncensored.sh`** ← **不在本仓** |
| model id | `Qwen3.8-Flash-Local` |
| 监听 | `10.100.65.1:8899`（CX-7 only，脚本里写死 `--host`） |
| 对外入口 | **AxonHub**（kimi/dsh 都走它） |

**为什么换掉了 EXL3**：prefill **1,130 → 2,430 tok/s（2.15×）**，decode 基本持平。根因是量化格式——EXL3 的 trellis 反量化是逐 token 查表解码、吃不进 tensor core；MLX affine 的反量化是乘加、能融合。

**换回 EXL3 只需**：`kill $(pgrep -f "tensorfold serve")` → `bash exllamav3-tabby/serve-local.sh`。EXL3 包和引擎都还在盘上。

---

## 二、本仓是什么

**EXL3 + TabbyAPI 的配方仓**（`exllamav3-tabby/`），外加 `vllm-plugin/`（NVFP4，另一条路）和 `tensorfold-mlx/`（已退役的旧补丁集）。

**下面是 EXL3 栈的完整运行说明。它在线上待命，但照着启动仍然有效**——切回去那天用得着。

### 运行时身份：先确认，别假设

`setup.sh --check` 只验 venv 里的 `.pth` 目标（`~/qwen38-exl3/exllamav3`，master），**而生产实际加载的是 worktree**——靠 `serve-local.sh` 的 `EXL3_SPEC_SRC` 把它前置到 `PYTHONPATH`。

```bash
P=$(pgrep -f 'tabbyAPI/main[.]py' | head -1)
readlink /proc/$P/cwd
tr '\0' '\n' < /proc/$P/environ | grep -E '^EXL3_|^HOST=|^CHUNK_SIZE='
```

**worktree 的 `.so` 与 master 的不同**（worktree 含 `EXL3_GR_TUNED`）。判断某开关是否真的存在，要 `grep -a` **实际加载的那个 `.so`**，不是源码目录。

### ★ 必知补丁：requeue 计数（**丢了就少报 3 倍速率**）

`exllamav3/generator/job.py` 的 `prepare_for_requeue()`：

```python
"rq_new_tokens": self.rq_new_tokens + self.new_tokens,   # ← 必须带累加
```

上游 2026-09-06 写漏了累加（只留 `self.new_tokens`）。**症状**：`output_chunking`（默认开）下，输出超过 `max_rq_tokens` 的请求**上报 decode 速率最多低 3 倍，完全静默**。实测同负载 4,897 → 15,000 上报 token，17.2 → 51.4 T/s，**耗时不变**——引擎没慢，是计数器少算。

**判据（不需要埋点）**：每个请求必须满足 `accepted_draft_tokens <= completion_tokens`。违反即中招。

补丁 `exllamav3-patches/0005-rq-token-accounting.patch`，分析 `docs/decode-drop-root-cause.md`。

### 配置与实测基线

配置从 `tabby-config.yml` 渲染到 `~/qwen38-exl3/state/config.yml`，**别手改渲染产物**：

```
profile concurrent   max_batch_size 4    max_seq_len 262144   cache_size 1048576
cache_mode "8,8"     chunk_size 8192（CHUNK_SIZE）            ngram_ram true
vision true          draft_num_tokens 5  dynamic_draft true
```

引擎环境变量（`env.sh`，都有实测依据）：`EXL3_MTP_HEAD_N=163840`、`EXL3_GR_TUNED=1`、`EXL3_MOE_COOP_WIDE=1`、`EXL3_GR_INT8=1`、`EXL3_INT8_GEMV=0`、`EXL3_DRAFT_CONFIDENCE=0.6`、`CHUNK_SIZE=8192`。

**实测（spark-2，2026-10-08）**：prefill @40k 冷 **1,136 T/s**；decode 四类均值 **52.0**（thinking off）/ **46.6**（on）；接受率 ~61%；启动 ~90 s。

**`CHUNK_SIZE` 是最有效的 prefill 杠杆**（不需重编译）：4096 → 1,004 T/s，**8192 → 1,145（+14%）**，decode 无变化，内存 +2 GiB。**16384 被 autosplit 拒绝**。

### 已测到顶的旋钮（别再重复扫）

`EXL3_MOE_TILE_N=128` −4.7%、`EXL3_HGEMM_F16ACC=1` −7.3%、`EXL3_QT_OPTIMIZED=1` −4.9%，全部更差。`NGRAM_RAM=false` 省 18 GiB 但 **decode −33%**（那张表是 PLE 逐层嵌入表，每次前向都用，decode 也算），不值。

**一条更正**：`EXL3_MOE_FUSED_ROWS` / `_WIDE` / `MIXEDK_MIN_ROWS` / `QC_PF_TWO_PASS_MIN_Q` 曾被记成"不是运行时开关"，**理由是错的**——判据是在 C++ 扩展里搜 `getenv`，而它们在 **Python 层**用 `os.environ.get` 读（`block_sparse_mlp.py:28,41,52`、`triton_paged.py:703`）且真的参与 dispatch。**这 6 个从未扫过**，见 `docs/exllamav3-optimization-backlog.md`。

---

## 三、访问模型

**两套栈共用同一条访问规则：只有 CX-7 链上的设备能直连 `10.100.65.1:8899`，其余一律走 AxonHub。**

```
CX-7 链上设备（只有对端 Spark）  ──直连──→  10.100.65.1:8899
其他一切（家庭网/Tailscale）    ──→  AxonHub（发 key）──→  10.100.65.1:8899
```

| 入口 | 结果 |
|---|---|
| `10.100.65.1` CX-7 | ✅ 200 |
| `10.100.64.1` 管理网 / `192.168.1.6` 家庭网 / `100.111.150.39` Tailscale / `127.0.0.1` | ❌ 全部不通 |

**隔离是物理的**：`10.100.65.0/24` 只存在于 `enP2p1s0f0np0` 上（`scope link`，邻居表只有对端一台），CX-7 是两台 Spark 之间的**直连电缆**。不需要防火墙规则。

**⚠️ 最大的一条陷阱**：TensorFold 的模板脚本（Mia 的 `tf-serve-mlx.sh`）用 `--host 0.0.0.0`，**直接套用会把整套隔离重新打开**。`~/tf-serve-uncensored.sh` 里是 `--host 10.100.65.1`，注释写明「不要改回 0.0.0.0」。

**AxonHub**（spark-0d97 的 `hermes` 容器，DB `~/.hermes/axonhub/axonhub.db`）：

- channel id=5 `qwen-local` → `http://10.100.65.1:8899`
- **改 model id 要同时改 `models` 表和 `settings.associations[].channelModel.modelId`**，只改 `supported_models` 会 422
- **改完不用 reload**，它每次请求读 DB

---

## 四、不要做

- **❗ 不要把 TabbyAPI 更新到上游 `main`。** 上游要求 `exllamav3 >= 1.5.4`（硬 `RuntimeError`），而 vcruz305 的 fork 版本号停在 `1.5.1.post1`——**即使同步那 21 个提交也不满足**。生产钉在 `local/port-bind-fix`。
- **不要 `pip install exllamav3`**（PyPI 或 TabbyAPI wheel）。运行时要 `setup.sh` 从源码编的 fork。
- **不要装 TabbyAPI 的 `cu12`/`cu13` extras**，也别跑它的 `start.sh`。
- **不要用 `vllm-plugin/`**（除非用户明确要 vLLM）——它的 `prepare_pack.sh` **就地改写 pack**，改完 exllamav3 加载不了。
- **不要用 `exllamav3-tabby/beta/`** 或 **`legacy/`**。
- **不要把 `MAX_SEQ_LEN` 抬过 262144**——那是训练窗口。
- **不要 `pkill -f`**（见下）。
- **不要用 `mv` 轮转日志**（见下）。
- **TensorFold 侧**：不要给这个 MLX 包设 `TENSORFOLD_VISION_WEIGHTS`（vision tower 打包在里面，EXL3 包才需要外部 tower）；不要用 `NGRAM_RAM` 那套思路去设 `TENSORFOLD_*`——参数名不同。

---

## 五、排查

### 通用（两套栈都适用）

| 现象 | 真相 |
|---|---|
| **`pkill -f 'xxx'` 会让 ssh 会话自己死掉** | 模式匹配到了远端 shell 自己的 cmdline。**按 PID 杀**。`pgrep -f` 同样会匹配到你自己那条 ssh 命令——判存在性要用 `[ -d /proc/PID ]`。 |
| **`mv` 日志文件后，测试读到的还是旧数据** | **`mv` 不重定向已运行的进程**——它继续写被移走的 inode。 |
| **速度数字忽高忽低** | **重启后首轮慢约 20%**（同一确定性用例 61.5 ms/round vs 稳态 49.0）。基准测试必须**丢弃预热轮**，再对单一固定用例重复 5–8 次取中位。曾因忽略这一点得出过完全相反的版本结论。 |

### EXL3 专属

| 现象 | 真相 |
|---|---|
| **启动报 `Insufficient VRAM in split for model and cache`，但 `free` 还有几十 G** | autosplit 的**余量预检**主动抛 OOM、异常被 `except` 吞掉。真正的 OOM 数字没打出来。相关：`CHUNK_SIZE` 越界。 |
| **`Port 8899 is currently in use. Switching to 8900.`** | **已修**（cherry-pick 上游 2fd6cc7）。现在端口被占会**明确报错退出**。 |
| **速率掉到 1/3** | requeue 计数补丁（见上）。 |
| **`strings` 源码目录找不到某开关** | 要看**实际加载的 `.so`**；backlog 里 `EXL3_MOE_FUSED_ROWS` 那批是 **Python 层**的，在 `.so` 里当然找不到——**当初据此误判成 no-op**。 |

**重启纪律**：

```bash
P=$(pgrep -f 'tabbyAPI/main[.]py' | head -1); kill $P
for i in $(seq 1 40); do pgrep -f 'tabbyAPI/main[.]py' >/dev/null || break; sleep 1; done
ss -ltn | grep -q ':8899' && echo '!! 端口仍被占用，别启动'
bash exllamav3-tabby/serve-local.sh
```

### TensorFold 专属（升级时必须做，否则静默劣化）

**① 中文草稿词表会被 `pip install` 静默覆盖。** 上游自带 `draft_vocab.txt` 是 **79,591 行纯英文/代码**，我们的是 **102,089 行**（含中文扩展）。升级后必须：

```bash
T=~/tensorfold-venv/lib/python3.12/site-packages/tensorfold/families/qwen4_exp/cuda/draft_vocab.txt
wc -l $T                                  # 必须是 102089
cp ~/draft_vocab.engine-0.6.1.txt $T
```

**② `TENSORFOLD_MTP_COPY` 已改名**为 `TENSORFOLD_COPY_DRAFTS`（且默认 ON，`=0/off/false/no` 才关）。旧名现在是**静默无操作**。

**③ 装 0.6.6 要装分支 HEAD（`ed78d6f`）而不是 `v0.6.6` 标签**——标签落后 12 个提交，缺 copy-drafts 等。**spark-2 连不上 github.com**，源码要从 spark-0d97（有代理 `127.0.0.1:7890`）下 tarball 再 scp。回滚：整份 venv 备份在 `~/tensorfold-venv-0.6.1.bak`。

**④ 换版本后首次启动会重建 CUDA 扩展**（`~/.cache/torch_extensions/py312_cu130/`），加载从 ~70s 涨到 ~100s，且**首轮请求更慢**。

**⑤ 采样已固定为 checkpoint 的推荐值（`tensorfold-mlx/patches/kernel/server_pin_sampling.py.patch`）。**

模型的 `generation_config.json` 推荐 `temperature 1.0 / top_k 20 / top_p 0.95`，而 TensorFold 的
`App.sampling_for` 原本是客户端优先：

```python
temp = float(fields.get("temperature", self.sampling["temperature"]))
```

于是**同一个会话在不同客户端下答案不同**（一个发 greedy、一个发 0.7，比的就不是同一个模型）。
补丁让 checkpoint 的值胜出，**`seed` 仍归客户端**——它管可复现性，不管分布。

- **生效时机**：改的是 venv 里的文件，**必须重启**才生效（Python 已把模块载入内存）
- **升级会冲掉它**：`pip install` 覆盖 server.py 后要重跑 `patches/apply_pin_sampling.py`（幂等）
- **退回开关**：`TENSORFOLD_PIN_SAMPLING=0`（**服务端环境变量，客户端够不着**，需要 greedy 的基准测试用得上）

**注意种子是按 prompt 推导的**（`seed_for(prompt)`），所以**即使 temperature 1.0，同一 prompt 重发也逐字相同**。
这对 agent 是好事（行为稳定、可缓存），代价是没有多样性——想测方差必须换 prompt。

---

## 六、两条路线

本仓保持**两条清晰隔离的路线**：

- **路线 A：`exllamav3-tabby/` + `vllm-plugin/`**（原生 EXL3）
- **路线 B：`tensorfold-mlx/`**（TensorFold/MLX 的历史补丁集）

**路线 B 为什么被换掉（以及后来为什么又回来了）**：它当年跑的是**自制转换的 MLX 包**（从 `huihui-ai/...-abliterated` 转的），**abliteration 损伤了质量**。2026-10-11 找到并换上了 **`yikoudazhong/Qwen3.8-Flash-Next-Uncensored-TF-4bit-MTP`**——**格式完全相同**（4-bit affine g32、uniform、105.43 GiB），但：

| | 自制包 | 新包 |
|---|---|---|
| 抗审查来源 | huihui-ai | **orcarouter**（与我们满意的 EXL3 包**同源**） |
| 源精度 | BF16 | FP8 |
| 校验 | dtype/shape 表（3747 项） | **对独立 MLX 构建逐值对拍**（861 张量模式）+ **单独核对 hyper-connection 离群值**（abs-max ≈ 50–70× RMS） |

**巧合对应**：我们转换日志里误差最大的**恰好就是 hyper-connection 权重**（`input_mix_weight_*` 相对误差 0.0875/0.0936）——正是新包作者点名的那一类。**所以问题在 huihui 的 abliteration 或我们的转换校验，不在源精度**（我们 BF16 比人家 FP8 更干净）。

**盘上的包**：

| 包 | 来源 | 大小 |
|---|---|---|
| `yikoudazhong-TF-4bit-MTP`（**在用**） | orcarouter Uncensored-FP8，第三方转换 | 105.43 GiB |
| `Qwen3.8-Flash-Next-Uncensored-EXL3-3bpw`（待命） | orcarouter Uncensored → Lygodactylus EXL3 | 68 G |
| `Huihui-...-MLX-4bit-g32`（弃用） | huihui abliterated，**自制转换** | 106 G |
| `Vontra-...-MLX-4bit-MTP` | 官方，**审查版** | 106 G |
| `Qwen3.8-Flash-Next-EXL3` | 官方，**审查版** | 84 G |

**要跑质量对照，用官方包做基线**——abliterated 和官方不是同一个模型，分数不可互推。

---

## 七、两机（EXL3 **不能**跨机跑这个模型）

**跨机 TP/EP 不存在**（`model_tp.py` 是 `multiprocessing`，单机多卡）。**跨机 PP 存在但只认 `Glm5NextForConditionalGeneration`**，`Qwen4ExpForConditionalGeneration` 启动即 `ValueError`。而且 PP 是串行的（无微批重叠），**单流不加速**。

**TensorFold 侧同理**：这条线是单机配方。要 2 机加速走 vLLM TP2 那几套（`:8888`），代价是停掉当前栈（内存互斥）。
