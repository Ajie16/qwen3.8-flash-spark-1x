# AGENTS.md — 给 AI agent 的操作说明

**动手前先读完这一页。** 这里记的是**当前真实在跑的部署**，和 `README.md` 的历史描述可能有出入。

---

## 一、当前在跑什么

**路线 A：原生 EXL3 + TabbyAPI，单机 spark-2（= spark-ac8f，`10.100.65.1`）**

| | |
|---|---|
| 权重 | `~/models/Qwen3.8-Flash-Next-Uncensored-EXL3-3bpw`（EXL3 3.05bpw / head 5 / mtp 3，68 GB） |
| 引擎 | exllamav3 fork，**worktree `~/qwen38-exl3/exllamav3-spec`**（分支 `feat/hybrid-draft`） |
| API 层 | TabbyAPI，**分支 `local/port-bind-fix`**（`677dc89`） |
| 启动 | **`bash exllamav3-tabby/serve-local.sh`** |
| model id | **`Qwen3.8-Flash-Local`** |
| 监听 | **`10.100.65.1:8899`（只绑 CX-7）** |
| 对外入口 | **AxonHub**（kimi/dsh 都走它） |

**路线 B（TensorFold/MLX）已退役**，见文末。

---

## 二、运行时身份：先确认，别假设

`setup.sh --check` 只验 venv 里的 `.pth` 目标（`~/qwen38-exl3/exllamav3`，master），**而生产实际加载的是 worktree**——靠 `serve-local.sh` 的 `EXL3_SPEC_SRC` 把它前置到 `PYTHONPATH`。

```bash
# 真正加载的是哪个引擎
P=$(pgrep -f 'tabbyAPI/main[.]py' | head -1)
readlink /proc/$P/cwd
tr '\0' '\n' < /proc/$P/environ | grep -E '^EXL3_|^HOST=|^CHUNK_SIZE='
```

**worktree 的 `.so` 与 master 的不同**（worktree 含 `EXL3_GR_TUNED`）。判断某开关是否真的存在，要 `strings` **实际加载的那个 `.so`**，不是源码目录。

---

## 三、关键：requeue 计数补丁（**丢了就少报 3 倍速率**）

`exllamav3/generator/job.py` 的 `prepare_for_requeue()`：

```python
"rq_new_tokens": self.rq_new_tokens + self.new_tokens,   # ← 必须带累加
```

上游 2026-09-06 写漏了累加（只留 `self.new_tokens`）。**症状**：`output_chunking`（默认开）下，输出超过 `max_rq_tokens` 的请求**上报 decode 速率最多低 3 倍，完全静默**。实测同负载 4,897 → 15,000 上报 token，17.2 → 51.4 T/s，**耗时不变**——引擎没慢，是计数器少算。

**判据（不需要埋点）**：每个请求必须满足

```
accepted_draft_tokens <= completion_tokens
```

一轮产出 `接受数 + 1` 个 token，所以这个不等式必然成立。**违反即中招。**

补丁在 `exllamav3-patches/0005-rq-token-accounting.patch`，分析见 `docs/decode-drop-root-cause.md`。

---

## 四、配置与实测基线

**配置**（`tabby-config.yml` 渲染 → `~/qwen38-exl3/state/config.yml`，**别手改渲染产物**）：

```
profile concurrent   max_batch_size 4    max_seq_len 262144   cache_size 1048576
cache_mode "8,8"     chunk_size 8192（CHUNK_SIZE）            ngram_ram true
vision true          draft_num_tokens 5  dynamic_draft true
```

**引擎环境变量**（`env.sh`，都有实测依据）：

```
EXL3_MTP_HEAD_N=163840     中文草稿覆盖（65536 时中文散文比不草稿还慢）
EXL3_GR_TUNED=1            融合 GR mix kernel（需 worktree 的 .so）
EXL3_MOE_COOP_WIDE=1  EXL3_GR_INT8=1  EXL3_INT8_GEMV=0
EXL3_DRAFT_CONFIDENCE=0.6  CHUNK_SIZE=8192
```

**实测基线（spark-2，2026-10-08）**：

| 指标 | 值 |
|---|---|
| prefill @40k 冷 | **1,136 T/s** |
| decode 四类均值 | **52.0 T/s**（thinking off）/ **46.6**（on） |
| 接受率 | ~61% |
| 启动 | ~90 s |

**`CHUNK_SIZE` 是最有效的 prefill 杠杆**（不需要重编译）：4096 → 1,004 T/s，**8192 → 1,145（+14%）**，decode 无变化，内存 +2 GiB。**16384 在这台机器上被 autosplit 拒绝**（`NGRAM_RAM=true` 时）。

---

## 五、访问模型

**服务只绑 `10.100.65.1`（CX-7），其余入口全部关闭。**

```
CX-7 链上设备（只有对端 Spark）  ──直连──→  10.100.65.1:8899
其他一切（家庭网/Tailscale）    ──→  AxonHub（发 key）──→  10.100.65.1:8899
```

**隔离是物理的**：CX-7 是两台 Spark 之间的直连电缆，网段上只有两个端点。

**AxonHub**（spark-0d97 的 `hermes` 容器，`127.0.0.1:8090`；DB `~/.hermes/axonhub/axonhub.db`）：

- channel id=5 `qwen-local` → `http://10.100.65.1:8899`
- `models` 表注册了 `Qwen3.8-Flash-Local`（**改 model id 要同时改 `models` 表和 `settings.associations[].channelModel.modelId`**，只改 channel 的 `supported_models` 不够，会 422）
- **改完不用 reload**，它每次请求读 DB

**已验证入口**（从 spark-0d97）：`10.100.65.1` ✅ / `10.100.64.1` ❌ / `192.168.1.6` ❌ / Tailscale ❌ / `127.0.0.1` ❌

---

## 六、不要做

- **❗ 不要把 TabbyAPI 更新到上游 `main`。** 上游 main 要求 `exllamav3 >= 1.5.4`（硬 `RuntimeError`），而 vcruz305 的 fork 版本号停在 `1.5.1.post1`——**即使同步那 21 个提交也不满足**。生产钉在 `local/port-bind-fix`（= 本地补丁 + 上游端口绑定修复 cherry-pick）。完整更新版停在 `local/hybrid-ngram-history`，等 exllamav3 升版本再说。
- **不要 `pip install exllamav3`**（PyPI 或 TabbyAPI wheel）。运行时要 `setup.sh` 从源码编的 fork。stock wheel 缺 GB10 kernel，启动器会拒绝。
- **不要装 TabbyAPI 的 `cu12`/`cu13` extras**，也别跑它的 `start.sh`——会拉 x86_64 轮子。
- **不要用 `vllm-plugin/`**（除非用户明确要 vLLM）。它的 `prepare_pack.sh` **就地改写 pack**，改完 exllamav3 加载不了。
- **不要用 `exllamav3-tabby/beta/`**（单请求 A/B shim）或 **`legacy/`**（stock 1.5.0 历史基线）。
- **不要把 `MAX_SEQ_LEN` 抬过 262144**——那是训练窗口，再长检索就失效。
- **不要用 `NGRAM_RAM=false` 去省内存。** 那张表是 **PLE（逐层嵌入）表**，是**每次前向都要用的模型层，decode 也算**——实测 decode **−33%**（冷启动时 −49~−62%）。省 18 GiB 换这个价不值。
- **不要 `pkill -f tabbyAPI`**（见排查）。
- **不要用 `mv` 轮转日志**（见排查）。

---

## 七、排查：这一路踩过的坑

| 现象 | 真相 |
|---|---|
| **`pkill -f 'tabbyAPI/main.py'` 会让 ssh 会话自己死掉** | 模式匹配到了远端 shell 自己的 cmdline。**按 PID 杀**，杀完确认端口释放。 |
| **`mv` 日志文件后，测试读到的还是旧数据** | **`mv` 不重定向已运行的进程**——它继续写被移走的 inode。要轮转就重启，或每次用新文件名。 |
| **启动报 `Insufficient VRAM in split for model and cache`，但 `free` 显示还有几十 G** | 这是 autosplit 的**余量预检**主动抛 OOM、异常被 `except` 吞掉，单卡没有"下一块设备"可切才抛这句。**真正的 OOM 消息（带着测得的 transient 数字）没打出来。** 相关：`CHUNK_SIZE` 越界。 |
| **`Port 8899 is currently in use. Switching to 8900.`** | **已修**（cherry-pick 了上游 2fd6cc7）。现在端口被占会**明确报错退出**，不再静默换端口。**之前这个坑让一整轮测试打在了旧引擎上。** |
| **速率掉到 1/3** | 见第三节 requeue 补丁。 |
| **`check_memory` 的估算不随 `CHUNK_SIZE` 变化** | 已修——现在含 `chunk × vocab × 2 B` 项。但它**仍是下界**（注意力/MoE 暂存没建模），这正是 16384 算术上够却被拒的原因。 |
| **`strings` 源码目录找不到某个开关** | 要看**实际加载的 `.so`**。backlog 里 `EXL3_MOE_FUSED_ROWS`、`EXL3_QC_PF_TWO_PASS_MIN_Q`、`EXL3_MOE_RECON_*` **压根不在二进制里**，设了是静默无操作。 |

**通用的重启纪律**：

```bash
P=$(pgrep -f 'tabbyAPI/main[.]py' | head -1); kill $P
for i in $(seq 1 40); do pgrep -f 'tabbyAPI/main[.]py' >/dev/null || break; sleep 1; done
ss -ltn | grep -q ':8899' && echo '!! 端口仍被占用，别启动'
bash exllamav3-tabby/serve-local.sh
# 确认新进程的启动时间晚于你的改动，且 readlink /proc/PID/fd/1 指向预期的日志
```

---

## 八、两条路线

仓里保持**两条清晰隔离的路线**：

- **路线 A：`exllamav3-tabby/` + `vllm-plugin/`**（原生 EXL3，当前生产）
- **路线 B：`tensorfold-mlx/`**（TensorFold/MLX，**已退役**）

**路线 B 退役的原因**：那套跑的是**自制转换的 MLX 包**（从 `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated` 转的），**abliteration 损伤了模型质量**。作为对照，Mia 的方案（`MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold`）用的是**官方未改动的 `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`**——质量差距的根源是去审查本身，不是转换技术。

**两条路线的模型来源不同，别混**：

| 包 | 来源 | 大小 |
|---|---|---|
| `Qwen3.8-Flash-Next-Uncensored-EXL3-3bpw`（**A，在跑**） | `orcarouter/Qwen3.8-Flash-Next-Uncensored` → Lygodactylus EXL3 | 68 G |
| `Huihui-...-MLX-4bit-g32`（B，退役） | `huihui-ai/...-abliterated`，**自制转换** | 106 G |
| `Vontra-...-MLX-4bit-MTP` | 官方（**审查版**，Mia 用这个） | 106 G |
| `Qwen3.8-Flash-Next-EXL3` | 官方（**审查版**） | 84 G |

**要跑质量对照，用官方包做基线**——abliterated 和官方不是同一个模型，分数不可互推。

---

## 九、两机（EXL3 **不能**跨机跑这个模型）

**跨机 TP/EP 不存在**（`model_tp.py` 是 `multiprocessing`，单机多卡）。**跨机 PP 存在但只认 `Glm5NextForConditionalGeneration`**，`Qwen4ExpForConditionalGeneration` 启动即 `ValueError`。而且 PP 是串行的（无微批重叠），**单流不加速**。

**要 2 机加速走 vLLM TP2 那几套（`:8888`）**，代价是停掉 EXL3（内存互斥）。
