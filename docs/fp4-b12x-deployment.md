# 路线 C · nvidia FP4-Local + B12X，双机 TP=2

> 与 A（EXL3）、B（TensorFold）的区别：**这条跑在两台 Spark 上**，而且是目前本机吞吐最高的方案。
> A、B 都是单机；C 用 CX-7 RoCE 做张量并行，16 并发能到 616 tok/s。

## 为什么是这套组合

权重和镜像**互相绑定，不能混搭**。

`nvidia/Qwen3.8-Flash-Next-FP4-Local` 的 95.12 GiB / 21 分片里有 **147,456 个 `nvfp4_csf_*` 张量**：

```
model.language_model.layers.0.mlp.experts.0.down_proj.weight_scale.nvfp4_csf_fixed
model.language_model.layers.0.mlp.experts.0.down_proj.weight_scale.nvfp4_csf_exceptions
```

这是 NVIDIA 对一个**新的无损 scale 压缩格式**（CSF）的落地，上游 vLLM 不支持。四个镜像的实际支持情况：

| 镜像 | CSF 相关文件 | b12x 包 | 能跑这个权重 |
|---|---|---|---|
| **`eugr/spark-vllm-b12x`** | **15** | ✅ | ✅ |
| `vllm/vllm-openai:qwen38-flash-next` | 0 | ❌ | ❌ |
| `ghcr.io/anemll/dspark-vllm-gx10` | 0 | ✅ | ❌（旧版无 CSF） |
| `lovedheart/qwen38-flash-next` | — | — | ❌（amd64） |

老镜像 `vllm/vllm-openai:qwen38-flash-next` 里那 4 个"CSF 命中"全是 `.so` 二进制里的偶然字节，**Python 代码零支持**。它跑 `RadixArk/Qwen3.8-Flash-Next-NVFP4`（标准 NVFP4，123.6 GiB）没问题，但那个包是另一条路。

**量化配置**是 ModelOpt 混合精度，545 层：

| 算法 | 层数 | 用在哪 |
|---|---|---|
| MXFP8 | 438 | dense linear |
| W4A16_NVFP4 | 58 | vision MLP + merger |
| NVFP4 | 49 | routed experts |

对应 `--quantization modelopt_mixed`。

## 参数来源

**不是自己试出来的**：来自 [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker)（★2408，"Docker configuration for running VLLM on dual DGX Sparks"）的
`recipes/qwen3.8-flash-next-nvfp4-cluster.yaml`，是**唯一针对这个 checkpoint 的公开双机配方**。副本存在 `fp4-b12x/qwen3.8-flash-next-nvfp4-cluster.yaml`。

**遇到这台机器的 vLLM 问题，先去查它的 recipes 目录，比翻日志快。**

它也确认了两件事：`max_num_batched_tokens: 4096`、**没有 `--enable-expert-parallel`**。

## ★★★ 三条决定成败的设置

### ① `--device /dev/infiniband:/dev/infiniband`（加 `--cap-add SYS_NICE`）

没有它容器看不见 `uverbs0-3` / `umad0-3` / `rdma_cm`，**NCCL 静默退回 TCP**，TP=2 从 616 掉到 281。

**判据**：`/sys/class/infiniband/roceP2p1s0f0/ports/1/counters/port_xmit_data` 应持续增长（正常工作下每请求约 19 MB，一天跑下来 242 GB）。

**`ss` 会误导** —— OOB 协调通道本来就是 TCP，RDMA 通了也会有 TCP 连接。

**确认**：`NCCL_DEBUG=INFO` 找 `NCCL INFO Using network IB` 和 `NET/IB: ... provider=Mlx5 speed=200000`。

### ② `VLLM_ENABLE_ROCE_ALLREDUCE=1` + `VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB`

裸 NCCL collective 每 token 约 122 次跨机同步、每次约 40 µs，把第二台机器省下的算力全吃了 —— **TP=2 只有 +23%**。换成 vLLM 自己的 RoCE all-reduce 路径后 **+75%**。

配套还要 `--speculative-config '{..., "draft_tensor_parallel_size": 2}'`：**MTP 草稿也要跨两卡**，否则投机解码只用一台的算力。

启动后日志里应该有：

```
b12x_roce_all_reduce.py:295  RoCEnante all-gather is live: first routed shard is (4096, 160) bfloat16
b12x_roce_all_reduce.py:277  RoCEnante all-reduce is live: first routed all-reduce is 81920 bytes
```

### ③ `--max-num-batched-tokens` 只能是 4096

8192 / 16384 都报 `RuntimeError: preparation ranks reached incompatible tuning boundaries`，**清 B12X 编译缓存无效**（试过三次）。

**这不是写死的尺寸表**，而是 `vllm/v1/worker/b12x_startup.py:291` 的跨 rank 一致性不变量：

```python
choices = [key for key, ranks in participants_by_key.items()
           if ready_by_key.get(key) == set(ranks)]   # 只有所有参与 rank 都就绪才批准
...
if authorized != pending:
    raise RuntimeError("preparation ranks reached incompatible tuning boundaries")
```

补丁掉它（`fp4-b12x/apply_tuning_patch.py`，**已存档为死路**）只把失败推到
`flashinfer/experimental/b12x/preparation/session.py:1027` 的 `tuning consolidation does not match pending races`。**是连锁断言，逐道打补丁既打地鼠又可能静默算错。**

## 实测性能

解码，sparkDash 计数提示词，关思考，丢弃预热：

| 并发 | 聚合 tok/s | 每流 |
|---|---|---|
| 1 | **111.5** | 111.5 |
| 2 | 187.3 | 93.8 |
| 4 | 291.2 | 73.0 |
| 6 | 355.8 | 60.0 |
| 8 | 442.2 | 55.3–57.3 |
| 12 | 527.5 | 44.0–46.0 |
| **16** | **616.1** | 38.5–40.8 |

**prefill 2,300–2,400 tok/s，5k → 179k 全程持平**（B12X 的 FP4 tensor-core 路径，不像 EXL3 那样被反量化卡住）。

**中文**：散文 35.4 tok/s = **62 字/秒**；技术说明 44.8 = **93 字/秒**；代码 57.2。（中文约 1.75 汉字/token。）

### 与 A/B 两条路线的对比

| 方案 | 单流 | 最佳聚合 |
|---|---|---|
| A · EXL3 + TabbyAPI | 48.3 | 56（4 并发，负扩展） |
| B · TensorFold + MLX | 47.7 | 110 |
| **C · FP4-Local + B12X TP=2** | **111.5** | **616（C16）** |

**唯一的取舍：C 是审查版权重（`base_model` 是官方 Qwen）。** 要抗审查只能回 A 或 B。

## MTP 接受率：强烈依赖内容类型

| 用例 | 接受率 |
|---|---|
| 数数 | 98.1% |
| JSON | 77.8% |
| 代码 | 69.1% |
| 英文散文 | 52.3% |
| 中文技术 | 50.0% |
| **中文散文** | **28.6%** |

**平均 62.6%，跨度极大。** 中文创作慢的直接原因就是它 —— `num_speculative_tokens=4` 是为代码/JSON 调的，在 28.6% 接受率下 4 个草稿只兑现 1.4 个。

**数数那个 98% 不能当参考**，极端可预测。

## 内存：两个会静默吃内存的泄漏

两台都是 121.69 GiB 统一内存，CPU/GPU 同一个池。**`tf-restart.sh` 已自动处理这两条**，它们合计曾吃掉 55 GiB。

| 泄漏 | 规模 | 原因 |
|---|---|---|
| **`/dev/shm` 残留** | **14 GiB/台** | 容器用 `--ipc host`，**`--shm-size` 完全失效**，用的是宿主机 60.8 GB tmpfs；每次运行留 `psm_*`，稀疏文件所以 `du` 只报 7.4 MB，但 tmpfs 照算 Shmem |
| page cache | 最高 41 GiB | vLLM 用 **`MemFree`** 判内存（不是 `MemAvailable`），上一次读 95 GB 权重的缓存不算可用 |

**修好后：ac8f 76 GiB 已用 / 44.6 GiB 可用；0d97 95 GiB 已用 / 26.5 GiB 可用。**

**0d97 多用 19 GiB，几乎全是 CLM 服务**（`serve_encoder.py` 15 GB + CLM 2.9 GB = 17.6 GiB，跑在 8091）。两台本身对称。

### PLE 表放 NVMe 还是 RAM

`VLLM_PLE_TABLE_MEMORY` 三种模式：

| 模式 | 位置 | 权重占用（rank0） |
|---|---|---|
| `ram` | 进程内存 | 55.78 GiB |
| `shared` | `/dev/shm`（仍是内存） | 55.78 GiB |
| **`disk`（当前）** | **NVMe，`io_uring` 直读** | **38.13 GiB** |

**实测差 2–4%：**

```
              C1      C4      C8
RAM(shared)  111.5   291.2   442.2
NVMe(disk)   108.0   283.1   432.0
```

**用 2–4% 换每台 13 GiB 内存，在 0d97 只有 13 GiB 余量时是对的取舍。**

**关于"表只放一台 RAM、对端走 CX-7 读"**：技术上可做（`nvmet-rdma` + `brd` 把 RAM 块设备导出，对端 `nvme connect` 当本地盘，vLLM 完全无感），**但不划算**：

- 内存账是反的 —— 单边放意味着那台扛整个 32.6 GiB，而两台对称时各放一半也是 32.6 GiB 总量
- 收益上限只有 2–4%（PLE 查表根本不在关键路径上）
- 唯一的适用场景是**两台内存不对称**：让内存多的那台扛表

## KV 按 C8 预留

`--kv-cache-memory 25769803776`（24 GiB）= 8 × 262,144 token × 实测 **12.003 KiB/token**，给到 **9.34× 并发**（2,447,669 token）。

**不要靠 `--gpu-memory-utilization` 让它贪心分配** —— 0.70 时它拿走 39.52 GiB / 345 万 token（13.17×），把 rank1 压到只剩 3.0 GiB 系统内存。

`--gpu-memory-utilization` 现在只是外层上界：rank0 0.70，**rank1 0.64**（它和 CLM 共用一台）。

## 启动成本

**热启动约 5 分钟**，构成如下（这是硬成本，换参数躲不掉）：

| 阶段 | 时间 | 能否压 |
|---|---|---|
| b12x ROUTE 读 95 GiB | **134s** | ❌ 不可压（O_DIRECT 直读 + Python 路由；page cache 预热 126→134s，**无效**） |
| CUDA graph 捕获 | **95s** | ✅ 限制捕获尺寸可省 60–80s |
| KV 分配 + 引擎收尾 + API | 107s | ⚠️ |
| 容器启动 + import | 30s | ❌ |
| 内核调优 | 3s | ✅ 已缓存 |

**⚠️ 不要清 `~/.cache/nvidia-fp4-local/b12x`**：会触发 3309 个 `gdn_prefill` 内核重编译（+170s），而且不解决任何已知问题。

## 工具与用法

```bash
# 日常：一条命令完成 停→清缓存→清shm→启动→等待→报告
./tf-restart.sh                        # 当前配置
./tf-restart.sh --ple disk --seqs 16   # 改参数（多个参数一条命令带上，别分开重启）
./tf-restart.sh --gmu0 0.70 --gmu1 0.64
./tf-restart.sh --fast                 # 限制 cudagraph 捕获，快 60-80s

# 基准
python3 bench.py decode  --n 1,2,4,8,16
python3 bench.py prefill --sizes 4000,16000,32000,64000,128000
python3 bench.py accept
```

**`tf-restart.sh` 会同时修改本地和远端两份 `serve-fp4-tp2.sh`** —— rank1 跑本地那份、rank0 跑远端那份，只改一份会让两台用上不同的 PLE 模式，而且**不报错**，只有各自日志里的权重体积不同（38 GB vs 56 GB）。

## 文件

| 文件 | 作用 |
|---|---|
| `serve-fp4-tp2.sh` | 启动器。`0` = head（出 API），`1` = worker（headless）。**先起 worker。** |
| `tf-restart.sh` | 重启封装，含两处内存清理 |
| `bench.py` | `decode` / `prefill` / `accept` 三种模式 |
| `apply_tuning_patch.py` | **存档的死路**，不要用 |
| `b12x_startup.py.patch` | 上面那个脚本产生的 diff |
| `qwen3.8-flash-next-nvfp4-cluster.yaml` | 上游参考配方 |

## 瓶颈在哪

两个阶段，限制因素不同。

### 先算理论访存量

```
num_hidden_layers      48
num_experts            512
num_experts_per_tok    10
hidden_size            2560
moe_intermediate_size  640

每 expert = 3 × 2560 × 640 = 4.92 M 参数 = 2.46 MB（4-bit）
全模型 expert = 48 × 512 × 2.46 MB = 60.4 GB（每 rank 一半 = 30.2 GB）
每 token 激活 10/512 → 每 rank 读 590 MB
```

### 各并发下的带宽需求（本机实测峰值 246 GB/s）

| 并发 | 聚合 tok/s | 带宽需求 | 占峰值 |
|---|---|---|---|
| **1** | 107.3 | **63 GB/s** | **26%** |
| 2 | 178.1 | 105 | 43% |
| 4 | 278.1 | 164 | 67% |
| **8** | 420.9 | **248** | **101% ← 饱和** |
| 16 | 616.1 | 363 | 148% ← 靠 expert 复用才可能 |

C16 算出来超过峰值，是因为 16 个请求各自的 top-10 有重叠（512 选 10、抽 160 次，约 86% 命中不同 expert），实际访存低于这个上界。**趋势是明确的：单流只跑到峰值的四分之一，高并发逼近上限。**

### 单流（107 tok/s，9.3 ms/token）：延迟受限

```
GPU 利用率     93–95%（nvidia-smi 的"有 kernel 在跑"，不是算力用满）
实际 FLOPS     ~5% 峰值
内存带宽       26%
```

**batch=1 把每个 expert 的 GEMM 退化成 GEMV**：`1×2560 @ 2560×640`，只用到 tensor core 天然 tile 的一行。48 层 × 10 expert = **每 token 480 个微小 GEMM**，每个都在等 2.46 MB 权重从 HBM 过来，而算的部分可以忽略。

每 token 读 590 MB 只做约 4.8 GFLOP —— **算术强度 0.08 FLOP/字节**。任何 GPU 在这种负载下都只能跑到带宽的一小部分，这不是配置问题。

### 高并发（C8 以上）：带宽饱和

```
421 tok/s × 590 MB = 248 GB/s ≈ 实测峰值 246 GB/s
```

**C8 已经撞在 `246 GB/s ÷ 590 MB/token ≈ 417 tok/s` 这条线上。** C16 的 616 超出部分全部来自 expert 复用，不是线性扩展。

### 次要因素：跨机 all-reduce 约 16%

TP=1 单机 63.9，TP=2 107.3 = **1.68×**（理想 2×）：

```
1 - 1.68/2 = 16%
```

每 token 122 次跨机 collective，总流量 355 KB（204 MB ÷ 600 token）——带宽上微不足道，但每次是一个同步点、约 40 µs 延迟。这部分已经靠 `VLLM_ENABLE_ROCE_ALLREDUCE` 从 40% 压到 16%，再压空间不大。

### 结论：没有可调的空间了，唯一杠杆是并发

这台机器的物理上限就是 246 GB/s。decode 的本质是"每 token 把 590 MB 权重从 HBM 搬一遍、做一遍几乎免费的计算"，所以：

```
最大 tok/s ≈ 246 GB/s ÷ 590 MB/token ≈ 417 tok/s（无 expert 复用时）
```

想要更高只能改模型本身：更低位宽（2-bit，质量掉）、更多机器（不解决，TP 已有 16% 通信成本）、或模型层面的 expert 复用。**现在这个配置已经是这套权重在这台硬件上的实际最优解。**

## 一次被误判为"性能退化"的测量

**2026-10-17 曾记录"同一配置从 C1 108 掉到 78.7"，后来确认不是退化：那是后台有请求在跑时的争用。**

同日空闲状态下复测：C1 **107.3** / C2 178.1 / C4 **278.1** / C8 **420.9**，与最初的 108.0 / 283.1 / 432.0 一致。

**教训：decode 基准必须在服务完全空闲时跑。** 判据是 `vllm:num_requests_running` 为 0、GPU 利用率采样持续为 0%、`generation_tokens_total` 不增长。`bench.py` 只在每个模式开头做两次预热，挡不住外部并发。
