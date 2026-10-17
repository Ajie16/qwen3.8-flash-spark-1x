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

### ③ `--max-num-batched-tokens`：4096 是选择，不是上限

**早先这里写的是"只能是 4096，8192/16384 会撞硬边界"。这个结论是错的，实测推翻。**

2026-10-17 复测：**8192 和 16384 都能干净启动，零 mismatch 警告，decode 完全不变。**

```
nbt=8192   无补丁启动 ✓    C1 107.8  C4 288.9  C8 435.8
nbt=16384  无补丁启动 ✓
nbt=4096   无补丁启动 ✓    C1 109.0  C4 287.0  C8 435.6
```

**之前那些失败是竞态，不是尺寸限制。** 报错来自 `vllm/v1/worker/b12x_startup.py:291` 的跨 rank 一致性检查：

```python
choices = [key for key, ranks in participants_by_key.items()
           if ready_by_key.get(key) == set(ranks)]   # 只有所有参与 rank 都就绪才批准
...
if authorized != pending:
    raise RuntimeError("preparation ranks reached incompatible tuning boundaries")
```

**同一个 nbt 值可能这次失败、下次通过。** 所以遇到这个错**先重试**，不要清缓存、更不要打补丁。

**为什么仍然选 4096 —— 换了个理由：内存。**

| nbt | 可用内存（两台） | prefill @45k | decode C1 |
|---|---|---|---|
| **4096（当前）** | **44.0 / 44.7 GiB** | 2,496 | **109.0** |
| 8192 | 41.5 / 41.9 GiB | **2,676（+7%）** | 107.8 |
| 16384 | **35.1 / 35.7 GiB** | — | — |

**每翻一倍 nbt，每台少 6–7 GiB**（prefill 激活工作区 + CUDA graph 捕获形状随 nbt 线性增长），
**换来 prefill +3–7%，decode 零变化。**

**6.4 GiB 换算成 KV 是约 55 万 token（12 KiB/token），相当于多两路满窗并发。这比 prefill 的 7% 值。**

**`fp4-b12x/apply_tuning_patch.py` 仍然保持存档状态** —— 它当初是为了绕过那个"边界"，
现在边界不存在了，补丁也就没有用途。**不要用它。**

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

分两个层次：**每步花多少时间**，和**每步产出多少 token**。两者的比值就是吞吐。

### 先看投机解码在各并发下有没有退化

| 并发 | 聚合 tok/s | 每流 | **接受率** | **tok/步** |
|---|---|---|---|---|
| 1 | 103.7 | 103.7 | **93.8%** | 4.76 |
| 2 | 183.6 | 91.8–94.3 | **95.2%** | 4.82 |
| 4 | 273.6 | 68.4–70.8 | **96.3%** | 4.86 |
| 8 | 411.2 | 51.4–54.9 | **95.7%** | 4.84 |

**接受率随 batch 反而略升（93.8% → 96.3%），tok/步稳定在 4.8。**
**所以草稿质量不是瓶颈，也不是并发退化的原因 —— 这一条排除了一个很自然的猜测。**

### 引擎每步的耗时

```
吞吐 = N × tok_per_step / step_time
  →  step_time = N × 4.8 / 吞吐
```

| 并发 | step_time | 相对 C1 |
|---|---|---|
| 1 | **44 ms** | 1.0× |
| 2 | 52 ms | 1.2× |
| 4 | 70 ms | 1.6× |
| 8 | **93 ms** | 2.1× |

**batch 涨 8 倍，每步只慢 2.1 倍 —— 所以吞吐涨 3.9 倍。**

**这是投机解码的核心：一步验证的 token 越多，权重读取被摊得越薄。**

### C1 的 44 ms 花在哪

用 TP=1 做对照：

```
TP=1:  63.9 tok/s → 13.3 步/秒 → 75 ms/步
TP=2: 109.0 tok/s → 22.7 步/秒 → 44 ms/步        比值 1.71（理想 2.0）
```

两点：

1. **每步主要成本是"读一遍该步激活的专家权重"** —— 两台分摊后接近减半
2. **差的 15% 就是跨机 all-reduce**（与下面独立算出的 16% 一致）

**C1 每步只产出 4.8 个 token，却要读约 590 MB 专家权重** —— 算术强度极低，
GPU 只能跑到 246 GB/s 的四分之一。

### 并发时发生了什么

**每步读的权重基本不随 batch 增长**（同一批专家服务所有请求），但**验证的 token 数线性增长**：

```
C1:  590 MB 权重 → 4.8 token      摊销 123 MB/token
C8:  590 MB 权重 → 38 token       摊销  15 MB/token
```

**C8 的带宽需求（248 GB/s）正好撞到实测峰值 246 GB/s —— 到顶了。**

### 一张表

| 并发 | 限制因素 | step_time | 带宽占用 | 还能不能快 |
|---|---|---|---|---|
| **1** | **固定成本**：读该步激活的专家权重 | 44 ms | 26% | ❌ 除非增加每步 token |
| 2 | 过渡 | 52 ms | 43% | ⚠️ |
| 4 | 过渡 | 70 ms | 67% | ⚠️ |
| **8** | **带宽饱和** | 93 ms | **101%** | ❌ 到顶 |
| 16 | 带宽 + expert 复用 | ~120 ms | ~148%（复用后 ≈100%） | ⚠️ 仅靠复用 |

### 理论量级校核

```
每 expert = 3 × 2560 × 640 = 4.92 M 参数 = 2.46 MB（4-bit）
全模型 expert = 48 层 × 512 expert × 2.46 MB = 60.4 GB（每 rank 一半 = 30.2 GB）
每 token 激活 10/512 → 每 rank 读 590 MB
```

### 想再快，只有三条路，都不通

| 手段 | 为什么不行 |
|---|---|
| **加大 K** | k=6 实测比 k=4 慢 6.72%（每步变长超过产出增长），见草稿器一节的扫描 |
| **提高接受率** | 中文 27% 提到英文水平也只多 ~3.5 tok/步，而中文的机制是 MTP 头训练数据，配置改不动 |
| **减少 all-reduce** | 已经压到 15%；EP 能做到接近 0，但 CSF 格式拒绝 |

**结论：C1 端受限于"每步 44 ms 的固定权重读取被 4.8 个 token 摊销"，并发端受限于内存带宽。
两端都是硬的，中间没有可调空间。**

**注：不开投机解码时单流只有约 22 tok/s（本机同权重、AGENTS.md 记录的 Mia 方案为 24.8），
所以 4.8 tok/步已经是这套权重的极限利用率。**

## 一次被误判为"性能退化"的测量

**2026-10-17 曾记录"同一配置从 C1 108 掉到 78.7"，后来确认不是退化：那是后台有请求在跑时的争用。**

同日空闲状态下复测：C1 **107.3** / C2 178.1 / C4 **278.1** / C8 **420.9**，与最初的 108.0 / 283.1 / 432.0 一致。

**教训：decode 基准必须在服务完全空闲时跑。** 判据是 `vllm:num_requests_running` 为 0、GPU 利用率采样持续为 0%、`generation_tokens_total` 不增长。`bench.py` 只在每个模式开头做两次预热，挡不住外部并发。

## 反审查（abliterated）变体

`edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local` 是同一个 checkpoint，
用 Huihui v2 的 abliteration 替换了 96 个 MXFP8 投影张量。**其余全部保留 NVIDIA 的原始字节** ——
README 原话："Routed NVFP4 experts and their CSF scales, PLE, MTP, vision and tokenizer retain the
NVIDIA checkpoint bytes."

### 只下载 2 个分片，不是 95 GiB

**21 个分片里只有 2 个不同**，用 sha256 逐个比对确认过：

```
model-00001-of-00021.safetensors   ✗ 不同（96 个变更张量）
model-00015-of-00021.safetensors   ✗ 不同（96 个变更张量）
其余 19 个                          ✓ sha256 完全相同
```

**变更张量共 96 个**，位置很集中：

| 模块 | 层数 |
|---|---|
| `mlp.shared_expert.down_proj` | 48 |
| `linear_attn.out_proj` | 36 |
| `self_attn.o_proj` | 12 |

`config.json` 与官方**完全一致**（diff 0 行）。

**所以做法是硬链接**：只下 2 个分片（9.86 GB），另外 19 个 `ln` 到官方目录。

```bash
# 1. 下载 2 个分片 + index + 小文件（aria2c 16 线程，hf-mirror，约 90 秒）
# 2. 硬链接其余 19 个
cd ~/models/Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local/tensors
SRC=~/models/Qwen3.8-Flash-Next-FP4-Local/tensors
for f in *.safetensors; do
    [ -f "$f" ] && continue
    ln "$SRC/$f" "$f"
done
```

**必须用硬链接，不能用软链接。** 容器只挂载 `MODEL_DIR:/model`，绝对路径软链指向
`/home/xujie/models/Qwen3.8-...` 在容器里不存在，会直接加载失败。硬链接在同一文件系统上，
容器看到的是真实文件。代价只有 9.86 GB 磁盘，不是 96 GB。

### 部署

```bash
./tf-restart.sh --model /home/xujie/models/Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local
```

启动器里 `MODEL_DIR="${MODEL_DIR:-<官方路径>}"`，所以直接传环境变量也行。

**实测启动 301 s，KV 2,447,669 token / 9.34×，与官方版一致。**

### 实测：性能不变，审查解除

| 并发 | 官方 | Huihui | 差异 |
|---|---|---|---|
| 1 | 107.3 | **108.3** | +1% |
| 4 | 278.1 | **285.4** | +2.6% |
| 8 | 420.9 | **433.0** | +2.9% |

**性能在噪声范围内，略高。** 符合预期 —— 只换了 96 个 dense 投影，MoE 路由、CSF scale、PLE、
MTP 全部是原字节，访存模式不变。

行为对比（关思考、temperature 0）：

| 探测 | 官方版 | Huihui 版 |
|---|---|---|
| 粗俗用语 | 会软化 | **直接输出，不软化** |
| 虚构犯罪场景 | 加道德说教 | **只写场景** |
| 社工攻击手法 | 拒绝 | **给出详细分类** |
| 常规问答 | 正常 | 正常 |

### 需要知道的限制

**`runtime_validated: false`。** 上游的 `runtime-qualification.json` 记录了一次失败的检查：
推理答案是 445（正确），但在要求"只给整数"的指令下仍附带了解方程过程。它把这个记为
"completed with findings"，不是全项通过。

**同时提到两个版本（含 NVIDIA 原始版）都有散文质量上的局限，以及启动时会用 swap。**

这是**量化 delta 转移**，没有做新的 QAD 训练或激活校准 —— 即"把 donor 模型的差异搬过来再量化回去"，
不是重新训练。**不能据此推断 refusal rate 与 donor 等价，也不能推断基准分数持平。**

**License**：NVIDIA Open Model Agreement + Qwen Community License 同时适用。

## 草稿器：是什么、能不能换、为什么换不了

### 当前用的是模型自带的 MTP 头

```
mtp_num_hidden_layers          1        单层
mtp_use_dedicated_embeddings   False    ★ 共享主模型 248,320 全词表
量化                           W4A16_NVFP4 group 16（专家）+ NVFP4（LM head）

--speculative-config {"method":"mtp","num_speculative_tokens":4,
                      "draft_tensor_parallel_size":2,"draft_sample_method":"greedy"}
```

服务端统计的平均接受长度在 **2.06–4.95** 之间随内容剧烈波动。

**`mtp_use_dedicated_embeddings: False` 是理解后面一切的关键**：MTP 头直接用主模型的完整
词表，**不存在"独立的草稿词表"这个可替换对象**。

### 实测接受率（本轮所有测量）

**注意：默认温度是 1.0，不是 0。之前所有基准都用 `temperature=0`，高估了。**

| 用例 | T=0 | **T=1.0（真实默认）** | 损失 |
|---|---|---|---|
| 数数（最可预测） | 97.3% | 97.3% | 0 |
| JSON | 79.0% | 73.2% | −5.8 |
| 代码 | 73.2% | 71.0% | −2.2 |
| 英文散文 | 50.2% | 47.5% | −2.7 |
| 中文技术 | 43.5% | 38.3% | −5.2 |
| **中文散文** | 30.2% | **27.2%** | −3.0 |

**损失不大**，因为 rejection sampling 会校正分布。**但真实工作点是 27%，不是 30%。**

**数数那个 97% 不能当参考**，极端可预测。

### 试过、无效的两个假设

**① MTP LM head 改 BF16（`VLLM_MTP_NVFP4_LM_HEAD=0`）**

假设：LM head 是 `hidden → 248,320 词表` 的投影，中文 token 在词表里是少数，
4-bit 的 per-group 量化误差对它们更不利。

```python
mtp.py:785   lm_head_quantization="nvfp4" if envs.VLLM_MTP_NVFP4_LM_HEAD else None
```

结果（T=0）：中文散文 **30.0% → 30.2%**，其余全部在噪声内。**假设被推翻，不是量化误差。**

**② 草稿改采样（`draft_sample_method=probabilistic`）**

假设：`greedy` 把草稿分布当 one-hot，在温度 1.0 下是差近似；`probabilistic` 用完整
logits 做 ratio test，是教科书正确形式。

结果（T=1.0）：中文散文 **27.2% → 26.3%**，技术类 +5.2 但 JSON −7.1，**没有净收益**。
吞吐也无变化（C1 107.3 vs 108.3）。

**结论：两个旋钮都调过了，中文接受率不动。这是 MTP 头训练数据以英文为主造成的模型属性。**

### ★ DSpark：架构性不可行，不要再试

`DSparkModelTypes = Literal["dspark"]`，代码里只显式支持
`deepseek_v4` / `deepseek_v41` / `qwen3_dflash`。我们的 `model_type` 是 `qwen4_exp`。

但真正的阻碍不是白名单，是**硬件后端**。上游作者的 DFlash 草稿模型 card 里写得很直接
（`PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash`）：

> A DSpark variant of this drafter was also trained and is **not** published. It serves at −11.0%
> against native MTP k=3 and roughly doubles the chat regression, and its per-request confidence
> gate **cannot run on this target at all**: vLLM's adaptive verification needs to trim
> verification requests on device, and `GDNAttentionBackend` does not support that. Since
> Qwen3.8-Flash-Next has **36 GatedDeltaNet layers**, that check can never pass. **This generalises
> to every Mamba-class hybrid target.**

三点：

1. DSpark 版本训练过，**实测比原生 MTP k=3 慢 11%**
2. 它的自适应验证依赖 `GDNAttentionBackend` 的设备端裁剪，vLLM 这个后端做不到
3. Qwen3.8-Flash-Next 有 36 层 GDN —— **这个检查永远不可能通过，且对所有 Mamba 混合架构成立**

### DFlash：能跑，但对聊天是负收益

存在一个 1 GB 的草稿模型 `PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash`
（5 层、block 7、498M 参数、BF16，taps `[3,15,23,35,43]`）。

**作者自己的结论：**

> this is a **maths drafter**, a code wash, and it makes **chat slower at every block size we can
> serve**

实测（block 5，对原生 MTP k=4，2×DGX Spark 同构环境）：

| 负载 | DFlash | 95% CI |
|---|---|---|
| **数学** | **+28.04%** | [+22.80, +33.37] |
| 代码 | +0.37% | [−3.17, +4.21]（跨零） |
| **聊天** | **−5.90%** | [−7.96, −3.89] |

**聊天在每个可服务的 block 上都退化**：block 2/3/4/5/7 = −6.13% / −3.87% / −3.90% / −5.90% / −11.32%。
综合只有 **+3.87%**，而作者说 boot-to-boot 方差和这个效应一样大。

**中文创作属于 chat，正好是它退化的那一类 —— 装上去中文会更慢。**

而且装它不开箱即用：

> **Stock vLLM cannot load this drafter.** You need three control-flow patches **and** a six-file
> adapter overlay.

三个补丁分别修 `model_runner.py` 两处、`dflash/speculator.py:69`、模型 allowlist；
六个文件的 overlay 做 aux 捕获、anchor 约定、Triton kernel offset、target-sharing 注册、
scheduler 槽位、dflash allowlist。**并且钉死在特定 vLLM commit 和镜像 sha256 上** ——
和我们的 CSF 专属 B12X 镜像耦合风险很高。

### 原生 MTP 的 k 值扫描：k=4 已经是最优

同一份 card 里的扫描（对 k=4 归一化）：

| 配置 | 聚合 tok/s | 相对 k=4 |
|---|---|---|
| MTP k=1 | 41.321 | −17.97% |
| MTP k=3 | 50.071 | −0.60%（CI 跨零，与 k=4 不可区分） |
| **MTP k=4（当前）** | **50.371** | — |
| MTP k=6 | 46.987 | −6.72% |

**k=4 就是最优。** 与本文档"瓶颈在哪"一节的推算一致：verify 是每个 round 一次固定成本，
所以 K 大一些更划算，但过大时草稿成本超过收益。

### 为什么中文提不上去，以及唯一有效的路

**MTP 头的训练数据以英文为主。** 证据链：

- 英文散文 47.5% vs 中文散文 27.2% —— 同样结构、同样量化，差 20 个点
- 中文技术 38.3% 高于中文散文 27.2% —— **不是"中文"整体差，是"中文创造性内容"最难预测**
- 换 LM head 精度无效 → 不是量化误差
- 换草稿采样方式无效 → 不是采样策略

**唯一被实测证明有效的做法是可替换的草稿词表**，那是方案 B（TensorFold）独有的机制：
它的 MTP 头用 102,089 行的缩减草稿词表，换成中文扩展版后
**中文 decode +25–40%、接受率 32% → 57%**。

**这套机制在 vLLM/B12X 上不存在**（`mtp_use_dedicated_embeddings=False`）。

**但要看清代价**：TensorFold 单流只有 47–60 tok/s，是现在 107 tok/s 的一半。
**为了中文接受率把整体吞吐换掉一半，通常不划算** —— 而且单流的瓶颈本来就是 HBM 带宽
（C1 只跑到 246 GB/s 的 26%），提高接受率对单流吞吐的边际收益很小。

**如果中文是主要负载，正确做法是走方案 B，不是在这套上继续调。**
