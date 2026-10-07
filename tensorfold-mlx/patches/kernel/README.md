# patches/kernel/ — TensorFold 0.6.1 内核级修改（venv 全量补丁）

20 个 unified diff，把上游 pristine v0.6.1 变成生产 venv 的完整代码状态。
**基准**：`ashhart/TensorFold` tag `v0.6.1`（checkout 到 `src/` 目录应用，`git apply` 已验证全部干净应用）。
**完整性**：已做 round-trip 验证（pristine + 本目录全部补丁 = venv 逐字节一致，唯一例外是 `draft_vocab.txt` 词表，见 `../draft-vocab/`）。

## 分组与状态（权威解释见仓库根 `docs/EXL3-FIX.md`）

### 生产在跑的性能修改（无脚本，仅此处可重放）

| 补丁 | 修改 | 状态 |
|---|---|---|
| `families_qwen4_exp_cuda_mtp.py.patch` | `logits=False`：prefill 吸收 MTP 缓存时跳过草稿词表投影 | ✅ 生产 |
| `families_qwen4_exp_cuda_multi.py.patch` | 首 token 先于 draft 重排（TTFT）+ n-gram `_read_ahead` + 视觉 scratch 回收 + MTP_COPY 钩子 | ✅ 生产 |
| `families_qwen4_exp_cuda_forward.py.patch` | `read_ahead()`：为下一 prefill piece 预读 n-gram 表行 | ✅ 生产 |
| `families_qwen4_exp_cuda_decode.py.patch` | read_ahead 接线（单流路径）+ `logits=False` 调用点 | ✅ 生产 |

### 留了口子但 EXL3 生产不触发

| 补丁 | 修改 | 状态 |
|---|---|---|
| `families_qwen4_exp_cuda_ssd_read.{py,cpp}.patch` | 自研 C++ SSD n-gram 原生读取器（pread 线程池、释放 GIL；`TENSORFOLD_SSD_NATIVE=0` 回退） | dormant（表 locked in memory） |
| `families_qwen4_exp_host_table.py.patch` | `SSDTable` 包 `ReadAhead` | dormant |
| `cuda_geometry.py.patch` + `families_qwen4_exp_cuda_engine.py.patch` | `TENSORFOLD_PREFILL_ROWS`（256–16384）；**EXL3 被 `is_exl3()` 锁死为 None**（staging 实验已证伪，§1.5） | 仅非 EXL3 包可用 |

### 视频/多图支持（Mia 血统 0.6.1 backport，可用性未实测）

`vision_videos.py.patch`（新文件）、`vision_qwen_cuda.py.patch`、`vision_images{,_http}.py.patch`、`vision_qwen_processing.py.patch`、`server_messages.py.patch`、`server_prompts.py.patch`。
⚠️ 0.6.1 视频可用性**未实测**（EXL3-FIX §8.2），引用旧"不可用"结论前先跑 smoke。

### 服务补丁（另有可重放脚本 `../engine/patch_ttsilence.py`、`patch_tt_reserve.py`）

`cuda_server.py.patch`、`cuda_http.py.patch`、`cuda_scheduler.py.patch`、`engine_call_gate.py.patch` —— 与 `../engine/` 脚本等价，此处 diff 用于审计。明细见 EXL3-FIX §4。

## 上游升级时怎么继承

1. `git clone --depth 1 --branch <新版本> https://github.com/ashhart/TensorFold.git`
2. 逐个 `git apply --check`：能应用的直接应用；失败的手工移植（上游可能已吸收 P1–P4，diff 即答案）
3. `../draft-vocab/draft_vocab.zh-aug.102089.txt` 拷贝为 `families/qwen4_exp/cuda/draft_vocab.txt`
4. 回归：`~/exl3bench.py` + `~/tensorfold-patches/tf_{conv,think,vision_regress}_test.py`
