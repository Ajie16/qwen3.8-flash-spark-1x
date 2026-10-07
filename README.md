# Qwen3.8-Flash-Next on one DGX Spark — 两条推理路线

单台 NVIDIA DGX Spark（GB10 / SM121 / 128 GB 统一内存 / aarch64）上跑
Qwen3.8-Flash-Next（176B MoE）的两套完整方案。**两条路线互不干扰，按目录隔离。**

| 路线 | 引擎 | 权重 | 冷 prefill @32k | 目录 |
|---|---|---|---|---|
| **A · 原生 EXL3** | exllamav3 1.5.1 fork + TabbyAPI | EXL3 3.05bpw / head5 | **~1,130 tok/s** | [`exllamav3-tabby/`](exllamav3-tabby/) |
| **B · TensorFold** | TensorFold 0.6.1 | MLX 4-bit affine g32 | **2,425 tok/s** | [`tensorfold-mlx/`](tensorfold-mlx/) |

> 数字是各自实测，**不同模型/不同会话不可直接比**（见下方"为什么快慢不同"）。

---

## 怎么选

**要抗审查 + 生态成熟 → 路线 A。**
EXL3 包有多个抗审查微调可选，exllamav3 是原生实现、TabbyAPI 提供 OpenAI 兼容接口，
`cache_size` 是共享 KV 池、并发行为可预测。

**要 prefill 速度 → 路线 B，但只能配审查版权重。**
TensorFold 的 `qwen4_exp` CUDA family **只接受** `bits=4, group_size=32, mode=affine`。
MLX 社区默认分组是 64，**抗审查的 MLX 包清一色 g64，会被直接拒绝加载**。
目前 g32 的只有官方（审查）包，抗审查要自己从 BF16 源转。

---

## 两条路线的关键差异

| | A · 原生 EXL3 | B · TensorFold |
|---|---|---|
| 权重格式 | EXL3 trellis | MLX 4-bit affine g32 |
| **抗审查可用** | ✅ 多份可选 | ❌ 需自行转换 |
| prefill @32k | ~1,130 tok/s | **2,425 tok/s** |
| prefill @75k | — | 2,105 tok/s |
| decode（中文） | 42–49 tok/s | 40–46 tok/s |
| MTP 接受率 | — | 53–59% |
| KV 池 | 1,572,864 token（共享池） | 32.5 GiB（按流分配） |
| n-gram 表 | `ngram_ram: true` 放内存 | **留在 NVMe**（MLX 专属 `--ple-on-ssd`） |
| 上下文上限 | 262,144 | 262,144 |
| 视觉 | TabbyAPI `vision: true` | 包内自带塔（EXL3 包需外挂 BF16 塔） |

**为什么快慢不同**：B 的 prefill 快 2 倍是**量化格式**的差别，不是服务端的差别。
TensorFold 的 EXL3 路径（同一引擎、同一张卡）实测只有 805 tok/s——
它没有为 EXL3 写 prefill 专家内核，而 MLX 4-bit 路径有。

---

## 仓库布局

```
.
├── exllamav3-tabby/          # 路线 A：原生 EXL3 引擎（配方主体）
│   ├── setup.sh              #   从源码编译 sm_121 的 exllamav3
│   ├── serve.sh              #   渲染 config 并启动 TabbyAPI
│   ├── tabby-config.yml      #   TabbyAPI 配置模板
│   └── tuning/ bench/ tools/ #   调参与基准
├── vllm-plugin/              # 路线 A 的 vLLM 变体（本机更慢，独立目录）
├── tensorfold-mlx/           # 路线 B：TensorFold + MLX 4-bit
│   ├── README.md             #   含全部实测数据与踩坑记录
│   ├── launchers/            #   启动脚本（int8 / bf16 / 尾部模板实验）
│   ├── patches/              #   引擎补丁，分 engine / experiments / upstream
│   └── draft-vocab/          #   中文草稿词表 + 上游对照
├── bench/                    # 两条路线共用的基准脚本
└── docs/                     # 基准数据与渲染页
```

**隔离约定**：
- 路线 A 的东西只在 `exllamav3-tabby/` 和 `vllm-plugin/`
- 路线 B 的东西只在 `tensorfold-mlx/`
- 两边**不共享任何代码或配置**，可以独立部署、独立回滚

---

## 引擎版本与依赖

**路线 A** 依赖两个仓库的特定提交，**不在本仓内**：

| 组件 | 来源 | 说明 |
|---|---|---|
| exllamav3 | `vcruz305/exllamav3` fork | GB10 统一内存补丁（`fix/gb10-uma-and-draft-window`）<br>另有 `feat/hybrid-draft`、`feat/spec-sampling` 两个分支 |
| TabbyAPI | `theroyallab/tabbyAPI` | 加一个本地提交 `b8c0497 local: allow max_history bump under EXL3_HYBRID_NGRAM` |

exllamav3 的三个分支已推到本仓的 `exllamav3/*` 命名空间，保留完整提交历史：

| 分支 | 说明 |
|---|---|
| `exllamav3/master` | GB10 UMA 预算 + draft window 修复（上游已推送） |
| `exllamav3/feat-hybrid-draft` | `EXL3_GR_TUNED` 融合 GR kernel、hybrid ngram 自适应退避 |
| `exllamav3/feat-spec-sampling` | 精确推测采样验证路径 |

**路线 B** 依赖 `ashhart/TensorFold` v0.6.1，补丁在本仓
[`tensorfold-mlx/patches/`](tensorfold-mlx/patches/) 内。

---

## 快速开始

### 路线 A（原生 EXL3）

```bash
cd exllamav3-tabby
./setup.sh            # 首次：编译 exllamav3（sm_121）
./serve.sh            # 渲染 config 到 ../state/config.yml 并启动 TabbyAPI
```

模型放在 `state/models/`，`tabby-config.yml` 里的 `model_name` 指向它。
详见 [`exllamav3-tabby/README.md`](exllamav3-tabby/README.md)。

### 路线 B（TensorFold）

```bash
cd tensorfold-mlx
# 1. 建引擎
python3 -m venv ~/tensorfold-venv
~/tensorfold-venv/bin/pip install --upgrade --no-deps \
  "git+https://github.com/ashhart/TensorFold.git@v0.6.1"
# 2. 打补丁（可选，说明见 tensorfold-mlx/README.md 的补丁一节）
~/tensorfold-venv/bin/python patches/engine/patch_ttsilence.py
# 3. 换中文词表（可选）
cp draft-vocab/draft_vocab.zh-aug.102089.txt \
   ~/tensorfold-venv/lib/python3.12/site-packages/tensorfold/families/qwen4_exp/cuda/draft_vocab.txt
# 4. 启动
bash launchers/tf-serve-mlx-int8.sh
```

详见 [`tensorfold-mlx/README.md`](tensorfold-mlx/README.md)。

---

## 硬件前提

- DGX Spark / GB10（SM121，121.69 GiB 可用统一内存）
- exllamav3 需从源码编译（`setup.sh` 处理 sm_121 的 arch flag）
- 统一内存耗尽会**冻结整机**而不是报错，务必留够系统余量

## License

见 [LICENSE](LICENSE)。
