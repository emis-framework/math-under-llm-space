---
title: Math Under Llm
emoji: 🌖
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 6.14.0
python_version: '3.13'
app_file: app.py
pinned: false
license: apache-2.0
short_description: 'Compute SVD of LLM Q/K/V weights directly from Hugging Face '
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

---
# Wang's Five Laws — LLM Spectral Analyzer
## 完整项目文档 README.md

---


# 🔬 Wang's Five Laws — LLM Spectral Analyzer

**静态分析 LLM 注意力权重，无需推理，无需 benchmark，直接评估推理能力。**

通过对 Q/K/V 权重矩阵做 SVD 分解，验证王氏五定律，
计算 Wang Score（= 1 − median SSR_QK），实现跨模型推理能力排行。

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19707844-blue)](https://doi.org/10.5281/zenodo.19707844)
[![HAL](https://img.shields.io/badge/HAL-hal--05609398-red)](https://hal.science/hal-05609398)
[![Wang's Law](https://img.shields.io/badge/Wang%27s%20Law-r%3D1-blue)](https://github.com/emis-framework/math-under-llm)

---

## 目录

1. [项目背景](#1-项目背景)
2. [王氏五定律速查](#2-王氏五定律速查)
3. [整体架构](#3-整体架构)
4. [目录结构](#4-目录结构)
5. [各层详细说明](#5-各层详细说明)
   - 5.1 [core 层](#51-core-层——计算引擎)
   - 5.2 [db 层](#52-db-层——数据持久化)
   - 5.3 [ui 层](#53-ui-层——用户界面)
   - 5.4 [app.py 入口](#54-apppy——主入口)
6. [数据库表结构](#6-数据库表结构)
7. [数据流全链路](#7-数据流全链路)
8. [函数调用关系图](#8-函数调用关系图)
9. [关键设计决策](#9-关键设计决策)
10. [部署说明](#10-部署说明)
11. [依赖清单](#11-依赖清单)

---

## 1. 项目背景

传统评估 LLM 推理能力需要跑 benchmark（耗时、昂贵、可刷榜）。

本项目发现：**只看权重矩阵的奇异值分解（SVD）结构，
就能静态评估模型推理质量**，无需任何推理。

核心原理：
- 对每一层注意力的 Q、K、V 权重矩阵做 SVD
- 计算奇异值谱之间的相关性、形状残差、子空间对齐度
- 这些指标与模型推理能力高度相关（经多个模型验证）

**运行方式**：HTTP Range Request 直接读取 HuggingFace 远程权重，
无需下载整个模型（一个 14B 模型只需读取约 200MB 数据而非 28GB）。

---

## 2. 王氏五定律速查

| 定律     | 名称               | 公式                                | 理论极值   | 实测范围     |
| -------- | ------------------ | ----------------------------------- | ---------- | ------------ |
| 第一定律 | 谱线性对齐         | Pearson r(s_Q, s_K)                 | → 1        | 0.94~0.99    |
| 第二定律 | 谱形状残差         | SSR = mean\|ŝ_Q − ŝ_K\|             | → 0        | 0.006~0.016  |
| 第三定律 | 精度-深度约束      | L_max = min(L_info, L_quant, L_dyn) | 由精度决定 | FP16→16层    |
| 第四定律 | 输出子空间解耦     | cosU(U_Q,U_V) < 1/√d_head           | 超正交     | ~20%低于随机 |
| 第五定律 | 输入子空间随机正交 | cosV ≈ 1/√d_model                   | ≈随机基线  | 符合理论     |

**Wang Score = 1 − median(SSR_QK)**（越高越好，理论极值=1）

---

## 3. 整体架构

```
┌─────────────────────────────────────────────────────┐
│                      app.py                          │
│              主入口，组装所有 Tab                      │
│              启动时调用 init_db()                      │
└──────┬──────────────────────────────────────────────┘
       │ 调用
       ▼
┌──────────────────────────────────────────────────────┐
│                     ui/ 层                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │tab_inspect  │  │tab_analyze  │  │tab_leaderbd │  │
│  │结构探测      │  │分析+写库     │  │排行榜        │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │
│         │                │                │          │
│         └────────────────┼────────────────┘          │
│  ┌─────────────┐         │                           │
│  │tab_database │─────────┘                           │
│  │数据库浏览    │                                      │
│  └─────────────┘                                     │
└──────┬──────────────────────────┬───────────────────┘
       │ 调用                      │ 调用
       ▼                          ▼
┌─────────────────┐    ┌─────────────────────────────┐
│    core/ 层      │    │          db/ 层              │
│                 │    │                             │
│ fetcher.py      │    │  schema.py  writer.py       │
│ 远程读取权重     │    │  建表       写入数据          │
│                 │    │                             │
│ layer_profile.py│    │  reader.py                  │
│ 推断层结构       │    │  查询数据                    │
│                 │    │                             │
│ metrics.py      │    │  SQLite 文件                 │
│ 计算五定律       │    │  /data/wang_laws.db          │
└─────────────────┘    └─────────────────────────────┘
```

**三层职责：**

| 层      | 职责                        | 不做什么               |
| ------- | --------------------------- | ---------------------- |
| `core/` | 纯计算，无 UI，无 DB        | 不写数据库，不渲染界面 |
| `db/`   | 纯数据库操作                | 不做计算，不渲染界面   |
| `ui/`   | 纯界面逻辑，调用 core 和 db | 不做底层计算           |

---

## 4. 目录结构

```
项目根目录/
│
├── app.py                    # 主入口：初始化DB，组装4个Tab
├── requirements.txt          # 依赖清单
│
├── core/                     # 计算引擎（纯Python，无副作用）
│   ├── __init__.py           # 空文件
│   ├── config.py             # 全局开关（DEBUG=True/False）
│   ├── debug.py              # 调试输出工具（受config.DEBUG控制）
│   ├── fetcher.py            # HTTP Range Request 读取远程权重
│   ├── layer_profile.py      # 自动推断模型层结构
│   └── metrics.py            # 计算王氏五定律全部指标
│
├── db/                       # 数据持久化层
│   ├── __init__.py           # 空文件
│   ├── schema.py             # 建表SQL + 数据库连接
│   ├── writer.py             # 写入分析结果 + 断点续传检查
│   └── reader.py             # 查询排行榜、模型详情、原始数据
│
└── ui/                       # Gradio 界面层
    ├── __init__.py           # 空文件
    ├── tab_inspect.py        # Tab1：模型结构探测
    ├── tab_analyze.py        # Tab2：分析模型 + 写库
    ├── tab_leaderboard.py    # Tab3：王氏评分排行榜
    └── tab_database.py       # Tab4：数据库浏览
```

---

## 5. 各层详细说明

### 5.1 `core/` 层——计算引擎

#### `core/config.py`

```python
DEBUG = False   # True → 打印详细调试信息；False → 静默运行
```

全局唯一开关。所有调试输出都受这个控制。

---

#### `core/debug.py`

| 函数     | 签名                           | 用途                                   |
| -------- | ------------------------------ | -------------------------------------- |
| `dlog`   | `(lines: list[str], msg: str)` | 向日志列表追加调试信息（仅DEBUG=True） |
| `dprint` | `(msg: str)`                   | 打印到stdout（仅DEBUG=True）           |

`dlog` 用于 `metrics.py` 和 `tab_analyze.py`（有 lines 列表的地方）。
`dprint` 用于 `fetcher.py`（没有 lines 列表的地方）。

---

#### `core/fetcher.py`

**核心思想**：safetensors 文件头记录了每个 tensor 的字节偏移，
用 HTTP Range Request 只下载需要的字节，无需下载整个文件。

| 函数                      | 签名                                             | 返回                             | 用途                       |
| ------------------------- | ------------------------------------------------ | -------------------------------- | -------------------------- |
| `get_file_url`            | `(model_id, filename)`                           | `str`                            | 拼接 HF 下载 URL           |
| `read_safetensors_header` | `(url, token)`                                   | `(header_dict, header_size)`     | 读取文件头（两次HTTP请求） |
| `load_tensor_remote`      | `(url, tensor_name, header, header_size, token)` | `torch.Tensor`                   | 按名读取单个tensor         |
| `get_safetensor_files`    | `(model_id, token)`                              | `list[str]`                      | 列出所有.safetensors文件   |
| `find_index_file`         | `(model_id, token)`                              | `dict\|None`                     | 读取分片索引文件           |
| `get_all_shard_files`     | `(model_id, token)`                              | `list[str]`                      | 获取全部分片文件名         |
| `load_all_shard_headers`  | `(model_id, token)`                              | `dict[filename, (header, size)]` | 读取所有分片的header       |
| `check_quantization`      | `(model_id, token)`                              | `(is_blocked, message)`          | 三重量化检测               |
| `http_error_msg`          | `(e, model_id)`                                  | `str`                            | HTTP错误码转中文提示       |

**`load_all_shard_headers` 返回结构：**
```python
{
    "model-00001-of-00006.safetensors": (header_dict, header_size),
    "model-00002-of-00006.safetensors": (header_dict, header_size),
    ...
}
# header_dict 结构：
{
    "model.layers.0.self_attn.q_proj.weight": {
        "dtype": "BF16",
        "shape": [4096, 4096],
        "data_offsets": [0, 33554432]
    },
    ...
}
```

**量化检测四重逻辑（按顺序）：**
1. 检测 `config.json` 中的 `quantization_config` 字段
2. 检测模型名是否含 `gptq/awq/gguf` 关键词
3. 检测文件列表是否有 `.gguf` 文件
4. 检测 header 中是否有量化专用 key（如 `qweight`, `qzeros`）

---

#### `core/layer_profile.py`

**核心思想**：从权重文件的 key 名自动推断模型结构，零 hard coding，
不依赖模型名称或配置文件（配置文件只是辅助参考）。

**关键数据结构：**

```python
@dataclass
class QKVKey:
    shard: str    # 所在分片文件名，如 "model-00001-of-00006.safetensors"
    key:   str    # 完整tensor名，如 "model.layers.0.self_attn.q_proj.weight"
    shape: list   # tensor形状，如 [4096, 4096]

@dataclass
class LayerProfile:
    prefix:    str         # 组件前缀，如 "model.language_model."
    layer_idx: int         # 层号（原始safetensors key中的N）
    q:         QKVKey      # Q权重位置
    k:         QKVKey      # K权重位置
    v:         QKVKey|None # V权重位置（None表示K=V共享）
    head_dim:    int       # 每个head的维度
    n_q_heads:   int       # Q head数量
    n_kv_heads:  int       # KV head数量（GQA时 < n_q_heads）
    d_model:     int       # 模型隐层维度（= q_shape[1]）
    kv_shared:   bool      # True = K和V共享（如Gemma全局层）
    complete:    bool      # True = Q/K都存在且head_dim推断成功
    infer_ok:   bool       # head_dim推断是否成功
    head_dim_source: str   # 推断来源："k_norm"/"q_norm"/"config"/"enum"
```

| 函数                    | 签名                                 | 返回                                 | 用途                                                |
| ----------------------- | ------------------------------------ | ------------------------------------ | --------------------------------------------------- |
| `classify_qkv_suffix`   | `(suffix: str)`                      | `'q'/'k'/'v'/None`                   | 从key后缀判断是Q/K/V                                |
| `is_norm_key`           | `(suffix: str)`                      | `bool`                               | 判断是否为norm key（辅助推断head_dim）              |
| `scan_model_structure`  | `(all_shard_headers, config_params)` | `dict[(prefix,layer), LayerProfile]` | **核心函数**：扫描全部headers，构建LayerProfile字典 |
| `summarize_structure`   | `(profiles)`                         | `str`                                | 生成人类可读的结构报告（Tab1使用）                  |
| `extract_config_params` | `(config: dict)`                     | `dict`                               | 从config.json提取关键参数（兼容Gemma4嵌套结构）     |

**`scan_model_structure` 工作流程：**
```
第一遍扫描：遍历所有shard的所有key
  → 用正则 r'layers\.(\d+)\.' 提取层号
  → prefix = key的layers.N.之前部分
  → suffix = key的layers.N.之后部分
  → classify_qkv_suffix(suffix) → 归类为Q/K/V
  → is_norm_key(suffix) → 收集k_norm/q_norm形状（辅助推断head_dim）

第二遍构建：对每个(prefix, layer_idx)槽
  → 检查Q/K是否都存在（必要条件）
  → V不存在 → kv_shared=True
  → _infer_head_dim() → 推断head_dim（5个优先级）
  → 计算n_q_heads, n_kv_heads, d_model
  → 构建LayerProfile
```

**head_dim 推断优先级：**
```
1. k_norm.shape[0]        ← 最可靠（Gemma系列有这个）
2. q_norm.shape[0]        ← 备用
3. config["head_dim"]     ← config.json直接给出
4. config["hidden_size"] / config["num_attention_heads"]  ← 计算得出
5. 枚举候选值 [512,256,128,96,80,64,48,40,32,16]  ← 最后手段
```

**KV共享检测（Gemma全局层）：**
```
V的key不存在于任何shard header → kv_shared=True → layer_type="global"
```

---

#### `core/metrics.py`

对一层的Q/K/V权重矩阵计算所有指标。

**底层计算函数：**

| 函数          | 签名                   | 返回                | 对应定律               |
| ------------- | ---------------------- | ------------------- | ---------------------- |
| `pearson`     | `(a, b: Tensor)`       | `float`             | 第一定律               |
| `spearman_r`  | `(a, b: Tensor)`       | `float`             | 第一定律（补充）       |
| `ssr`         | `(a, b: Tensor)`       | `float`             | 第二定律               |
| `svr`         | `(a, b: Tensor)`       | `(alpha, residual)` | 尺度因子               |
| `cos_U`       | `(U_a, U_b: Tensor)`   | `float`             | 第四定律（左奇异向量） |
| `cos_V`       | `(Vt_a, Vt_b: Tensor)` | `float`             | 第五定律（右奇异向量） |
| `sigma_stats` | `(s: Tensor)`          | `(max, min, cond)`  | 第三定律               |

**`ssr` 计算细节：**
```python
# 归一化后逐元素绝对差的均值
n  = min(len(a), len(b))
an = a[:n] / ||a[:n]||    # L2归一化
bn = b[:n] / ||b[:n]||
SSR = mean(|an - bn|)
```

**主分析函数：**

```python
def analyze_layer(
    W_q: torch.Tensor,      # shape: [n_q_heads * head_dim, d_model]
    W_k: torch.Tensor,      # shape: [n_kv_heads * head_dim, d_model]
    W_v: torch.Tensor,      # shape: [n_kv_heads * head_dim, d_model]
    profile: LayerProfile,
) -> tuple[list[dict], str]:
    # 返回：(records列表, 格式化日志字符串)
```

**`analyze_layer` 工作流程：**
```
对每个 kv_head（0 ~ n_kv_heads-1）：
  切片：k_t = W_k[kv_h*d_head : (kv_h+1)*d_head, :]
  SVD：U_k, s_k, Vt_k = svd(k_t)
  计算 sigma_stats(s_k)

  对每个 q_head（属于这个kv_head的group，GQA时 group = n_q/n_kv）：
    切片：q_t = W_q[h*d_head : (h+1)*d_head, :]
    SVD：U_q, s_q, Vt_q = svd(q_t)

    计算所有指标：
      pearson_QK, spearman_QK, ssr_QK, alpha_QK  ← Q vs K奇异值
      pearson_QV, ssr_QV, alpha_QV                ← Q vs V奇异值
      ssr_KV, alpha_KV                            ← K vs V奇异值
      cosU_QK, cosU_QV, cosU_KV                  ← 左奇异向量
      cosV_QK, cosV_QV, cosV_KV                  ← 右奇异向量
      sigma_max/min/cond for Q, K, V

    append到records

特殊处理：kv_shared=True时，KV指标设为理论值（ssr=0, pearson=1, cosU=1等）
```

**`records` 每条记录的字段（共37个字段）：**
```python
{
    "prefix": str,       "layer": int,
    "kv_head": int,      "q_head": int,
    "kv_shared": bool,   "head_dim": int,
    "d_model": int,      "n_q_heads": int,    "n_kv_heads": int,
    # 第一定律
    "pearson_QK": float, "spearman_QK": float,
    "pearson_QV": float, "pearson_KV": float,
    # 第二定律
    "ssr_QK": float,     "ssr_QV": float,     "ssr_KV": float,
    # 第三定律
    "sigma_max_Q": float, "sigma_min_Q": float, "cond_Q": float,
    "sigma_max_K": float, "sigma_min_K": float, "cond_K": float,
    "sigma_max_V": float, "sigma_min_V": float, "cond_V": float,
    # 第四定律
    "cosU_QK": float,    "cosU_QV": float,    "cosU_KV": float,
    # 第五定律
    "cosV_QK": float,    "cosV_QV": float,    "cosV_KV": float,
    # 尺度因子
    "alpha_QK": float,   "alpha_QV": float,   "alpha_KV": float,
    "alpha_res_QK": float, "alpha_res_QV": float, "alpha_res_KV": float,
}
```

```python
def summarize_records(records: list[dict], model_id: str) -> str:
    # 对records做统计汇总，返回格式化文本
    # 按prefix分组，对每个指标计算 Median/Mean/Min/Max
    # KV指标自动排除kv_shared=True的行（避免理论值污染统计）
```

---

### 5.2 `db/` 层——数据持久化

#### `db/schema.py`

数据库路径逻辑：
```python
def get_db_path() -> str:
    if os.path.exists("/data"):   # HF Space bucket挂载点
        return "/data/wang_laws.db"
    return "wang_laws.db"         # 本地开发回退
```

| 函数             | 签名     | 用途                               |
| ---------------- | -------- | ---------------------------------- |
| `get_db_path`    | `()`     | 返回数据库文件路径                 |
| `get_connection` | `()`     | 返回SQLite连接（WAL模式，Row工厂） |
| `init_db`        | `()`     | 建表+建索引，幂等，返回连接        |
| `get_db_stats`   | `(conn)` | 返回各表行数+文件大小              |

---

#### `db/writer.py`

| 函数                   | 签名                                                | 用途                                             |
| ---------------------- | --------------------------------------------------- | ------------------------------------------------ |
| `infer_layer_type`     | `(kv_shared: bool)`                                 | `True→"global"`, `False→"standard"`              |
| `get_analyzed_layers`  | `(conn, model_id, prefix)`                          | 返回已完成的层号集合（断点续传用）               |
| `is_layer_complete`    | `(conn, model_id, prefix, layer, expected_records)` | 检查某层记录数是否达到预期                       |
| `upsert_model`         | `(conn, model_id, model_type, notes)`               | 写入/更新模型元数据                              |
| `upsert_component`     | `(conn, model_id, prefix, n_layers, ...)`           | 写入/更新组件信息                                |
| `write_layer_records`  | `(conn, model_id, records: list[dict])`             | 批量写入一层的逐头数据（INSERT OR REPLACE）      |
| `update_model_summary` | `(conn, model_id, prefix)`                          | 重算并写入model_summary的all/standard/global三行 |

**`update_model_summary` 逻辑：**
```
对 layer_type in ["all", "standard", "global"]：
  从 layer_head_metrics 查对应行
  计算各指标的 median/mean
  wang_score 统一用 standard 层的 median(ssr_QK) 计算
    （即使写 all/global 行，wang_score 也来自 standard 层）
  INSERT OR REPLACE 写入 model_summary
```

---

#### `db/reader.py`

| 函数                  | 签名                                                           | 返回           | 用途                         |
| --------------------- | -------------------------------------------------------------- | -------------- | ---------------------------- |
| `get_leaderboard`     | `(conn, prefix_filter, layer_type, limit)`                     | `pd.DataFrame` | 排行榜查询，按wang_score降序 |
| `get_model_summary`   | `(conn, model_id)`                                             | `pd.DataFrame` | 某模型所有组件的汇总统计     |
| `get_layer_metrics`   | `(conn, model_id, prefix, layer_type, start_layer, end_layer)` | `pd.DataFrame` | 逐头原始数据查询             |
| `get_analyzed_models` | `(conn)`                                                       | `pd.DataFrame` | 所有已分析模型列表           |
| `get_resume_status`   | `(conn, model_id, prefix)`                                     | `dict`         | 断点续传状态：已完成层号集合 |

---

### 5.3 `ui/` 层——用户界面

#### `ui/tab_inspect.py` — Tab1：结构探测

**函数：**

```python
def inspect_model(model_id, hf_token, progress) -> (str, pd.DataFrame):
    """
    工作流程：
    1. check_quantization()         ← 量化检测，失败则返回
    2. 读取 config.json             ← extract_config_params()
    3. load_all_shard_headers()     ← 读取所有分片header
    4. scan_model_structure()       ← 构建LayerProfile字典
    5. summarize_structure()        ← 生成文本报告
    6. 构建概览DataFrame            ← 每层一行
    返回：(日志文本, 层结构DataFrame)
    """

def build_tab_inspect() -> (inspect_model_id, inspect_token):
    """
    构建Tab1的Gradio组件
    返回：(model_id文本框, token文本框)
    ← 返回值供app.py做Tab1→Tab2的联动同步
    """
```

**UI组件：**
```
模型ID输入框 + Token输入框 + 探测按钮
→ 日志文本框（结构报告）
→ 层结构表格（prefix/layer/d_model/head_dim/n_q/n_kv/kv_shared等）
```

---

#### `ui/tab_analyze.py` — Tab2：分析（核心Tab）

**函数：**

```python
def run_analysis(model_id, hf_token, start_layer, end_layer, progress)
    -> (str, pd.DataFrame):
    """
    完整工作流程：

    [准备阶段]
    1.  init_db()                          ← 获取DB连接
    2.  check_quantization()               ← 量化检测
    3.  读取 config.json
    4.  upsert_model()                     ← 写模型元数据到DB
    5.  load_all_shard_headers()           ← 读所有分片header
    6.  scan_model_structure()             ← 构建LayerProfile字典
    7.  upsert_component() for each prefix ← 写组件信息到DB
    8.  按 start_layer~end_layer 过滤层

    [断点续传检查]
    9.  get_analyzed_layers() for each prefix
        → done_layers: dict[prefix, set[int]]
        → 打印待分析层和已跳过层

    [逐层分析循环]
    for each (prefix, layer_idx) in filtered（按prefix+layer排序）:
      10. 检查：layer_idx in done_layers[prefix] → continue（跳过）
      11. load_tensor_remote(Q)              ← HTTP Range Request
      12. load_tensor_remote(K)
      13. kv_shared ? W_v=W_k.clone() : load_tensor_remote(V)
      14. analyze_layer(W_q, W_k, W_v, prof) ← 计算五定律
      15. write_layer_records(conn, model_id, records) ← 写DB
      16. update_model_summary(conn, model_id, prefix) ← 更新排行榜
      17. del W_q, W_k, W_v                 ← 释放内存

    [收尾]
    18. 更新 models.analyze_sec（总耗时）
    19. summarize_records()                  ← 生成汇总文本
    返回：(日志文本, 逐头结果DataFrame)
    """

def build_tab_analyze() -> (model_id_input, token_input):
    """构建Tab2的Gradio组件，返回值供app.py联动"""
```

**UI组件：**
```
模型ID + Token + 起始层号 + 结束层号 + 分析按钮
侧边栏：推荐模型列表 + 层号说明
→ 分析日志文本框（逐头详情）
→ 逐头结果表格（37列全指标）
```

---

#### `ui/tab_leaderboard.py` — Tab3：排行榜

**函数：**

```python
def _format_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """
    格式化显示：
    - model_id → model_name（取最后一段）
    - wang_score → wang_score_pct（百分制字符串）
    - 数值列 → 6位小数字符串
    - 选择展示列（隐藏冗余列）
    """

def load_leaderboard(prefix_filter, layer_type) -> (pd.DataFrame, str):
    """
    调用 reader.get_leaderboard()
    prefix_filter 空字符串 → None（不过滤）
    layer_type="all" → 实际查 "standard"（排行榜默认用standard）
    """

def build_tab_leaderboard():
    """
    UI：组件过滤输入框 + 层类型下拉 + 刷新按钮
    → 状态文本 + 排行榜表格 + 指标说明
    用户手动点刷新（不自动加载）
    """
```

---

#### `ui/tab_database.py` — Tab4：数据库浏览

**函数：**

```python
def load_db_stats() -> str:
    """调用 get_db_stats()，返回各表行数+文件大小"""

def load_model_list() -> pd.DataFrame:
    """调用 get_analyzed_models()，返回模型列表"""

def load_model_detail(model_id) -> (pd.DataFrame, str):
    """
    调用 get_model_summary() → summary_df
    调用 get_resume_status() for each prefix → 断点续传状态文本
    """

def load_layer_data(model_id, prefix, layer_type, start_layer, end_layer)
    -> (pd.DataFrame, str):
    """调用 get_layer_metrics()，返回逐头原始数据"""

def build_tab_database():
    """
    UI分为4个区块：
    1. 数据库统计（行数+文件大小）
    2. 已分析模型列表
    3. 模型详情+断点续传状态
    4. 逐头原始数据查询（支持按prefix/layer_type/层号范围过滤）
    """
```

---

### 5.4 `app.py`——主入口

```python
# 启动时执行（模块级）
init_db()   # 建表，幂等

# Gradio Blocks
with gr.Blocks(...) as demo:
    # 标题 + 五定律表格 + DOI徽章

    with gr.Tabs():
        inspect_model_id, inspect_token = build_tab_inspect()
        analyze_model_id, analyze_token = build_tab_analyze()
        build_tab_leaderboard()
        build_tab_database()

    # Tab1 → Tab2 联动（避免重复输入）
    inspect_model_id.change(fn=lambda x:x,
        inputs=inspect_model_id, outputs=analyze_model_id)
    inspect_token.change(fn=lambda x:x,
        inputs=inspect_token, outputs=analyze_token)
```

---

## 6. 数据库表结构

共4张表，SQLite 存储于 `/data/wang_laws.db`。

### `models` — 模型基本信息
```sql
CREATE TABLE models (
    model_id      TEXT PRIMARY KEY,   -- "google/gemma-4-e2b"
    model_type    TEXT,               -- "gemma4" / "qwen2" 等
    analyzed_at   TIMESTAMP,          -- 最后分析时间
    analyze_sec   REAL,               -- 本次分析总耗时（秒）
    notes         TEXT                -- 备注
);
```

### `components` — 组件信息
```sql
CREATE TABLE components (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id      TEXT NOT NULL,
    prefix        TEXT NOT NULL,      -- "model.language_model."
    n_layers      INTEGER,            -- 该组件完整层数
    head_dim_min  INTEGER,            -- 最小head_dim（异构层存在时有意义）
    head_dim_max  INTEGER,            -- 最大head_dim
    has_kv_shared INTEGER DEFAULT 0,  -- 是否有K=V共享层
    has_global    INTEGER DEFAULT 0,  -- 是否有global层
    d_model       INTEGER,            -- 输入维度
    UNIQUE(model_id, prefix)
);
```

### `layer_head_metrics` — 逐头原始数据（主数据表）
```sql
CREATE TABLE layer_head_metrics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id      TEXT NOT NULL,
    prefix        TEXT NOT NULL,
    layer         INTEGER NOT NULL,
    layer_type    TEXT DEFAULT 'standard',  -- "standard" / "global"
    kv_head       INTEGER NOT NULL,
    q_head        INTEGER NOT NULL,
    kv_shared     INTEGER DEFAULT 0,   -- 1=K=V共享（理论值），0=正常
    head_dim      INTEGER,
    d_model       INTEGER,
    n_q_heads     INTEGER,
    n_kv_heads    INTEGER,
    -- 第一定律
    pearson_QK REAL, spearman_QK REAL, pearson_QV REAL, pearson_KV REAL,
    -- 第二定律
    ssr_QK REAL, ssr_QV REAL, ssr_KV REAL,
    -- 第三定律
    sigma_max_Q REAL, sigma_min_Q REAL, cond_Q REAL,
    sigma_max_K REAL, sigma_min_K REAL, cond_K REAL,
    sigma_max_V REAL, sigma_min_V REAL, cond_V REAL,
    -- 第四定律
    cosU_QK REAL, cosU_QV REAL, cosU_KV REAL,
    -- 第五定律
    cosV_QK REAL, cosV_QV REAL, cosV_KV REAL,
    -- 尺度因子
    alpha_QK REAL, alpha_res_QK REAL,
    alpha_QV REAL, alpha_res_QV REAL,
    alpha_KV REAL, alpha_res_KV REAL,

    UNIQUE(model_id, prefix, layer, kv_head, q_head)
);
```

### `model_summary` — 汇总统计（排行榜用）
```sql
CREATE TABLE model_summary (
    model_id          TEXT NOT NULL,
    prefix            TEXT NOT NULL,
    layer_type        TEXT NOT NULL DEFAULT 'all',  -- all/standard/global
    -- 第一定律
    median_pearson_QK REAL, mean_pearson_QK REAL,
    -- 第二定律
    median_ssr_QK REAL, mean_ssr_QK REAL,
    median_ssr_QV REAL, mean_ssr_QV REAL,
    -- 第三定律
    median_cond_Q REAL, mean_cond_Q REAL,
    -- 第四定律
    median_cosU_QK REAL, median_cosU_QV REAL,
    -- 第五定律
    median_cosV_QK REAL, median_cosV_QV REAL,
    -- 王氏评分（始终用standard层计算，即使layer_type=all/global）
    wang_score        REAL,
    n_layers          INTEGER,
    n_records         INTEGER,
    updated_at        TIMESTAMP,

    PRIMARY KEY(model_id, prefix, layer_type)
);
```

**每个(model_id, prefix)在model_summary中有3行：**
```
(model_id, prefix, "all")       ← 全部层混合统计
(model_id, prefix, "standard")  ← 只含standard层
(model_id, prefix, "global")    ← 只含global层（如Gemma全局层）
```

**layer_type 推断规则（零hard coding）：**
```
kv_shared=True  → layer_type="global"
kv_shared=False → layer_type="standard"
```

---

## 7. 数据流全链路

```
用户输入模型ID（如 "google/gemma-4-e2b"）
    │
    ▼
[Tab1 或 Tab2]
check_quantization()
    → 检测config.json / 模型名 / 文件列表 / header内容
    → 量化模型直接拒绝
    │
    ▼
load_all_shard_headers()
    → 对每个.safetensors文件：
        HTTP GET bytes=0-7          → header_size（8字节小端整数）
        HTTP GET bytes=8-{8+size}   → JSON header
    → 返回 {filename: (header_dict, header_size)}
    │
    ▼
scan_model_structure()
    → 两遍扫描所有key → 构建 {(prefix,layer): LayerProfile}
    → 自动推断：head_dim / n_q_heads / n_kv_heads / kv_shared
    │
    ▼（Tab2专有）
断点续传检查
    → get_analyzed_layers() → done_layers: dict[prefix, set[int]]
    │
    ▼（逐层循环）
load_tensor_remote(W_q)  → HTTP GET bytes={abs_start}-{abs_end}
load_tensor_remote(W_k)  → 同上
load_tensor_remote(W_v)  → 同上（kv_shared时直接clone W_k）
    │
    ▼
analyze_layer(W_q, W_k, W_v, profile)
    → 按head切片
    → SVD分解每个head
    → 计算37个指标
    → 返回 records: list[dict]
    │
    ▼
write_layer_records(conn, model_id, records)
    → INSERT OR REPLACE 批量写入 layer_head_metrics
    │
    ▼
update_model_summary(conn, model_id, prefix)
    → 查询 layer_head_metrics
    → 计算 median/mean
    → wang_score = 1 - median(ssr_QK) [用standard层]
    → INSERT OR REPLACE 写入 model_summary（all/standard/global 3行）
    │
    ▼
[Tab3 排行榜]
get_leaderboard()
    → SELECT from model_summary WHERE layer_type='standard'
    → ORDER BY wang_score DESC
    → 格式化展示
```

---

## 8. 函数调用关系图

```
app.py
├── init_db()                              [db/schema.py]
├── build_tab_inspect()                    [ui/tab_inspect.py]
│   └── inspect_model()
│       ├── check_quantization()           [core/fetcher.py]
│       ├── extract_config_params()        [core/layer_profile.py]
│       ├── load_all_shard_headers()       [core/fetcher.py]
│       │   ├── get_all_shard_files()
│       │   │   └── find_index_file()
│       │   └── read_safetensors_header()
│       ├── scan_model_structure()         [core/layer_profile.py]
│       │   ├── classify_qkv_suffix()
│       │   ├── is_norm_key()
│       │   └── _infer_head_dim()
│       └── summarize_structure()
│
├── build_tab_analyze()                    [ui/tab_analyze.py]
│   └── run_analysis()
│       ├── init_db()                      [db/schema.py]
│       ├── check_quantization()           [core/fetcher.py]
│       ├── extract_config_params()        [core/layer_profile.py]
│       ├── upsert_model()                 [db/writer.py]
│       ├── load_all_shard_headers()       [core/fetcher.py]
│       ├── scan_model_structure()         [core/layer_profile.py]
│       ├── upsert_component()             [db/writer.py]
│       ├── get_analyzed_layers()          [db/writer.py]
│       ├── load_tensor_remote() ×3        [core/fetcher.py]
│       ├── analyze_layer()                [core/metrics.py]
│       │   ├── pearson()
│       │   ├── spearman_r()
│       │   ├── ssr()
│       │   ├── svr()
│       │   ├── cos_U()
│       │   ├── cos_V()
│       │   └── sigma_stats()
│       ├── write_layer_records()          [db/writer.py]
│       │   └── infer_layer_type()
│       ├── update_model_summary()         [db/writer.py]
│       │   └── _calc_summary_row()
│       └── summarize_records()            [core/metrics.py]
│
├── build_tab_leaderboard()                [ui/tab_leaderboard.py]
│   └── load_leaderboard()
│       ├── init_db()                      [db/schema.py]
│       ├── get_leaderboard()              [db/reader.py]
│       └── _format_leaderboard()
│
└── build_tab_database()                   [ui/tab_database.py]
    ├── load_db_stats()
    │   └── get_db_stats()                 [db/schema.py]
    ├── load_model_list()
    │   └── get_analyzed_models()          [db/reader.py]
    ├── load_model_detail()
    │   ├── get_model_summary()            [db/reader.py]
    │   └── get_resume_status()            [db/reader.py]
    └── load_layer_data()
        └── get_layer_metrics()            [db/reader.py]
```

---

## 9. 关键设计决策

### 零 hard coding 原则
任何模型相关的参数（head_dim、层数、组件结构）
都从权重文件的 key 名自动推断，不写死任何模型名或层号。

### GQA 支持
当 `n_q_heads > n_kv_heads` 时（如 Llama-3-8B 的 32Q/8KV），
`group = n_q / n_kv`，每个KV head对应group个Q head，
全部独立计算，每个Q head一条记录。

### K=V 共享（Gemma全局层）
Gemma-4-31B 每6层有一个全局层，V权重不存在（K和V共享）。
检测方式：V的key不在任何shard的header中。
处理方式：`W_v = W_k.clone()`，KV相关指标设为理论值。
存储方式：`kv_shared=1`，`layer_type="global"`。

### 断点续传粒度
以 `(model_id, prefix, layer)` 为粒度。
某层的所有head全部写入才算完成。
允许随时中断，下次从未完成的层继续。

### 排行榜的 wang_score
无论 `model_summary` 的 `layer_type` 是 all/standard/global，
`wang_score` 统一从 standard 层的 `ssr_QK` 计算，
避免全局层（K=V共享，SSR=0）人为拉高评分。

### 每个(model_id, prefix)在排行榜中是一行
排行榜以 `(model_id, prefix)` 为单位，
多模态模型（如Gemma-4）的language_model和vision_tower分别占一行。

---

## 10. 部署说明

### HuggingFace Space 部署

1. 创建 Space，选择 Gradio SDK
2. 在 Space Settings 中添加 **Persistent Storage**（挂载到 `/data`）
   - `wang_laws.db` 重启后不丢失
3. 上传所有文件（保持目录结构）
4. 如需访问私有模型，在 Space Secrets 中设置 `HF_TOKEN`

#### 配置管理员写入权限（重要）

在 **Space Settings → Secrets** 中添加：

| Secret 名称   | 值               | 说明                            |
| ------------- | ---------------- | ------------------------------- |
| `WRITE_TOKEN` | 你自己设置的密码 | 管理员写库密钥，不进入 git repo |

**工作原理：**
```
HF Space Secrets（加密存储，不在 git 中）
    ↓ HF 运行时自动注入
Docker 容器环境变量 WRITE_TOKEN
    ↓ 服务端读取
os.environ.get("WRITE_TOKEN")
    ↓ 与用户输入的 Admin Token 比对（纯服务端，前端不可见）
True  → 写入数据库
False → 只读模式，分析正常运行
```

**三类用户的体验：**

| 用户            | Admin Write Token | 行为                               |
| --------------- | ----------------- | ---------------------------------- |
| 你（管理员）    | 填写正确密钥      | 分析结果写入数据库，排行榜更新     |
| 审稿人 / 复现者 | 留空              | 分析正常运行，指标完整显示，不写库 |
| 恶意用户        | 随意填写          | 分析可以跑，写库被拒绝             |

**未配置 `WRITE_TOKEN` 时：**
```python
# check_write_permission() 的行为：
server_token = os.environ.get("WRITE_TOKEN", "")
if not server_token:
    return False   # 服务端未配置 → 拒绝所有写入
```
即使有人猜到任意字符串也无法写入。

### 本地运行

```bash
pip install -r requirements.txt

# 可选：设置写入权限
export WRITE_TOKEN="your_secret_password"

python app.py
# 浏览器打开 http://127.0.0.1:7860
```

本地运行时数据库存于当前目录的 `wang_laws.db`。
不设置 `WRITE_TOKEN` 则所有人都是只读模式。
```

---

## 改动汇总

| 文件                    | 改动                                                                         |
| ----------------------- | ---------------------------------------------------------------------------- |
| `db/writer.py`          | 末尾追加 `check_write_permission()`，其余不变                                |
| `ui/tab_analyze.py`     | 完整重写：加 `admin_token` 参数，所有写库操作加 `can_write` 判断，日志改英文 |
| `README.md`             | 第10节部署说明扩充写权限配置说明                                             |
| `db/schema.py`          | 不变                                                                         |
| `db/reader.py`          | 不变                                                                         |
| `ui/tab_inspect.py`     | 不变                                                                         |
| `ui/tab_leaderboard.py` | 不变                                                                         |
| `ui/tab_database.py`    | 不变                                                                         |
| `app.py`                | 不变                                                                         |


### 注意事项

- 分析大模型（如 70B）时每层需要约 30 秒（受 HF CDN 网速限制）
- HF Space 免费版有 48 小时超时限制，建议开启断点续传分批分析
- 量化模型（GPTQ/AWQ/GGUF）自动拒绝，需使用原始 BF16 版本

---

## 11. 依赖清单

```
gradio>=4.0.0     # Web UI 框架
requests          # HTTP Range Request 读取远程权重
numpy             # 数值计算（统计汇总）
scipy             # spearman相关系数
torch             # SVD分解（torch.linalg.svd）
huggingface_hub   # list_repo_files（文件列表）
```

Python 内置（无需安装）：
```
sqlite3    # 数据库
struct     # 解析safetensors header的8字节整数
json       # 解析safetensors header JSON
re         # 正则提取层号
datetime   # 时间戳
dataclasses # LayerProfile数据结构
```
