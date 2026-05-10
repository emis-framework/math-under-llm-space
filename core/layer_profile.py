# core/layer_profile.py
"""
从 safetensors headers 自动推断每一层的结构：
- head_dim（优先 k_norm/q_norm shape，其次 config，最后枚举）
- K=V 共享检测（v_key 是否存在）
- 组件前缀自动分离
- 零 hard coding
"""

import re
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# QKV 后缀分类
# ─────────────────────────────────────────────

# 精确排除列表（不是 Q/K/V 主权重）
_EXCLUDE_PATTERNS = [
    "norm",        # layernorm, k_norm, q_norm 等
    "rope",        # rotary embedding
    "lm_head",
    "o_proj",      # 输出投影
    "out_proj",
    "post",        # audio tower 的 post linear
    "relative",    # audio tower relative_k_proj
    "per_dim",     # audio tower per_dim_scale
    "scalar",
    "gate_proj",   # FFN
    "up_proj",
    "down_proj",
    "ffw_layer",   # audio FFN
    "depthwise",
    "conv",
    "linear_start",
    "linear_end",
    "per_layer",
    "embed",
    "input_max",   # audio 量化统计量
    "input_min",
    "output_max",
    "output_min",
]

_Q_PATTERNS = ["q_proj", "wq", "query", "q_a", "q_b"]
_K_PATTERNS = ["k_proj", "wk", "key",   "k_a", "k_b"]
_V_PATTERNS = ["v_proj", "wv", "value", "v_a", "v_b"]

# k_norm / q_norm：用于推断 head_dim，不是 QKV
_NORM_KEYS = ["k_norm", "q_norm"]


def classify_qkv_suffix(suffix: str) -> str | None:
    """
    layers.{N}. 之后的后缀 → 'q' / 'k' / 'v' / None

    支持：
      标准:   self_attn.q_proj.weight
      嵌套:   self_attn.q_proj.linear.weight  (audio/vision tower)
    """
    if not suffix.endswith(".weight"):
        return None

    s = suffix.lower()

    # 排除非 QKV
    if any(e in s for e in _EXCLUDE_PATTERNS):
        return None

    if any(p in s for p in _Q_PATTERNS):
        return "q"
    if any(p in s for p in _K_PATTERNS):
        return "k"
    if any(p in s for p in _V_PATTERNS):
        return "v"
    return None


def is_norm_key(suffix: str) -> bool:
    """判断是否为 norm key（用于推断 head_dim）"""
    s = suffix.lower()
    return any(n in s for n in _NORM_KEYS) and suffix.endswith(".weight")


# ─────────────────────────────────────────────
# LayerProfile 数据结构
# ─────────────────────────────────────────────

@dataclass
class QKVKey:
    """单个 Q/K/V weight 的位置信息"""
    shard:    str    # 所在 shard 文件名
    key:      str    # 完整 key 名
    shape:    list   # weight shape


@dataclass
class LayerProfile:
    """
    一个 (prefix, layer_idx) 槽的完整结构信息
    所有字段均从权重文件自动推断，零 hard coding
    """
    prefix:    str
    layer_idx: int

    # QKV 位置
    q:         QKVKey | None = None
    k:         QKVKey | None = None
    v:         QKVKey | None = None   # None = K=V 共享

    # 自动推断的维度
    head_dim:    int = 0
    n_q_heads:   int = 0
    n_kv_heads:  int = 0
    d_model:     int = 0   # = q_shape[1]

    # 标志
    kv_shared:   bool = False   # V 是否复用 K
    complete:    bool = False   # Q/K 都存在才算 complete
    infer_ok:    bool = False   # head_dim 推断成功

    # 推断来源（调试用）
    head_dim_source: str = ""   # "k_norm" / "q_norm" / "config" / "enum"

    # 原始 norm shape（用于推断 head_dim）
    k_norm_shape: list = field(default_factory=list)
    q_norm_shape: list = field(default_factory=list)

    def summary(self) -> str:
        kv_tag = "[K=V共享]" if self.kv_shared else ""
        return (
            f"Layer {self.layer_idx:3d} | "
            f"d_model={self.d_model:5d} | "
            f"head_dim={self.head_dim:4d}({self.head_dim_source}) | "
            f"n_q={self.n_q_heads:3d} n_kv={self.n_kv_heads:3d} | "
            f"{kv_tag}"
        )


# ─────────────────────────────────────────────
# 核心：自动推断 head_dim
# ─────────────────────────────────────────────

def _infer_head_dim(
    q_shape:      list,
    k_shape:      list,
    k_norm_shape: list,
    q_norm_shape: list,
    config_params: dict,
) -> tuple[int, str]:
    """
    推断 head_dim，返回 (head_dim, source)

    优先级：
    1. k_norm.shape[0]  → 最可靠（Gemma 系列）
    2. q_norm.shape[0]  → 备用
    3. config head_dim
    4. config hidden_size / num_attention_heads
    5. 枚举候选值
    """
    q_rows = q_shape[0] if q_shape else 0
    k_rows = k_shape[0] if k_shape else 0

    # 1. k_norm
    if k_norm_shape and len(k_norm_shape) == 1:
        d = k_norm_shape[0]
        if d > 0 and (q_rows == 0 or q_rows % d == 0):
            return d, "k_norm"

    # 2. q_norm
    if q_norm_shape and len(q_norm_shape) == 1:
        d = q_norm_shape[0]
        if d > 0 and (q_rows == 0 or q_rows % d == 0):
            return d, "q_norm"

    # 3. config head_dim
    if config_params:
        d = config_params.get("head_dim")
        if d and q_rows % d == 0 and k_rows % d == 0:
            return d, "config"

        # 4. config hidden_size / num_heads
        hs = config_params.get("hidden_size") or 0
        nh = config_params.get("num_attention_heads") or 0
        if hs and nh:
            d = hs // nh
            if d > 0 and q_rows % d == 0 and k_rows % d == 0:
                return d, "config_calc"

    # 5. 枚举
    for d in [512, 256, 128, 96, 80, 64, 48, 40, 32, 16]:
        if q_rows % d == 0 and k_rows % d == 0:
            return d, "enum"

    return 0, "failed"


# ─────────────────────────────────────────────
# 主扫描函数
# ─────────────────────────────────────────────

def scan_model_structure(
    all_shard_headers: dict[str, tuple[dict, int]],
    config_params:     dict = None,
) -> dict[tuple[str, int], LayerProfile]:
    """
    扫描所有 shard headers，构建完整的 LayerProfile 字典。

    返回：
    {
        (prefix, layer_idx): LayerProfile,
        ...
    }

    特性：
    - 零 hard coding
    - 自动检测 K=V 共享
    - 自动推断 head_dim
    - 不同组件的同编号层完全独立
    """
    config_params = config_params or {}

    # ── 第一遍：收集所有原始信息 ─────────────────
    # slot → { "q/k/v/k_norm/q_norm": QKVKey }
    raw: dict[tuple[str, int], dict] = {}

    for shard_name, (header, _) in all_shard_headers.items():
        for key, info in header.items():
            m = re.search(r'layers\.(\d+)\.', key)
            if not m:
                continue

            layer_idx = int(m.group(1))
            prefix    = key[:m.start()]    # 精确截断
            suffix    = key[m.end():]

            slot = (prefix, layer_idx)
            if slot not in raw:
                raw[slot] = {}

            shape = info.get("shape", [])

            # 分类
            role = classify_qkv_suffix(suffix)
            if role and role not in raw[slot]:
                raw[slot][role] = QKVKey(
                    shard=shard_name,
                    key=key,
                    shape=shape
                )
                continue

            # 收集 norm shape（用于 head_dim 推断）
            if is_norm_key(suffix):
                s = suffix.lower()
                if "k_norm" in s and "k_norm_shape" not in raw[slot]:
                    raw[slot]["k_norm_shape"] = shape
                elif "q_norm" in s and "q_norm_shape" not in raw[slot]:
                    raw[slot]["q_norm_shape"] = shape

    # ── 第二遍：构建 LayerProfile ─────────────────
    profiles: dict[tuple[str, int], LayerProfile] = {}

    for slot, data in raw.items():
        prefix, layer_idx = slot

        q = data.get("q")
        k = data.get("k")
        v = data.get("v")

        # Q/K 必须存在才有意义
        if q is None or k is None:
            continue

        # K=V 共享检测：v_key 不存在
        kv_shared = (v is None)

        k_norm_shape = data.get("k_norm_shape", [])
        q_norm_shape = data.get("q_norm_shape", [])

        # 推断 head_dim
        head_dim, source = _infer_head_dim(
            q_shape      = q.shape,
            k_shape      = k.shape,
            k_norm_shape = k_norm_shape,
            q_norm_shape = q_norm_shape,
            config_params= config_params,
        )

        infer_ok  = head_dim > 0
        n_q_heads = q.shape[0] // head_dim if infer_ok and q.shape else 0
        n_kv_heads= k.shape[0] // head_dim if infer_ok and k.shape else 0
        d_model   = q.shape[1] if q.shape and len(q.shape) > 1 else 0

        # 验证整除性
        if infer_ok and q.shape and q.shape[0] % head_dim != 0:
            infer_ok = False

        profiles[slot] = LayerProfile(
            prefix       = prefix,
            layer_idx    = layer_idx,
            q            = q,
            k            = k,
            v            = v,
            head_dim     = head_dim,
            n_q_heads    = n_q_heads,
            n_kv_heads   = n_kv_heads,
            d_model      = d_model,
            kv_shared    = kv_shared,
            complete     = infer_ok and n_q_heads > 0 and n_kv_heads > 0,
            infer_ok     = infer_ok,
            head_dim_source = source,
            k_norm_shape = k_norm_shape,
            q_norm_shape = q_norm_shape,
        )

    return profiles


# ─────────────────────────────────────────────
# 结构概览（供 Tab1 展示）
# ─────────────────────────────────────────────

def summarize_structure(
    profiles: dict[tuple[str, int], LayerProfile]
) -> str:
    """生成人类可读的结构概览文本"""
    if not profiles:
        return "⚠️ 未发现任何有效层\n"

    # 按 prefix 分组
    by_prefix: dict[str, list[LayerProfile]] = {}
    for (prefix, _), prof in profiles.items():
        by_prefix.setdefault(prefix, []).append(prof)

    lines = []
    for prefix in sorted(by_prefix):
        profs = sorted(by_prefix[prefix], key=lambda p: p.layer_idx)
        layer_idxs = [p.layer_idx for p in profs]
        complete   = [p for p in profs if p.complete]
        kv_shared  = [p for p in profs if p.kv_shared]

        # 检测异构 head_dim
        head_dims = sorted(set(p.head_dim for p in complete))

        lines.append(f"\n{'─'*70}")
        lines.append(f"组件：'{prefix}'")
        lines.append(
            f"  层数：{len(profs)}  "
            f"范围：{layer_idxs[0]}~{layer_idxs[-1]}  "
            f"完整层：{len(complete)}"
        )
        lines.append(f"  head_dim：{head_dims}")

        if kv_shared:
            lines.append(
                f"  K=V共享层：{[p.layer_idx for p in kv_shared]}"
            )

        # 异构层详情
        if len(head_dims) > 1:
            lines.append("  ⚠️  异构 head_dim 检测到：")
            for d in head_dims:
                idxs = [p.layer_idx for p in complete if p.head_dim == d]
                lines.append(f"    head_dim={d:4d} → 层 {idxs}")

        # 每层一行简要信息
        lines.append("")
        for p in profs:
            if p.complete:
                lines.append(f"    {p.summary()}")
            else:
                lines.append(
                    f"    Layer {p.layer_idx:3d} | "
                    f"⚠️ 不完整 "
                    f"(head_dim推断:{p.head_dim_source})"
                )

    lines.append(f"\n{'─'*70}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# config 解析（兼容 Gemma4 text_config）
# ─────────────────────────────────────────────

def extract_config_params(config: dict) -> dict:
    """
    兼容不同模型的 config.json 字段：
    - 标准：顶层字段
    - Gemma4：text_config 子字段
    """
    if not config:
        return {}

    text_cfg = config.get("text_config", {}) or {}

    def get(*keys):
        for k in keys:
            v = config.get(k)
            if v is not None:
                return v
            v = text_cfg.get(k)
            if v is not None:
                return v
        return None

    return {
        "model_type":          get("model_type"),
        "hidden_size":         get("hidden_size"),
        "num_attention_heads": get("num_attention_heads"),
        "num_key_value_heads": get("num_key_value_heads"),
        "head_dim":            get("head_dim"),
    }