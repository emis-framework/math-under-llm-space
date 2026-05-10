import gradio as gr
import requests
import struct
import json
import re
import numpy as np
import torch
from scipy.stats import spearmanr
from huggingface_hub import list_repo_files
import pandas as pd

# ─────────────────────────────────────────────
# dtype 映射
# ─────────────────────────────────────────────

DTYPE_MAP = {
    "F32":  (torch.float32,  4),
    "F16":  (torch.float16,  2),
    "BF16": (torch.bfloat16, 2),
    "F64":  (torch.float64,  8),
    "I32":  (torch.int32,    4),
    "I64":  (torch.int64,    8),
    "I8":   (torch.int8,     1),
    "U8":   (torch.uint8,    1),
}
try:
    DTYPE_MAP["F8_E4M3"] = (torch.float8_e4m3fn, 1)
    DTYPE_MAP["F8_E5M2"] = (torch.float8_e5m2,   1)
except AttributeError:
    pass

UNSUPPORTED_SVD_DTYPES = {"I8", "U8", "I32", "I64", "F8_E4M3", "F8_E5M2"}
QUANTIZED_KEY_SIGNATURES = ["qweight", "qzeros", "scales", "g_idx", "packed_weight"]

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def get_file_url(model_id: str, filename: str) -> str:
    return f"https://huggingface.co/{model_id}/resolve/main/{filename}"


def read_safetensors_header(url: str, token: str = None) -> tuple[dict, int]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(url, headers={**headers, "Range": "bytes=0-7"}, timeout=30)
    r.raise_for_status()
    header_size = struct.unpack("<Q", r.content)[0]
    r = requests.get(
        url,
        headers={**headers, "Range": f"bytes=8-{8 + header_size - 1}"},
        timeout=30
    )
    r.raise_for_status()
    raw = json.loads(r.content)
    raw.pop("__metadata__", None)
    return raw, header_size


def load_tensor_remote(
    url: str, tensor_name: str,
    header: dict, header_size: int,
    token: str = None
) -> torch.Tensor | None:
    if tensor_name not in header:
        return None
    info      = header[tensor_name]
    dtype_str = info["dtype"]
    shape     = info["shape"]
    offsets   = info["data_offsets"]

    if dtype_str not in DTYPE_MAP:
        raise ValueError(f"未知 dtype: {dtype_str}")
    if dtype_str in UNSUPPORTED_SVD_DTYPES:
        raise ValueError(f"dtype={dtype_str} 为量化格式，无法 SVD")

    torch_dtype, _ = DTYPE_MAP[dtype_str]
    abs_start = 8 + header_size + offsets[0]
    abs_end   = 8 + header_size + offsets[1] - 1

    req_headers = {"Range": f"bytes={abs_start}-{abs_end}"}
    if token:
        req_headers["Authorization"] = f"Bearer {token}"

    r = requests.get(url, headers=req_headers, timeout=120)
    r.raise_for_status()
    raw_bytes = r.content

    if torch_dtype == torch.bfloat16:
        tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.int16).view(torch.bfloat16)
    else:
        tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch_dtype)

    return tensor.reshape(shape).float()


def get_safetensor_files(model_id: str, token: str = None) -> list:
    kwargs = {"token": token} if token else {}
    return sorted(
        f for f in list_repo_files(model_id, **kwargs)
        if f.endswith(".safetensors")
    )


def find_index_file(model_id: str, token: str = None) -> dict | None:
    url = f"https://huggingface.co/{model_id}/resolve/main/model.safetensors.index.json"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(url, headers=headers, timeout=15)
    return r.json() if r.status_code == 200 else None


def _http_error_msg(e: requests.exceptions.HTTPError, model_id: str) -> str:
    code = e.response.status_code
    if code == 401: return "❌ 401 未授权"
    if code == 403: return f"❌ 403 禁止访问：请先接受 {model_id} 的使用协议"
    if code == 404: return f"❌ 404 未找到：{model_id}"
    return f"❌ HTTP {code}：{e}"


def extract_config_params(config: dict) -> dict:
    if config is None:
        return {}
    text_cfg = config.get("text_config", {}) or {}

    def get_field(*keys):
        for k in keys:
            v = config.get(k)
            if v is not None: return v
            v = text_cfg.get(k)
            if v is not None: return v
        return None

    return {
        "hidden_size":         get_field("hidden_size"),
        "num_attention_heads": get_field("num_attention_heads"),
        "num_key_value_heads": get_field("num_key_value_heads"),
        "head_dim":            get_field("head_dim"),
        "model_type":          get_field("model_type"),
    }


# ─────────────────────────────────────────────
# QKV 后缀分类
# ─────────────────────────────────────────────

def _classify_qkv_suffix(suffix: str) -> str | None:
    """
    layers.{N}. 之后的后缀 → 'q'/'k'/'v'/None
    
    支持格式：
      标准:   self_attn.q_proj.weight
      嵌套:   self_attn.q_proj.linear.weight  (audio/vision tower)
    
    Gemma-4 实测后缀：
      audio:   self_attn.q_proj.linear.weight   [1024, 1024]
      audio:   self_attn.k_proj.linear.weight   [1024, 1024]
      audio:   self_attn.v_proj.linear.weight   [1024, 1024]
      vision:  self_attn.q_proj.linear.weight   [768, 768]
      vision:  self_attn.k_proj.linear.weight   [768, 768]
      vision:  self_attn.v_proj.linear.weight   [768, 768]
      text:    self_attn.q_proj.weight           [2048, 1536]
      text:    self_attn.k_proj.weight           [256, 1536]
      text:    self_attn.v_proj.weight           [256, 1536]
    """
    if not suffix.endswith(".weight"):
        return None

    s = suffix.lower()

    # 精确排除非QKV权重
    excludes = [
        "norm", "rope", "embed", "lm_head", "layernorm", "ln_",
        "o_proj", "out_proj",           # 输出投影
        "post", "relative",             # audio tower 特有
        "per_dim", "scalar",            # audio tower 特有
        "gate_proj", "up_proj", "down_proj",  # FFN
        "ffw_layer",                    # audio FFN
        "depthwise", "conv",            # audio conv
        "linear_start", "linear_end",   # audio conv
        "per_layer",                    # language model 特有
    ]
    if any(e in s for e in excludes):
        return None

    # Q/K/V 匹配
    if any(p in s for p in ["q_proj", "wq", "query", "q_a", "q_b"]):
        return "q"
    if any(p in s for p in ["k_proj", "wk",           "k_a", "k_b"]):
        # 排除 k_norm（已在上面 norm 过滤，但双重保险）
        if "k_norm" in s:
            return None
        return "k"
    if any(p in s for p in ["v_proj", "wv", "value",  "v_a", "v_b"]):
        return "v"
    return None


# ─────────────────────────────────────────────
# ★ 核心：按原始层号扫描，不合并不重排
# 返回结构：
# {
#   (prefix, layer_idx): {
#       "q": (shard, key),
#       "k": (shard, key),
#       "v": (shard, key),
#   }
# }
# key 是 (prefix, layer_idx) 元组，保证不同组件同编号层不混淆
# ─────────────────────────────────────────────

def scan_all_qkv(all_shard_headers: dict) -> dict:
    """
    扫描所有 shard 中的 Q/K/V weight。
    以 (prefix, layer_idx) 为 key，保证：
    - 不同组件的同编号层互相独立
    - 层号是 safetensors 里的原始值
    """
    result: dict[tuple[str, int], dict] = {}

    for shard_name, (header, _) in all_shard_headers.items():
        for key in header.keys():
            m = re.search(r'layers\.(\d+)\.', key)
            if not m:
                continue

            layer_idx = int(m.group(1))
            prefix    = key[:m.start()]   # 精确截断
            suffix    = key[m.end():]

            role = _classify_qkv_suffix(suffix)
            if role is None:
                continue

            slot = (prefix, layer_idx)
            if slot not in result:
                result[slot] = {"q": None, "k": None, "v": None}

            if result[slot][role] is None:
                result[slot][role] = (shard_name, key)

    # 只保留 QKV 完整的槽
    return {
        slot: qkv for slot, qkv in result.items()
        if all(qkv[r] is not None for r in ("q", "k", "v"))
    }


# ─────────────────────────────────────────────
# 量化检测
# ─────────────────────────────────────────────

def check_quantization(model_id: str, token: str = None) -> tuple[bool, str]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    warnings = []

    try:
        r = requests.get(
            f"https://huggingface.co/{model_id}/resolve/main/config.json",
            headers=headers, timeout=15
        )
        if r.status_code == 200:
            cfg  = r.json()
            qcfg = cfg.get("quantization_config", {})
            qt   = (qcfg.get("quant_type","") or
                    qcfg.get("quant_method","") or
                    cfg.get("quantization","")).lower()
            if "gptq" in qt:
                return True, f"❌ GPTQ {qcfg.get('bits','?')}bit，请用原始 BF16 版本。"
            if "awq" in qt:
                return True, "❌ AWQ 量化，请用原始 BF16 版本。"
            if "bitsandbytes" in qt or "bnb" in qt:
                warnings.append("⚠️  bitsandbytes 量化，结果可能失真")
    except Exception:
        warnings.append("⚠️  无法读取 config.json")

    for kw in ["gptq","awq","gguf"]:
        if kw in model_id.lower():
            return True, f"❌ 模型名含 '{kw.upper()}'，请使用原始 BF16 版本。"

    try:
        all_files = list(list_repo_files(model_id, token=token))
        if any(f.endswith(".gguf") for f in all_files):
            return True, "❌ 检测到 .gguf 文件，不支持。"
        if not any(f.endswith(".safetensors") for f in all_files):
            return True, "❌ 未找到 .safetensors 文件。"
    except Exception as e:
        warnings.append(f"⚠️  文件列表检测失败：{e}")

    try:
        index_data  = find_index_file(model_id, token)
        first_shard = (
            sorted(set(index_data["weight_map"].values()))[0]
            if index_data else get_safetensor_files(model_id, token)[0]
        )
        hdr, _ = read_safetensors_header(get_file_url(model_id, first_shard), token)
        bad = [k for k in hdr if any(s in k for s in QUANTIZED_KEY_SIGNATURES)]
        if bad:
            return True, f"❌ 量化 key：{bad[:3]}"
        good = {hdr[k].get("dtype","") for k in list(hdr)[:20]} - UNSUPPORTED_SVD_DTYPES
        if good:
            warnings.append(f"✅ 权重格式：{good}")
    except Exception as e:
        warnings.append(f"⚠️  header 检测失败：{e}")

    return False, "\n".join(warnings) if warnings else "✅ 未检测到量化，可以正常分析"


# ─────────────────────────────────────────────
# GQA 推断
# ─────────────────────────────────────────────

def infer_gqa_params(
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    config_params: dict
) -> tuple[int, int, int]:
    q_rows, k_rows = W_q.shape[0], W_k.shape[0]

    d_head = config_params.get("head_dim") if config_params else None
    if not d_head and config_params:
        nh = config_params.get("num_attention_heads") or 1
        hs = config_params.get("hidden_size") or 0
        if hs and nh:
            d_head = hs // nh
    if not d_head:
        for c in [256, 128, 96, 80, 64, 48, 40, 32]:
            if q_rows % c == 0 and k_rows % c == 0:
                d_head = c
                break
    if not d_head:
        raise ValueError(f"无法推断 d_head：W_q={W_q.shape}, W_k={W_k.shape}")

    n_q  = q_rows // d_head
    n_kv = k_rows // d_head
    if n_q % n_kv != 0:
        raise ValueError(f"n_q={n_q} 不能被 n_kv={n_kv} 整除")
    return n_q, n_kv, d_head


# ─────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────

def compute_pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    am, bm = a - a.mean(), b - b.mean()
    den = torch.norm(am) * torch.norm(bm)
    return float(torch.dot(am, bm) / den) if den != 0 else 0.0

def compute_ssr(a: torch.Tensor, b: torch.Tensor) -> float:
    n  = min(a.shape[0], b.shape[0])
    an = a[:n] / (torch.norm(a[:n]) + 1e-10)
    bn = b[:n] / (torch.norm(b[:n]) + 1e-10)
    return float(torch.mean(torch.abs(an - bn)))

def compute_svr(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    n  = min(a.shape[0], b.shape[0])
    sa, sb = a[:n], b[:n]
    den = torch.dot(sb, sb)
    if den == 0: return 1.0, 0.0
    alpha = torch.dot(sa, sb) / den
    return float(alpha), float(torch.mean((sa - alpha * sb) ** 2))

def compute_cosU(U_a: torch.Tensor, U_b: torch.Tensor) -> float:
    r = min(U_a.shape[0], U_b.shape[0])
    c = min(U_a.shape[1], U_b.shape[1])
    Ua = U_a[:r, :c] / (torch.norm(U_a[:r, :c], dim=0, keepdim=True) + 1e-10)
    Ub = U_b[:r, :c] / (torch.norm(U_b[:r, :c], dim=0, keepdim=True) + 1e-10)
    return float(torch.diag(torch.abs(Ua.T @ Ub)).mean())

def compute_cosV(Vt_a: torch.Tensor, Vt_b: torch.Tensor) -> float:
    r = min(Vt_a.shape[0], Vt_b.shape[0])
    c = min(Vt_a.shape[1], Vt_b.shape[1])
    Va = Vt_a[:r, :c] / (torch.norm(Vt_a[:r, :c], dim=1, keepdim=True) + 1e-10)
    Vb = Vt_b[:r, :c] / (torch.norm(Vt_b[:r, :c], dim=1, keepdim=True) + 1e-10)
    return float(torch.abs((Va * Vb).sum(dim=1)).mean())


# ─────────────────────────────────────────────
# 逐头分析（原始层号直接传入，不做任何变换）
# ─────────────────────────────────────────────

def analyze_layer_heads(
    W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor,
    prefix: str,          # 组件前缀，用于日志
    layer_idx: int,       # 原始层号，直接来自 safetensors key
    n_q: int, n_kv: int, d_head: int,
) -> tuple[list[dict], str]:

    group   = n_q // n_kv
    records = []
    lines   = [
        f"\n{'─'*80}\n"
        f"[{prefix}] Layer {layer_idx:3d}  "
        f"n_q={n_q} n_kv={n_kv} group={group} d_head={d_head}\n"
        f"{'─'*80}\n"
        f"  {'KV':>3} {'Q':>3} │"
        f" {'P_QK':>7} {'Sp_QK':>7} {'SSR_QK':>8} │"
        f" {'SSR_QV':>8} {'SSR_KV':>8} │"
        f" {'cosU_QK':>8} {'cosU_QV':>8} {'cosU_KV':>8} │"
        f" {'cosV_QK':>8} {'cosV_QV':>8} {'cosV_KV':>8} │"
        f" {'α_QK':>7} {'α_QV':>7} {'α_KV':>7}\n"
    ]

    for kv_h in range(n_kv):
        k_t = W_k[kv_h*d_head:(kv_h+1)*d_head, :]
        v_t = W_v[kv_h*d_head:(kv_h+1)*d_head, :]
        U_k, s_k, Vt_k = torch.linalg.svd(k_t, full_matrices=False)
        U_v, s_v, Vt_v = torch.linalg.svd(v_t, full_matrices=False)

        alpha_kv, res_kv = compute_svr(s_k, s_v)
        cosU_KV = compute_cosU(U_k, U_v)
        cosV_KV = compute_cosV(Vt_k, Vt_v)
        ssr_kv  = compute_ssr(s_k, s_v)
        pkv     = compute_pearson(
            s_k[:min(len(s_k), len(s_v))],
            s_v[:min(len(s_k), len(s_v))]
        )

        for q_off in range(group):
            h   = kv_h * group + q_off
            q_t = W_q[h*d_head:(h+1)*d_head, :]
            U_q, s_q, Vt_q = torch.linalg.svd(q_t, full_matrices=False)

            nqk = min(len(s_q), len(s_k))
            nqv = min(len(s_q), len(s_v))

            pqk    = compute_pearson(s_q[:nqk], s_k[:nqk])
            spqk   = float(spearmanr(s_q[:nqk].numpy(), s_k[:nqk].numpy())[0])
            ssr_qk = compute_ssr(s_q, s_k)
            a_qk, r_qk = compute_svr(s_q, s_k)
            cU_QK  = compute_cosU(U_q, U_k)
            cV_QK  = compute_cosV(Vt_q, Vt_k)

            pqv    = compute_pearson(s_q[:nqv], s_v[:nqv])
            ssr_qv = compute_ssr(s_q, s_v)
            a_qv, r_qv = compute_svr(s_q, s_v)
            cU_QV  = compute_cosU(U_q, U_v)
            cV_QV  = compute_cosV(Vt_q, Vt_v)

            smxq = float(s_q.max())
            smnq = float(s_q[s_q>1e-10].min()) if (s_q>1e-10).any() else 0.
            smxk = float(s_k.max())
            smnk = float(s_k[s_k>1e-10].min()) if (s_k>1e-10).any() else 0.
            smxv = float(s_v.max())
            smnv = float(s_v[s_v>1e-10].min()) if (s_v>1e-10).any() else 0.

            records.append({
                # ★ prefix + layer_idx 完整保留，不做任何变换
                "prefix":        prefix,
                "layer":         layer_idx,
                "kv_head":       kv_h,
                "q_head":        h,
                "pearson_QK":    round(pqk,   6),
                "spearman_QK":   round(spqk,  6),
                "pearson_QV":    round(pqv,   6),
                "pearson_KV":    round(pkv,   6),
                "ssr_QK":        round(ssr_qk, 8),
                "ssr_QV":        round(ssr_qv, 8),
                "ssr_KV":        round(ssr_kv, 8),
                "cosU_QK":       round(cU_QK,  6),
                "cosU_QV":       round(cU_QV,  6),
                "cosU_KV":       round(cosU_KV,6),
                "cosV_QK":       round(cV_QK,  6),
                "cosV_QV":       round(cV_QV,  6),
                "cosV_KV":       round(cosV_KV,6),
                "alpha_QK":      round(a_qk,   4),
                "alpha_QV":      round(a_qv,   4),
                "alpha_KV":      round(alpha_kv,4),
                "alpha_res_QK":  round(r_qk,   6),
                "alpha_res_QV":  round(r_qv,   6),
                "alpha_res_KV":  round(res_kv, 6),
                "sigma_max_Q":   round(smxq, 4),
                "sigma_min_Q":   round(smnq, 4),
                "sigma_max_K":   round(smxk, 4),
                "sigma_min_K":   round(smnk, 4),
                "sigma_max_V":   round(smxv, 4),
                "sigma_min_V":   round(smnv, 4),
                "cond_Q":        round(smxq/(smnq+1e-10), 2),
                "cond_K":        round(smxk/(smnk+1e-10), 2),
                "cond_V":        round(smxv/(smnv+1e-10), 2),
            })

            lines.append(
                f"  {kv_h:>3d} {h:>3d} │"
                f" {pqk:>+7.4f} {spqk:>+7.4f} {ssr_qk:>8.6f} │"
                f" {ssr_qv:>8.6f} {ssr_kv:>8.6f} │"
                f" {cU_QK:>8.4f} {cU_QV:>8.4f} {cosU_KV:>8.4f} │"
                f" {cV_QK:>8.4f} {cV_QV:>8.4f} {cosV_KV:>8.4f} │"
                f" {a_qk:>7.4f} {a_qv:>7.4f} {alpha_kv:>7.4f}\n"
            )

    return records, "".join(lines)


# ─────────────────────────────────────────────
# 主分析函数
# ─────────────────────────────────────────────

def analyze_model(
    model_id:    str,
    hf_token:    str,
    start_layer: int,   # ★ 原始层号起点
    end_layer:   int,   # ★ 原始层号终点（含）
    progress=gr.Progress()
):
    if not model_id.strip():
        return "❌ 请输入模型 ID", None

    token = hf_token.strip() or None
    log   = [f"🔍 分析模型：{model_id}  层范围：{start_layer}~{end_layer}\n{'═'*80}\n"]
    all_records: list[dict] = []

    # ── 量化检测 ─────────────────────────────────
    progress(0.02, desc="量化检测...")
    blocked, qmsg = check_quantization(model_id, token)
    log.append(f"【量化检测】\n{qmsg}\n{'─'*80}\n")
    if blocked:
        return "".join(log), None

    # ── config.json ───────────────────────────────
    config_params = {}
    try:
        r = requests.get(
            f"https://huggingface.co/{model_id}/resolve/main/config.json",
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=15
        )
        if r.status_code == 200:
            config_params = extract_config_params(r.json())
            log.append(
                f"📋 config：model_type={config_params.get('model_type')}  "
                f"hidden={config_params.get('hidden_size')}  "
                f"n_heads={config_params.get('num_attention_heads')}  "
                f"n_kv={config_params.get('num_key_value_heads')}  "
                f"head_dim={config_params.get('head_dim')}\n"
                f"{'─'*80}\n"
            )
    except Exception:
        log.append("⚠️  无法读取 config.json\n")

    # ── 获取 shard 列表 ───────────────────────────
    progress(0.05, desc="读取模型索引...")
    try:
        index_data  = find_index_file(model_id, token)
        shard_files = (
            sorted(set(index_data["weight_map"].values()))
            if index_data else get_safetensor_files(model_id, token)
        )
        log.append(f"📦 共 {len(shard_files)} 个 shard\n")
    except requests.exceptions.HTTPError as e:
        return _http_error_msg(e, model_id), None

    # ── 读取所有 shard header ─────────────────────
    progress(0.08, desc="读取 shard headers...")
    all_shard_headers: dict[str, tuple[dict, int]] = {}
    for sf in shard_files:
        try:
            h, hs = read_safetensors_header(get_file_url(model_id, sf), token)
            all_shard_headers[sf] = (h, hs)
        except Exception as e:
            log.append(f"⚠️  {sf} 读取失败：{e}\n")

    log.append(f"🔑 总 key 数：{sum(len(h) for h,_ in all_shard_headers.values())}\n")

    # ── 扫描所有 QKV 槽 ───────────────────────────
    progress(0.12, desc="扫描 QKV 结构...")
    all_slots = scan_all_qkv(all_shard_headers)

    if not all_slots:
        sample = list(next(iter(all_shard_headers.values()))[0].keys())[:20]
        return "".join(log) + "⚠️ 无法识别 Q/K/V\n" + "\n".join(sample), None

    # ── 按原始层号过滤 [start_layer, end_layer] ───
    # ★ 直接用 safetensors key 里的层号，不做任何变换
    filtered_slots = {
        (prefix, layer_idx): qkv
        for (prefix, layer_idx), qkv in all_slots.items()
        if start_layer <= layer_idx <= end_layer
    }

    if not filtered_slots:
        # 打印实际存在的层号范围供参考
        by_prefix: dict[str, list[int]] = {}
        for (prefix, layer_idx) in all_slots:
            by_prefix.setdefault(prefix, []).append(layer_idx)
        info = "\n".join(
            f"   {p}: {sorted(v)}"
            for p, v in sorted(by_prefix.items())
        )
        return "".join(log) + f"⚠️ 层范围 {start_layer}~{end_layer} 内无数据。\n实际层号：\n{info}\n", None

    # ── 打印结构概览 ──────────────────────────────
    by_prefix: dict[str, list[int]] = {}
    for (prefix, layer_idx) in filtered_slots:
        by_prefix.setdefault(prefix, []).append(layer_idx)

    log.append(f"📐 层范围 {start_layer}~{end_layer} 内发现的组件：\n")
    for p, idxs in sorted(by_prefix.items()):
        log.append(f"   '{p}' → 层号 {sorted(idxs)}\n")
    log.append(f"{'═'*80}\n")

    # ── 按 (prefix, layer_idx) 顺序分析 ──────────
    # ★ sorted 保证输出有序，但层号本身不变
    sorted_slots = sorted(filtered_slots.items(), key=lambda x: (x[0][0], x[0][1]))
    total = len(sorted_slots)

    for i, ((prefix, layer_idx), qkv) in enumerate(sorted_slots):
        progress(0.15 + 0.80 * i / max(total, 1),
                 desc=f"{prefix} layer {layer_idx}...")

        q_shard, q_key = qkv["q"]
        k_shard, k_key = qkv["k"]
        v_shard, v_key = qkv["v"]

        try:
            W_q = load_tensor_remote(
                get_file_url(model_id, q_shard), q_key,
                *all_shard_headers[q_shard], token)
            W_k = load_tensor_remote(
                get_file_url(model_id, k_shard), k_key,
                *all_shard_headers[k_shard], token)
            W_v = load_tensor_remote(
                get_file_url(model_id, v_shard), v_key,
                *all_shard_headers[v_shard], token)
        except Exception as e:
            log.append(f"[{prefix}] Layer {layer_idx}: ❌ 加载失败：{e}\n")
            continue

        if W_q is None or W_k is None or W_v is None:
            log.append(f"[{prefix}] Layer {layer_idx}: ⚠️ tensor 为 None\n")
            continue

        try:
            n_q, n_kv, d_head = infer_gqa_params(W_q, W_k, config_params)
        except ValueError as e:
            log.append(f"[{prefix}] Layer {layer_idx}: ❌ GQA 推断失败：{e}\n")
            del W_q, W_k, W_v
            continue

        records, layer_log = analyze_layer_heads(
            W_q, W_k, W_v,
            prefix,      # 传入原始前缀
            layer_idx,   # ★ 传入原始层号，函数内不做任何变换
            n_q, n_kv, d_head,
        )
        all_records.extend(records)
        log.append(layer_log)
        del W_q, W_k, W_v

    # ── 汇总 ─────────────────────────────────────
    if not all_records:
        return "".join(log) + "\n❌ 未获得任何有效结果\n", None

    df = pd.DataFrame(all_records)

    def stat(arr, name):
        return (f"  {name:<14}"
                f" Median={np.median(arr):.6f}"
                f" Mean={np.mean(arr):.6f}"
                f" Min={np.min(arr):.6f}"
                f" Max={np.max(arr):.6f}\n")

    summary = [f"\n{'═'*80}\n📊 汇总 — {model_id}  层 {start_layer}~{end_layer}\n{'═'*80}\n"]

    # 按 prefix 分组汇总
    for pfx in df["prefix"].unique():
        pdf = df[df["prefix"] == pfx]
        summary.append(
            f"\n▶ {pfx}\n"
            f"  记录：{len(pdf)} 条，"
            f"层：{sorted(pdf['layer'].unique())}\n"
        )
        summary += [
            "  【第一定律 Pearson r → 1】\n",
            stat(pdf["pearson_QK"].values, "Q-K:"),
            stat(pdf["pearson_QV"].values, "Q-V:"),
            stat(pdf["pearson_KV"].values, "K-V:"),
            "  【第二定律 SSR → 0】\n",
            stat(pdf["ssr_QK"].values, "Q-K:"),
            stat(pdf["ssr_QV"].values, "Q-V:"),
            stat(pdf["ssr_KV"].values, "K-V:"),
            "  【第四定律 cosU 输出子空间】\n",
            stat(pdf["cosU_QK"].values, "cosU Q-K:"),
            stat(pdf["cosU_QV"].values, "cosU Q-V:"),
            stat(pdf["cosU_KV"].values, "cosU K-V:"),
            "  【第五定律 cosV 输入子空间】\n",
            stat(pdf["cosV_QK"].values, "cosV Q-K:"),
            stat(pdf["cosV_QV"].values, "cosV Q-V:"),
            stat(pdf["cosV_KV"].values, "cosV K-V:"),
            "  【第三定律 条件数】\n",
            stat(pdf["cond_Q"].values, "cond Q:"),
            stat(pdf["cond_K"].values, "cond K:"),
            stat(pdf["cond_V"].values, "cond V:"),
        ]

    summary.append(f"\n⚡ 理论极值：Pearson→1, SSR→0, cosU(QV)<1/√d_head\n{'═'*80}\n")
    log.extend(summary)

    return "".join(log), df


def inspect_model_structure(
    model_id: str,
    hf_token: str,
    progress=gr.Progress()
) -> str:
    """
    不做任何分析，只打印模型的原始 key 结构。
    让用户自己看清楚每一层到底有什么。
    """
    token = hf_token.strip() or None
    log   = [f"🔬 结构探测：{model_id}\n{'═'*80}\n"]

    # 获取 shard 列表
    try:
        index_data  = find_index_file(model_id, token)
        shard_files = (
            sorted(set(index_data["weight_map"].values()))
            if index_data else get_safetensor_files(model_id, token)
        )
    except Exception as e:
        return f"❌ 获取文件列表失败：{e}"

    # 读取所有 header
    all_shard_headers = {}
    for sf in shard_files:
        try:
            h, hs = read_safetensors_header(get_file_url(model_id, sf), token)
            all_shard_headers[sf] = (h, hs)
        except Exception as e:
            log.append(f"⚠️  {sf}：{e}\n")

    # ── 收集所有含 layers.{N}. 的 key ────────────
    # 结构：{ layer_idx: [ (prefix, suffix, shape, dtype) ] }
    layer_entries: dict[int, list] = {}

    for shard_name, (header, _) in all_shard_headers.items():
        for key, info in header.items():
            m = re.search(r'layers\.(\d+)\.', key)
            if not m:
                continue
            layer_idx = int(m.group(1))
            prefix    = key[:m.start()]
            suffix    = key[m.end():]
            shape     = info.get("shape", [])
            dtype     = info.get("dtype", "?")

            if layer_idx not in layer_entries:
                layer_entries[layer_idx] = []
            layer_entries[layer_idx].append((prefix, suffix, shape, dtype))

    if not layer_entries:
        return "".join(log) + "⚠️ 未找到任何含 layers.{N}. 的 key\n"

    # ── 打印结构 ──────────────────────────────────
    log.append(f"📊 共发现层号：{sorted(layer_entries.keys())}\n")
    log.append(f"{'─'*80}\n")

    for layer_idx in sorted(layer_entries.keys()):
        entries = layer_entries[layer_idx]

        # 按 prefix 分组
        by_prefix: dict[str, list] = {}
        for prefix, suffix, shape, dtype in entries:
            by_prefix.setdefault(prefix, []).append((suffix, shape, dtype))

        log.append(f"\n【Layer {layer_idx}】— 共 {len(entries)} 个 key，"
                   f"涉及 {len(by_prefix)} 个组件前缀\n")

        for prefix, items in sorted(by_prefix.items()):
            log.append(f"  前缀: '{prefix}'\n")
            for suffix, shape, dtype in sorted(items):
                log.append(f"    {suffix:<50} {str(shape):<20} {dtype}\n")

    log.append(f"\n{'═'*80}\n")
    log.append("📌 说明：\n")
    log.append("  - 如果每层只有一个前缀 → 该层属于单一组件\n")
    log.append("  - 如果每层有多个前缀 → 不同组件恰好共用同一层号（独立权重，不混合）\n")
    log.append("  - 层号只是 key 名里的数字，不代表物理上是同一层\n")

    return "".join(log)

# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────

with gr.Blocks(title="Wang's Five Laws — LLM Spectral Analyzer") as demo:

    with gr.Tabs():

        # ── Tab 1：结构探测 ────────────────────────
        with gr.Tab("🔬 结构探测"):
            gr.Markdown("""
            **先运行这个**，看清模型的原始 key 结构，
            再决定分析哪些层号。
            """)
            with gr.Row():
                inspect_model_input = gr.Textbox(
                    label="模型 ID",
                    value="google/gemma-4-e2b"
                )
                inspect_token_input = gr.Textbox(
                    label="HF Token",
                    type="password"
                )
            inspect_btn = gr.Button("🔍 探测结构", variant="secondary")
            inspect_output = gr.Textbox(
                label="原始结构",
                lines=50, max_lines=200
            )
            inspect_btn.click(
                fn=inspect_model_structure,
                inputs=[inspect_model_input, inspect_token_input],
                outputs=[inspect_output]
            )

        # ── Tab 2：分析 ───────────────────────────
        with gr.Tab("📊 分析"):
            gr.Markdown("""
            # 🔬 Wang's Five Laws — LLM Spectral Analyzer
            **Mathematical Foundations of Large Language Models (MF-LLM)**

            通过 **HTTP Range Request** 直接读取 HF 权重，**无需下载整个模型**。  
            按 safetensors 原始层号分析，支持混合模态模型（视觉/音频/语言同时输出）。

            | 定律 | 指标 | 理论极值 |
            |------|------|---------|
            | 第一定律 | Pearson r | → 1 |
            | 第二定律 | SSR | → 0 |
            | 第三定律 | 条件数 κ | 越小越好 |
            | 第四定律 | cosU(Uq,Uv) | < 1/√d_head（超正交） |
            | 第五定律 | cosV | ≈ 1/√d_model（随机正交） |

            [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19707844-blue)](https://doi.org/10.5281/zenodo.19707844)
            [![HAL](https://img.shields.io/badge/HAL-hal--05609398-red)](https://hal.science/hal-05609398)
            """)
            with gr.Row():
                with gr.Column(scale=2):
                    model_input = gr.Textbox(
                        label="HuggingFace 模型 ID",
                        value="google/gemma-4-e2b"
                    )
                    token_input = gr.Textbox(
                        label="HF Access Token",
                        type="password"
                    )
                    with gr.Row():
                        start_layer_input = gr.Number(
                            label="起始层号（含）",
                            value=0, minimum=0, maximum=999, precision=0
                        )
                        end_layer_input = gr.Number(
                            label="结束层号（含）",
                            value=5, minimum=0, maximum=999, precision=0
                        )
                    analyze_btn = gr.Button("🚀 开始分析", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### ✅ 推荐模型
                    ```
                    google/gemma-4-e2b
                    google/gemma-4-31b-it
                    Qwen/Qwen2.5-14B-Instruct
                    meta-llama/Llama-3-8B
                    deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
                    ```
                    ### 层号说明
                    - 层号 = safetensors key 中 `layers.{N}` 的 **N**
                    - **不按组件重排**，原始值直接输出
                    - 混合模态模型（如 Gemma-4）：
                    - `layers.0~11` 同时含 audio/vision/text 层
                    - 全部输出，按前缀区分组件

                    ### 示例：Gemma-4-E2B
                    | 组件 | 层范围 |
                    |------|--------|
                    | audio_tower | 0~11 |
                    | language_model | 0~34 |
                    | vision_tower | 0~15 |
                    """)

            log_output = gr.Textbox(label="分析日志", lines=40, max_lines=300)
            table_output = gr.Dataframe(
                label="逐头全指标结果表",
                headers=[
                    "prefix","layer","kv_head","q_head",
                    "pearson_QK","spearman_QK","pearson_QV","pearson_KV",
                    "ssr_QK","ssr_QV","ssr_KV",
                    "cosU_QK","cosU_QV","cosU_KV",
                    "cosV_QK","cosV_QV","cosV_KV",
                    "alpha_QK","alpha_QV","alpha_KV",
                    "alpha_res_QK","alpha_res_QV","alpha_res_KV",
                    "sigma_max_Q","sigma_min_Q",
                    "sigma_max_K","sigma_min_K",
                    "sigma_max_V","sigma_min_V",
                    "cond_Q","cond_K","cond_V",
                ]
            )

            analyze_btn.click(
                fn=analyze_model,
                inputs=[model_input, token_input,
                        start_layer_input, end_layer_input],
                outputs=[log_output, table_output]
            )

    

if __name__ == "__main__":
    demo.launch()