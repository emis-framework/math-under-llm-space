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
    if code == 401: return "❌ 401 未授权：请填写有效的 HF Access Token"
    if code == 403: return f"❌ 403 禁止访问：请先接受 {model_id} 的使用协议"
    if code == 404: return f"❌ 404 未找到：模型 {model_id} 不存在"
    return f"❌ HTTP {code}：{e}"


# ─────────────────────────────────────────────
# Gemma4 / 嵌套 config 安全解析
# ─────────────────────────────────────────────

def extract_config_params(config: dict) -> dict:
    if config is None:
        return {}
    text_cfg = config.get("text_config", {}) or {}

    def get_field(*keys):
        for k in keys:
            v = config.get(k)
            if v is not None:
                return v
            v = text_cfg.get(k)
            if v is not None:
                return v
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
    """layers.{N}. 之后的后缀 → 'q'/'k'/'v'/None"""
    if not suffix.endswith(".weight"):
        return None
    excludes = ["norm", "rope", "embed", "lm_head", "layernorm", "ln_"]
    s = suffix.lower()
    if any(e in s for e in excludes):
        return None
    if any(p in s for p in ["q_proj", "wq", "query", "q_a", "q_b"]):
        return "q"
    if any(p in s for p in ["k_proj", "wk", "key",   "k_a", "k_b"]):
        return "k"
    if any(p in s for p in ["v_proj", "wv", "value", "v_a", "v_b"]):
        return "v"
    return None


# ─────────────────────────────────────────────
# 【核心】按组件前缀分组发现所有 QKV 层
# 每个前缀 = 一个独立组件（语言模型/视觉编码器/音频塔等）
# 组件内部层号保持原始值，不重排
# ─────────────────────────────────────────────

def discover_all_components(all_shard_headers: dict) -> dict:
    """
    返回：
    {
      prefix (str): {
          layer_idx (int): {
              "q": (shard_name, full_key),
              "k": (shard_name, full_key),
              "v": (shard_name, full_key),
          }
      }
    }
    每个 prefix 是一个独立的模型组件。
    层号是该组件内的原始层号，不做任何重排。
    """
    # 第一遍：收集所有前缀及其层角色
    prefix_data: dict[str, dict[int, dict]] = {}

    for shard_name, (header, _) in all_shard_headers.items():
        for key in header.keys():
            m = re.search(r'layers\.(\d+)\.', key)
            if not m:
                continue

            layer_idx = int(m.group(1))
            prefix    = key[:m.start()]   # 精确截断，不用 split
            suffix    = key[m.end():]

            role = _classify_qkv_suffix(suffix)
            if role is None:
                continue

            if prefix not in prefix_data:
                prefix_data[prefix] = {}
            if layer_idx not in prefix_data[prefix]:
                prefix_data[prefix][layer_idx] = {"q": None, "k": None, "v": None}

            if prefix_data[prefix][layer_idx][role] is None:
                prefix_data[prefix][layer_idx][role] = (shard_name, key)

    # 第二遍：只保留每个前缀中 QKV 完整的层
    result = {}
    for prefix, layers in prefix_data.items():
        complete = {
            idx: qkv for idx, qkv in layers.items()
            if all(qkv[r] is not None for r in ("q", "k", "v"))
        }
        if complete:
            result[prefix] = complete

    return result


# ─────────────────────────────────────────────
# 组件类型推断（用于 modality 标注）
# ─────────────────────────────────────────────

VISION_PREFIX_PATTERNS = [
    "vision", "visual", "img", "image",
    "patch_embed", "vit", "clip",
]
AUDIO_PREFIX_PATTERNS = [
    "audio", "speech", "whisper",
]
TEXT_PREFIX_PATTERNS = [
    "language_model", "transformer", "model.layers",
    "text", "decoder", "encoder",
]

def infer_modality(prefix: str) -> str:
    p = prefix.lower()
    if any(v in p for v in VISION_PREFIX_PATTERNS):
        return "vision"
    if any(a in p for a in AUDIO_PREFIX_PATTERNS):
        return "audio"
    # 默认视为 text（language model）
    return "text"


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
            cfg = r.json()
            qcfg = cfg.get("quantization_config", {})
            qt = (qcfg.get("quant_type", "") or
                  qcfg.get("quant_method", "") or
                  cfg.get("quantization", "")).lower()
            if "gptq" in qt:
                return True, f"❌ GPTQ {qcfg.get('bits','?')}bit 量化，请用原始 BF16 版本。"
            if "awq" in qt:
                return True, "❌ AWQ 量化，请用原始 BF16 版本。"
            if "bitsandbytes" in qt or "bnb" in qt:
                warnings.append("⚠️  检测到 bitsandbytes 量化，结果可能失真")
    except Exception:
        warnings.append("⚠️  无法读取 config.json")

    for kw in ["gptq", "awq", "gguf"]:
        if kw in model_id.lower():
            return True, f"❌ 模型名含 '{kw.upper()}'，请使用原始 BF16 版本。"

    try:
        all_files = list(list_repo_files(model_id, token=token))
        if any(f.endswith(".gguf") for f in all_files):
            return True, "❌ 检测到 .gguf 文件，不支持该格式。"
        if not any(f.endswith(".safetensors") for f in all_files):
            return True, "❌ 未找到 .safetensors 文件。"
    except Exception as e:
        warnings.append(f"⚠️  文件列表检测失败：{e}")

    try:
        index_data = find_index_file(model_id, token)
        first_shard = (
            sorted(set(index_data["weight_map"].values()))[0]
            if index_data else get_safetensor_files(model_id, token)[0]
        )
        hdr, _ = read_safetensors_header(get_file_url(model_id, first_shard), token)
        bad = [k for k in hdr if any(s in k for s in QUANTIZED_KEY_SIGNATURES)]
        if bad:
            return True, f"❌ 检测到量化 key：{bad[:3]}"
        good = {hdr[k].get("dtype", "") for k in list(hdr)[:20]} - UNSUPPORTED_SVD_DTYPES
        if good:
            warnings.append(f"✅ 权重格式：{good}")
    except Exception as e:
        warnings.append(f"⚠️  header 检测失败：{e}")

    return False, "\n".join(warnings) if warnings else "✅ 未检测到量化，可以正常分析"


# ─────────────────────────────────────────────
# GQA 参数推断
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
        for c in [256, 128, 96, 80, 64, 32]:
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

def compute_pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    am, bm = a - a.mean(), b - b.mean()
    den = torch.norm(am) * torch.norm(bm)
    return float(torch.dot(am, bm) / den) if den != 0 else 0.0

def compute_ssr(a: torch.Tensor, b: torch.Tensor) -> float:
    n = min(a.shape[0], b.shape[0])
    an = a[:n] / (torch.norm(a[:n]) + 1e-10)
    bn = b[:n] / (torch.norm(b[:n]) + 1e-10)
    return float(torch.mean(torch.abs(an - bn)))

def compute_svr(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    n = min(a.shape[0], b.shape[0])
    sa, sb = a[:n], b[:n]
    den = torch.dot(sb, sb)
    if den == 0:
        return 1.0, 0.0
    alpha = torch.dot(sa, sb) / den
    return float(alpha), float(torch.mean((sa - alpha * sb) ** 2))

def compute_cosU(U_a: torch.Tensor, U_b: torch.Tensor) -> float:
    r = min(U_a.shape[0], U_b.shape[0])
    c = min(U_a.shape[1], U_b.shape[1])
    Ua = U_a[:r, :c]
    Ub = U_b[:r, :c]
    Ua = Ua / (torch.norm(Ua, dim=0, keepdim=True) + 1e-10)
    Ub = Ub / (torch.norm(Ub, dim=0, keepdim=True) + 1e-10)
    return float(torch.diag(torch.abs(Ua.T @ Ub)).mean())

def compute_cosV(Vt_a: torch.Tensor, Vt_b: torch.Tensor) -> float:
    r = min(Vt_a.shape[0], Vt_b.shape[0])
    c = min(Vt_a.shape[1], Vt_b.shape[1])  # ← 关键：列也取 min
    Va = Vt_a[:r, :c]
    Vb = Vt_b[:r, :c]
    Va = Va / (torch.norm(Va, dim=1, keepdim=True) + 1e-10)
    Vb = Vb / (torch.norm(Vb, dim=1, keepdim=True) + 1e-10)
    return float(torch.abs((Va * Vb).sum(dim=1)).mean())


# ─────────────────────────────────────────────
# 逐头分析（保留原始层号）
# ─────────────────────────────────────────────

def analyze_layer_heads(
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    layer_idx: int,          # 原始层号，不重排
    n_q: int, n_kv: int, d_head: int,
    modality: str,
) -> tuple[list[dict], str]:

    group = n_q // n_kv
    records, lines = [], []

    lines.append(
        f"\n{'─'*80}\n"
        f"Layer {layer_idx:3d}  [{modality}]  "
        f"n_q={n_q} n_kv={n_kv} group={group} d_head={d_head}\n"
        f"{'─'*80}\n"
        f"  {'KV':>3} {'Q':>3} │"
        f" {'P_QK':>7} {'Sp_QK':>7} {'SSR_QK':>8} │"
        f" {'SSR_QV':>8} {'SSR_KV':>8} │"
        f" {'cosU_QK':>8} {'cosU_QV':>8} {'cosU_KV':>8} │"
        f" {'cosV_QK':>8} {'cosV_QV':>8} {'cosV_KV':>8} │"
        f" {'α_QK':>7} {'α_QV':>7} {'α_KV':>7}\n"
    )

    for kv_h in range(n_kv):
        k_t = W_k[kv_h*d_head:(kv_h+1)*d_head, :]
        v_t = W_v[kv_h*d_head:(kv_h+1)*d_head, :]
        U_k, s_k, Vt_k = torch.linalg.svd(k_t, full_matrices=False)
        U_v, s_v, Vt_v = torch.linalg.svd(v_t, full_matrices=False)

        alpha_kv, res_kv = compute_svr(s_k, s_v)
        cosU_KV = compute_cosU(U_k, U_v)
        cosV_KV = compute_cosV(Vt_k, Vt_v)
        ssr_kv  = compute_ssr(s_k, s_v)
        pkv     = compute_pearson_corr(s_k[:min(len(s_k),len(s_v))],
                                       s_v[:min(len(s_k),len(s_v))])

        for q_off in range(group):
            h = kv_h * group + q_off
            q_t = W_q[h*d_head:(h+1)*d_head, :]
            U_q, s_q, Vt_q = torch.linalg.svd(q_t, full_matrices=False)

            nqk = min(len(s_q), len(s_k))
            nqv = min(len(s_q), len(s_v))

            pqk  = compute_pearson_corr(s_q[:nqk], s_k[:nqk])
            spqk = float(spearmanr(s_q[:nqk].numpy(), s_k[:nqk].numpy())[0])
            ssr_qk = compute_ssr(s_q, s_k)
            alpha_qk, res_qk = compute_svr(s_q, s_k)
            cosU_QK = compute_cosU(U_q, U_k)
            cosV_QK = compute_cosV(Vt_q, Vt_k)

            pqv = compute_pearson_corr(s_q[:nqv], s_v[:nqv])
            ssr_qv = compute_ssr(s_q, s_v)
            alpha_qv, res_qv = compute_svr(s_q, s_v)
            cosU_QV = compute_cosU(U_q, U_v)
            cosV_QV = compute_cosV(Vt_q, Vt_v)

            smxq = float(s_q.max()); smnq = float(s_q[s_q>1e-10].min()) if (s_q>1e-10).any() else 0.
            smxk = float(s_k.max()); smnk = float(s_k[s_k>1e-10].min()) if (s_k>1e-10).any() else 0.
            smxv = float(s_v.max()); smnv = float(s_v[s_v>1e-10].min()) if (s_v>1e-10).any() else 0.

            records.append({
                "layer": layer_idx, "modality": modality,
                "kv_head": kv_h, "q_head": h,
                "pearson_QK": round(pqk,6), "spearman_QK": round(spqk,6),
                "pearson_QV": round(pqv,6), "pearson_KV":  round(pkv,6),
                "ssr_QK": round(ssr_qk,8), "ssr_QV": round(ssr_qv,8),
                "ssr_KV": round(ssr_kv,8),
                "cosU_QK": round(cosU_QK,6), "cosU_QV": round(cosU_QV,6),
                "cosU_KV": round(cosU_KV,6),
                "cosV_QK": round(cosV_QK,6), "cosV_QV": round(cosV_QV,6),
                "cosV_KV": round(cosV_KV,6),
                "alpha_QK": round(alpha_qk,4), "alpha_QV": round(alpha_qv,4),
                "alpha_KV": round(alpha_kv,4),
                "alpha_res_QK": round(res_qk,6), "alpha_res_QV": round(res_qv,6),
                "alpha_res_KV": round(res_kv,6),
                "sigma_max_Q": round(smxq,4), "sigma_min_Q": round(smnq,4),
                "sigma_max_K": round(smxk,4), "sigma_min_K": round(smnk,4),
                "sigma_max_V": round(smxv,4), "sigma_min_V": round(smnv,4),
                "cond_Q": round(smxq/(smnq+1e-10),2),
                "cond_K": round(smxk/(smnk+1e-10),2),
                "cond_V": round(smxv/(smnv+1e-10),2),
            })

            lines.append(
                f"  {kv_h:>3d} {h:>3d} │"
                f" {pqk:>+7.4f} {spqk:>+7.4f} {ssr_qk:>8.6f} │"
                f" {ssr_qv:>8.6f} {ssr_kv:>8.6f} │"
                f" {cosU_QK:>8.4f} {cosU_QV:>8.4f} {cosU_KV:>8.4f} │"
                f" {cosV_QK:>8.4f} {cosV_QV:>8.4f} {cosV_KV:>8.4f} │"
                f" {alpha_qk:>7.4f} {alpha_qv:>7.4f} {alpha_kv:>7.4f}\n"
            )

    return records, "".join(lines)


# ─────────────────────────────────────────────
# 主分析函数
# ─────────────────────────────────────────────

def analyze_model(
    model_id: str,
    hf_token: str,
    max_layers: int,
    progress=gr.Progress()
):
    if not model_id.strip():
        return "❌ 请输入模型 ID", None

    token     = hf_token.strip() or None
    max_l     = int(max_layers)
    log_lines = [f"🔍 分析模型：{model_id}\n{'═'*80}\n"]
    all_records: list[dict] = []

    # ── 量化检测 ─────────────────────────────────
    progress(0.02, desc="量化检测...")
    blocked, qmsg = check_quantization(model_id, token)
    log_lines.append(f"【量化检测】\n{qmsg}\n{'─'*80}\n")
    if blocked:
        return "".join(log_lines), None

    # ── config.json ───────────────────────────────
    config_params = {}
    try:
        r = requests.get(
            f"https://huggingface.co/{model_id}/resolve/main/config.json",
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=15
        )
        if r.status_code == 200:
            raw_cfg       = r.json()
            config_params = extract_config_params(raw_cfg)
            log_lines.append(
                f"📋 config.json：\n"
                f"   model_type          = {config_params.get('model_type')}\n"
                f"   hidden_size         = {config_params.get('hidden_size')}\n"
                f"   num_attention_heads = {config_params.get('num_attention_heads')}\n"
                f"   num_key_value_heads = {config_params.get('num_key_value_heads')}\n"
                f"   head_dim            = {config_params.get('head_dim')}\n"
                f"{'─'*80}\n"
            )
    except Exception:
        log_lines.append("⚠️  无法读取 config.json\n")

    # ── 获取 shard 列表 ───────────────────────────
    progress(0.05, desc="读取模型索引...")
    try:
        index_data = find_index_file(model_id, token)
        if index_data:
            shard_files = sorted(set(index_data["weight_map"].values()))
            log_lines.append(f"📦 分片模型，共 {len(shard_files)} 个 shard\n")
        else:
            shard_files = get_safetensor_files(model_id, token)
            log_lines.append(f"📦 文件：{shard_files}\n")
    except requests.exceptions.HTTPError as e:
        return _http_error_msg(e, model_id), None

    # ── 读取所有 shard header ─────────────────────
    progress(0.08, desc="读取 shard headers...")
    all_shard_headers: dict[str, tuple[dict, int]] = {}
    total_keys = 0
    for sf in shard_files:
        try:
            h, hs = read_safetensors_header(get_file_url(model_id, sf), token)
            all_shard_headers[sf] = (h, hs)
            total_keys += len(h)
        except Exception as e:
            log_lines.append(f"⚠️  {sf} header 读取失败：{e}\n")

    log_lines.append(f"🔑 总 key 数：{total_keys}\n")

    # ── 发现所有组件 ──────────────────────────────
    progress(0.12, desc="识别组件结构...")
    all_components = discover_all_components(all_shard_headers)

    if not all_components:
        sample = []
        for sf, (h, _) in list(all_shard_headers.items())[:1]:
            sample = list(h.keys())[:30]
        return "".join(log_lines) + "⚠️ 无法识别 Q/K/V key，前30个 key：\n" + "\n".join(sample), None

    # ── 打印组件概览 ──────────────────────────────
    log_lines.append("📐 发现组件：\n")
    for prefix, layers in sorted(all_components.items()):
        modality = infer_modality(prefix)
        sorted_l = sorted(layers.keys())
        log_lines.append(
            f"   [{modality:6s}] prefix='{prefix}' "
            f"层数={len(sorted_l)} "
            f"范围={sorted_l[0]}~{sorted_l[-1]}\n"
        )
    log_lines.append(f"{'─'*80}\n")

    # ── 逐组件逐层分析 ────────────────────────────
    # 按前缀排序，每个组件独立分析，层号保持原始值
    component_done = 0
    total_components = len(all_components)

    for prefix, layers in sorted(all_components.items()):
        modality    = infer_modality(prefix)
        sorted_idxs = sorted(layers.keys())

        log_lines.append(
            f"\n{'═'*80}\n"
            f"🔷 组件：'{prefix}'  [{modality}]  "
            f"共 {len(sorted_idxs)} 层\n"
            f"{'═'*80}\n"
        )

        # 组件内最多分析 max_layers 层（从原始层0开始，保持原始编号）
        layers_in_component = 0
        gqa_logged = False

        for layer_idx in sorted_idxs:
            if layers_in_component >= max_l:
                log_lines.append(
                    f"  ⏸️  已达到最大层数 {max_l}，该组件剩余层跳过\n"
                )
                break

            overall_progress = (
                component_done / total_components
                + (layers_in_component / max(len(sorted_idxs), 1)) / total_components
            )
            progress(
                0.15 + 0.80 * overall_progress,
                desc=f"{modality} 层 {layer_idx}..."
            )

            qkv = layers[layer_idx]
            q_shard, q_key = qkv["q"]
            k_shard, k_key = qkv["k"]
            v_shard, v_key = qkv["v"]

            try:
                W_q = load_tensor_remote(
                    get_file_url(model_id, q_shard), q_key,
                    *all_shard_headers[q_shard], token
                )
                W_k = load_tensor_remote(
                    get_file_url(model_id, k_shard), k_key,
                    *all_shard_headers[k_shard], token
                )
                W_v = load_tensor_remote(
                    get_file_url(model_id, v_shard), v_key,
                    *all_shard_headers[v_shard], token
                )
            except Exception as e:
                log_lines.append(f"Layer {layer_idx}: ❌ 加载失败：{e}\n")
                layers_in_component += 1
                continue

            if W_q is None or W_k is None or W_v is None:
                log_lines.append(f"Layer {layer_idx}: ⚠️ tensor 为 None，跳过\n")
                layers_in_component += 1
                continue

            try:
                # 组件内不传全局 config（避免参数错配视觉组件）
                # 对语言模型组件才传 config_params
                cfg = config_params if modality == "text" else {}
                n_q, n_kv, d_head = infer_gqa_params(W_q, W_k, cfg)
            except ValueError as e:
                log_lines.append(f"Layer {layer_idx}: ❌ GQA 推断失败：{e}\n")
                del W_q, W_k, W_v
                layers_in_component += 1
                continue

            if not gqa_logged:
                log_lines.append(
                    f"🧠 注意力结构：n_q={n_q} n_kv={n_kv} "
                    f"group={n_q//n_kv} d_head={d_head}\n"
                    f"   W_q={list(W_q.shape)} "
                    f"W_k={list(W_k.shape)} "
                    f"W_v={list(W_v.shape)}\n"
                )
                gqa_logged = True

            records, layer_log = analyze_layer_heads(
                W_q, W_k, W_v,
                layer_idx,          # ← 原始层号，不重排
                n_q, n_kv, d_head,
                modality=modality,
            )
            all_records.extend(records)
            log_lines.append(layer_log)

            del W_q, W_k, W_v
            layers_in_component += 1

        component_done += 1

    # ── 全局汇总 ──────────────────────────────────
    if not all_records:
        return "".join(log_lines) + "\n❌ 未获得任何有效结果\n", None

    df = pd.DataFrame(all_records)

    def stat_block(arr, name):
        if len(arr) == 0:
            return f"  {name:<14} 无数据\n"
        return (
            f"  {name:<14}"
            f" Median={np.median(arr):.6f}"
            f" Mean={np.mean(arr):.6f}"
            f" Min={np.min(arr):.6f}"
            f" Max={np.max(arr):.6f}\n"
        )

    # 按 modality 分组汇总
    summary = [f"\n{'═'*80}\n📊 王氏五定律全局汇总 — {model_id}\n{'═'*80}\n"]

    for mod in df["modality"].unique():
        mdf = df[df["modality"] == mod]
        summary.append(
            f"\n▶ [{mod}] {len(mdf)} 条记录 "
            f"（{mdf['layer'].nunique()} 层 × "
            f"{mdf.groupby('layer').size().iloc[0]} 头/层）\n"
        )
        summary += [
            f"  【第一定律 Pearson r → 1】\n",
            stat_block(mdf["pearson_QK"].values, "Q-K:"),
            stat_block(mdf["pearson_QV"].values, "Q-V:"),
            stat_block(mdf["pearson_KV"].values, "K-V:"),
            f"  【第二定律 SSR → 0】\n",
            stat_block(mdf["ssr_QK"].values, "Q-K:"),
            stat_block(mdf["ssr_QV"].values, "Q-V:"),
            stat_block(mdf["ssr_KV"].values, "K-V:"),
            f"  【第四定律 cosU 输出子空间】\n",
            stat_block(mdf["cosU_QK"].values, "cosU Q-K:"),
            stat_block(mdf["cosU_QV"].values, "cosU Q-V:"),
            stat_block(mdf["cosU_KV"].values, "cosU K-V:"),
            f"  【第五定律 cosV 输入子空间】\n",
            stat_block(mdf["cosV_QK"].values, "cosV Q-K:"),
            stat_block(mdf["cosV_QV"].values, "cosV Q-V:"),
            stat_block(mdf["cosV_KV"].values, "cosV K-V:"),
            f"  【第三定律 条件数】\n",
            stat_block(mdf["cond_Q"].values,  "cond Q:"),
            stat_block(mdf["cond_K"].values,  "cond K:"),
            stat_block(mdf["cond_V"].values,  "cond V:"),
        ]

    summary.append(f"\n⚡ 理论极值：Pearson→1, SSR→0, cosU(QV)<1/√d_head\n{'═'*80}\n")
    log_lines.extend(summary)

    return "".join(log_lines), df


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────

with gr.Blocks(title="Wang's Five Laws — LLM Spectral Analyzer") as demo:

    gr.Markdown("""
    # 🔬 Wang's Five Laws — LLM Spectral Analyzer
    **Mathematical Foundations of Large Language Models (MF-LLM)**

    通过 **HTTP Range Request** 直接读取 HF 权重，**无需下载整个模型**。  
    支持 GQA + 多模态（视觉/音频/语言各组件独立分析，原始层号保留）。

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
                placeholder="google/gemma-4-e2b",
                value="google/gemma-4-e2b"
            )
            token_input = gr.Textbox(
                label="HF Access Token（公开模型可留空）",
                placeholder="hf_xxxxxxxxxxxxxxxx",
                type="password"
            )
            max_layers_input = gr.Slider(
                label="每个组件最大分析层数",
                minimum=1, maximum=100, value=4, step=1
            )
            analyze_btn = gr.Button("🚀 开始分析", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("""
            ### ✅ 推荐模型
            ```
            google/gemma-4-e2b          ← 视觉+语言
            google/gemma-4-31b-it       ← 视觉+语言
            Qwen/Qwen2.5-14B-Instruct
            meta-llama/Llama-3-8B
            deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
            ```
            ### 多模态分析说明
            - 每个组件（语言/视觉/音频）**独立分析**
            - 层号保持**原始编号**，不重排
            - 汇总统计**按 modality 分组**展示
            """)

    log_output = gr.Textbox(
        label="分析日志",
        lines=40, max_lines=200
    )
    table_output = gr.Dataframe(
        label="逐头全指标结果表",
        headers=[
            "layer","modality","kv_head","q_head",
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
        inputs=[model_input, token_input, max_layers_input],
        outputs=[log_output, table_output]
    )

if __name__ == "__main__":
    demo.launch()