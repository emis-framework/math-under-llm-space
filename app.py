import gradio as gr
import requests
import struct
import json
import re                          # [改动1] 新增：用于多模态层名过滤
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
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

# [改动1] 多模态视觉层关键词 → 跳过这些层
VISION_KEY_PATTERNS = [
    "vision", "visual", "image_encoder",
    "img_encoder", "patch_embed", "vit",
    "vision_tower", "mm_projector",
]


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
    # 过滤 __metadata__
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
    raw = r.content

    if torch_dtype == torch.bfloat16:
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.int16).view(torch.bfloat16)
    else:
        tensor = torch.frombuffer(bytearray(raw), dtype=torch_dtype)

    return tensor.reshape(shape).float()


def get_safetensor_files(model_id: str, token: str = None) -> list:
    kwargs = {"token": token} if token else {}
    return sorted(
        f for f in list_repo_files(model_id, **kwargs)
        if f.endswith(".safetensors")
    )


def find_index_file(model_id: str, token: str = None) -> dict | None:
    url = (f"https://huggingface.co/{model_id}/resolve/main/"
           f"model.safetensors.index.json")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(url, headers=headers, timeout=15)
    return r.json() if r.status_code == 200 else None


def _http_error_msg(e: requests.exceptions.HTTPError, model_id: str) -> str:
    code = e.response.status_code
    if code == 401: return "❌ 401 未授权：请填写有效的 HF Access Token"
    if code == 403: return f"❌ 403 禁止访问：请先接受 {model_id} 的使用协议"
    if code == 404: return f"❌ 404 未找到：模型 {model_id} 不存在"
    return f"❌ HTTP {code}：{e}"


# [改动1] 判断一个 key 是否属于视觉模态层
def is_vision_key(key: str) -> bool:
    key_lower = key.lower()
    return any(pat in key_lower for pat in VISION_KEY_PATTERNS)


# ─────────────────────────────────────────────
# 量化三重检测（不变）
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
            qt = (qcfg.get("quant_type","") or
                  qcfg.get("quant_method","") or
                  cfg.get("quantization","")).lower()
            if "gptq" in qt:
                bits = qcfg.get("bits","?")
                return True, f"❌ 检测到 GPTQ {bits}bit 量化，请改用原始 BF16 版本。"
            if "awq" in qt:
                return True, "❌ 检测到 AWQ 量化，请改用原始 BF16 版本。"
            if "bitsandbytes" in qt or "bnb" in qt:
                warnings.append("⚠️  检测到 bitsandbytes 量化，结果可能失真")
    except Exception:
        warnings.append("⚠️  无法读取 config.json")

    mid_lower = model_id.lower()
    for kw in ["gptq","awq","gguf"]:
        if kw in mid_lower:
            return True, f"❌ 模型名含 '{kw.upper()}'，为量化版本，请使用原始 BF16 版本。"

    try:
        all_files = list(list_repo_files(model_id, token=token))
        if any(f.endswith(".gguf") for f in all_files):
            return True, "❌ 检测到 .gguf 文件，不支持该格式。"
        if not any(f.endswith(".safetensors") for f in all_files):
            return True, "❌ 未找到 .safetensors 文件，仅支持 safetensors 格式。"
    except Exception as e:
        warnings.append(f"⚠️  文件列表检测失败：{e}")

    try:
        index_data = find_index_file(model_id, token)
        if index_data:
            first_shard = sorted(set(index_data["weight_map"].values()))[0]
        else:
            sf = get_safetensor_files(model_id, token)
            first_shard = sf[0]
        hdr, _ = read_safetensors_header(get_file_url(model_id, first_shard), token)
        all_keys = list(hdr.keys())
        bad_keys = [k for k in all_keys
                    if any(sig in k for sig in QUANTIZED_KEY_SIGNATURES)]
        if bad_keys:
            return True, f"❌ 检测到量化 key：{bad_keys[:3]}，请使用原始 BF16 版本。"
        dtypes = {hdr[k].get("dtype","") for k in all_keys[:20]}
        good   = dtypes - UNSUPPORTED_SVD_DTYPES
        if good:
            warnings.append(f"✅ 权重格式：{good}")
    except Exception as e:
        warnings.append(f"⚠️  header 检测失败：{e}")

    msg = "\n".join(warnings) if warnings else "✅ 未检测到量化，可以正常分析"
    return False, msg


# ─────────────────────────────────────────────
# GQA 参数自动推断（不变）
# ─────────────────────────────────────────────

def infer_gqa_params(
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    config: dict | None
) -> tuple[int,int,int]:
    q_rows = W_q.shape[0]
    k_rows = W_k.shape[0]

    d_head = None
    if config:
        d_head = (
            config.get("head_dim") or
            config.get("kv_channels") or
            config.get("hidden_size", 0) // max(config.get("num_attention_heads", 1), 1)
        )
        if d_head == 0:
            d_head = None

    if not d_head:
        for candidate in [256, 128, 96, 80, 64, 32]:
            if q_rows % candidate == 0 and k_rows % candidate == 0:
                d_head = candidate
                break

    if not d_head:
        raise ValueError(
            f"无法推断 d_head：W_q={W_q.shape}, W_k={W_k.shape}"
        )

    n_q_heads  = q_rows // d_head
    n_kv_heads = k_rows // d_head

    if n_q_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_q_heads={n_q_heads} 不能被 n_kv_heads={n_kv_heads} 整除"
        )
    return n_q_heads, n_kv_heads, d_head


# ─────────────────────────────────────────────
# [改动2] 指标计算函数：新增右奇异向量对齐
# ─────────────────────────────────────────────

def compute_pearson_corr(s_a: torch.Tensor, s_b: torch.Tensor) -> float:
    am = s_a - s_a.mean()
    bm = s_b - s_b.mean()
    num = torch.dot(am, bm)
    den = torch.norm(am, 2) * torch.norm(bm, 2)
    return float(num / den) if den != 0 else 0.0


def compute_singular_value_ratio(
    s_a: torch.Tensor, s_b: torch.Tensor
) -> tuple[float, float]:
    min_len = min(s_a.shape[0], s_b.shape[0])
    sa = s_a[:min_len]
    sb = s_b[:min_len]
    num = torch.dot(sa, sb)
    den = torch.dot(sb, sb)
    if den == 0:
        return 1.0, 0.0
    alpha    = num / den
    residual = torch.mean((sa - alpha * sb) ** 2).item()
    return float(alpha), float(residual)


def compute_ssr(s_a: torch.Tensor, s_b: torch.Tensor) -> float:
    min_len = min(s_a.shape[0], s_b.shape[0])
    sa = s_a[:min_len]
    sb = s_b[:min_len]
    sa_n = sa / (torch.norm(sa) + 1e-10)
    sb_n = sb / (torch.norm(sb) + 1e-10)
    return float(torch.mean(torch.abs(sa_n - sb_n)))


def compute_left_vector_alignment(
    U_a: torch.Tensor, U_b: torch.Tensor
) -> float:
    """
    左奇异向量（输出子空间）对齐度：
    cosU = mean_i |<u_a_i, u_b_i>|
    对应第四定律：cos(Uq,Uk) ≈ 1/√d_head（随机正交）
                  cos(Uq,Uv) < 1/√d_head（超正交）
    """
    min_c = min(U_a.shape[1], U_b.shape[1])
    Ua = U_a[:, :min_c]
    Ub = U_b[:, :min_c]
    Ua_n = Ua / (torch.norm(Ua, dim=0, keepdim=True) + 1e-10)
    Ub_n = Ub / (torch.norm(Ub, dim=0, keepdim=True) + 1e-10)
    return float(torch.diag(torch.abs(Ua_n.T @ Ub_n)).mean())


# [改动2] 新增：右奇异向量（输入子空间）对齐度
def compute_right_vector_alignment(
    Vt_a: torch.Tensor, Vt_b: torch.Tensor
) -> float:
    """
    右奇异向量（输入子空间）对齐度：
    cosV = mean_i |<v_a_i, v_b_i>|
    对应第五定律：所有对之间 ≈ 1/√d_model（全局随机正交）
    注意：SVD 返回 Vt（转置），每行是一个右奇异向量
    """
    min_r = min(Vt_a.shape[0], Vt_b.shape[0])
    Va_n = Vt_a[:min_r, :]
    Vb_n = Vt_b[:min_r, :]
    Va_n = Va_n / (torch.norm(Va_n, dim=1, keepdim=True) + 1e-10)
    Vb_n = Vb_n / (torch.norm(Vb_n, dim=1, keepdim=True) + 1e-10)
    return float(torch.abs((Va_n * Vb_n).sum(dim=1)).mean())


# ─────────────────────────────────────────────
# [改动3] 逐头分析：Q-K + Q-V + K-V 全指标
# ─────────────────────────────────────────────

def analyze_layer_heads(
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,       # [改动3] 新增 W_v 输入
    layer_idx: int,
    n_q_heads: int,
    n_kv_heads: int,
    d_head: int,
    modality: str = "text",  # [改动4] 新增 modality 标记
) -> tuple[list[dict], str]:
    """
    GQA 逐头全指标分析：
    对每个 KV 头：
      - 计算 K-V 对的全部指标（只算一次）
      - 对组内每个 Q 头：计算 Q-K、Q-V 全部指标
    """
    group_size = n_q_heads // n_kv_heads
    records    = []
    log_lines  = []

    log_lines.append(
        f"\n{'─'*80}\n"
        f"Layer {layer_idx:3d}  [{modality}]  "        # [改动4] 显示模态
        f"n_q={n_q_heads} n_kv={n_kv_heads} "
        f"group={group_size} d_head={d_head}\n"
        f"{'─'*80}\n"
    )
    # 表头
    log_lines.append(
        f"  {'KV':>3} {'Q':>3} │"
        f" {'P_QK':>7} {'Sp_QK':>7} {'SSR_QK':>8} │"
        f" {'SSR_QV':>8} {'SSR_KV':>8} │"
        f" {'cosU_QK':>8} {'cosU_QV':>8} {'cosU_KV':>8} │"
        f" {'cosV_QK':>8} {'cosV_QV':>8} {'cosV_KV':>8} │"
        f" {'α_QK':>7} {'α_QV':>7} {'α_KV':>7}\n"
    )

    for kv_h in range(n_kv_heads):

        # ── 提取 K / V 头矩阵 ─────────────────────────
        k_tensor = W_k[kv_h * d_head : (kv_h + 1) * d_head, :]
        v_tensor = W_v[kv_h * d_head : (kv_h + 1) * d_head, :]  # [改动3]

        U_k, s_k, Vt_k = torch.linalg.svd(k_tensor, full_matrices=False)
        U_v, s_v, Vt_v = torch.linalg.svd(v_tensor, full_matrices=False)  # [改动3]

        # ── K-V 指标（每个 KV 头只算一次）─────────────
        alpha_kv,  alpha_res_kv  = compute_singular_value_ratio(s_k, s_v)
        cosU_KV   = compute_left_vector_alignment(U_k, U_v)
        cosV_KV   = compute_right_vector_alignment(Vt_k, Vt_v)   # [改动2]
        ssr_kv    = compute_ssr(s_k, s_v)
        pearson_kv = compute_pearson_corr(
            s_k[:min(s_k.shape[0], s_v.shape[0])],
            s_v[:min(s_k.shape[0], s_v.shape[0])]
        )

        for q_offset in range(group_size):
            h_idx    = kv_h * group_size + q_offset
            q_tensor = W_q[h_idx * d_head : (h_idx + 1) * d_head, :]
            U_q, s_q, Vt_q = torch.linalg.svd(q_tensor, full_matrices=False)

            min_qk = min(s_q.shape[0], s_k.shape[0])
            min_qv = min(s_q.shape[0], s_v.shape[0])

            # ── Q-K 指标 ──────────────────────────────
            pearson_qk  = compute_pearson_corr(s_q[:min_qk], s_k[:min_qk])
            spearman_qk = float(spearmanr(
                s_q[:min_qk].cpu().numpy(),
                s_k[:min_qk].cpu().numpy()
            )[0])
            ssr_qk     = compute_ssr(s_q, s_k)
            alpha_qk,  alpha_res_qk  = compute_singular_value_ratio(s_q, s_k)
            cosU_QK    = compute_left_vector_alignment(U_q, U_k)
            cosV_QK    = compute_right_vector_alignment(Vt_q, Vt_k)   # [改动2]

            # ── Q-V 指标 ──────────────────────────────  [改动3]
            pearson_qv  = compute_pearson_corr(s_q[:min_qv], s_v[:min_qv])
            ssr_qv      = compute_ssr(s_q, s_v)
            alpha_qv,  alpha_res_qv  = compute_singular_value_ratio(s_q, s_v)
            cosU_QV    = compute_left_vector_alignment(U_q, U_v)
            cosV_QV    = compute_right_vector_alignment(Vt_q, Vt_v)   # [改动2]

            # ── 奇异值范围 ─────────────────────────────  [改动3]
            sig_max_q = float(s_q.max())
            sig_min_q = float(s_q[s_q > 1e-10].min()) if (s_q > 1e-10).any() else 0.0
            sig_max_k = float(s_k.max())
            sig_min_k = float(s_k[s_k > 1e-10].min()) if (s_k > 1e-10).any() else 0.0
            sig_max_v = float(s_v.max())
            sig_min_v = float(s_v[s_v > 1e-10].min()) if (s_v > 1e-10).any() else 0.0

            # 条件数（第三定律）
            cond_q = sig_max_q / (sig_min_q + 1e-10)
            cond_k = sig_max_k / (sig_min_k + 1e-10)
            cond_v = sig_max_v / (sig_min_v + 1e-10)

            records.append({
                # 位置信息
                "layer":          layer_idx,
                "modality":       modality,          # [改动4]
                "kv_head":        kv_h,
                "q_head":         h_idx,
                # 第一定律：谱线性对齐
                "pearson_QK":     round(pearson_qk,   6),
                "spearman_QK":    round(spearman_qk,  6),
                "pearson_QV":     round(pearson_qv,   6),  # [改动3]
                "pearson_KV":     round(pearson_kv,   6),  # [改动3]
                # 第二定律：SSR
                "ssr_QK":         round(ssr_qk,        8),
                "ssr_QV":         round(ssr_qv,        8),  # [改动3]
                "ssr_KV":         round(ssr_kv,        8),  # [改动3]
                # 第四定律：左奇异向量（输出子空间）
                "cosU_QK":        round(cosU_QK,       6),
                "cosU_QV":        round(cosU_QV,       6),  # [改动3]
                "cosU_KV":        round(cosU_KV,       6),  # [改动3]
                # 第五定律：右奇异向量（输入子空间）[改动2]
                "cosV_QK":        round(cosV_QK,       6),
                "cosV_QV":        round(cosV_QV,       6),
                "cosV_KV":        round(cosV_KV,       6),
                # 尺度因子
                "alpha_QK":       round(alpha_qk,      4),
                "alpha_QV":       round(alpha_qv,      4),  # [改动3]
                "alpha_KV":       round(alpha_kv,      4),  # [改动3]
                "alpha_res_QK":   round(alpha_res_qk,  6),
                "alpha_res_QV":   round(alpha_res_qv,  6),  # [改动3]
                "alpha_res_KV":   round(alpha_res_kv,  6),  # [改动3]
                # 奇异值范围 [改动3]
                "sigma_max_Q":    round(sig_max_q, 4),
                "sigma_min_Q":    round(sig_min_q, 4),
                "sigma_max_K":    round(sig_max_k, 4),
                "sigma_min_K":    round(sig_min_k, 4),
                "sigma_max_V":    round(sig_max_v, 4),
                "sigma_min_V":    round(sig_min_v, 4),
                # 条件数（第三定律）[改动3]
                "cond_Q":         round(cond_q, 2),
                "cond_K":         round(cond_k, 2),
                "cond_V":         round(cond_v, 2),
            })

            log_lines.append(
                f"  {kv_h:>3d} {h_idx:>3d} │"
                f" {pearson_qk:>+7.4f} {spearman_qk:>+7.4f} {ssr_qk:>8.6f} │"
                f" {ssr_qv:>8.6f} {ssr_kv:>8.6f} │"
                f" {cosU_QK:>8.4f} {cosU_QV:>8.4f} {cosU_KV:>8.4f} │"
                f" {cosV_QK:>8.4f} {cosV_QV:>8.4f} {cosV_KV:>8.4f} │"
                f" {alpha_qk:>7.4f} {alpha_qv:>7.4f} {alpha_kv:>7.4f}\n"
            )

    return records, "".join(log_lines)


# ─────────────────────────────────────────────
# 主分析函数
# ─────────────────────────────────────────────

def analyze_model(
    model_id:   str,
    hf_token:   str,
    max_layers: int,
    progress=gr.Progress()
):
    if not model_id.strip():
        return "❌ 请输入模型 ID", None

    token     = hf_token.strip() or None
    log_lines = [f"🔍 分析模型：{model_id}\n{'═'*80}\n"]
    all_records: list[dict] = []

    # ── 量化检测 ─────────────────────────────────
    progress(0.02, desc="量化检测...")
    is_blocked, quant_msg = check_quantization(model_id, token)
    log_lines.append(f"【量化检测】\n{quant_msg}\n{'─'*80}\n")
    if is_blocked:
        return "".join(log_lines), None

    # ── config.json ───────────────────────────────
    config = None
    try:
        r = requests.get(
            f"https://huggingface.co/{model_id}/resolve/main/config.json",
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=15
        )
        if r.status_code == 200:
            config = r.json()
            log_lines.append(
                f"📋 config.json：\n"
                f"   model_type          = {config.get('model_type')}\n"
                f"   hidden_size         = {config.get('hidden_size')}\n"
                f"   num_attention_heads = {config.get('num_attention_heads')}\n"
                f"   num_key_value_heads = {config.get('num_key_value_heads')}\n"
                f"   head_dim            = {config.get('head_dim')}\n"
                f"{'─'*80}\n"
            )
    except Exception:
        log_lines.append("⚠️  无法读取 config.json，将从 weight shape 自动推断\n")

    # ── 分片索引 ──────────────────────────────────
    progress(0.05, desc="读取模型索引...")
    try:
        index_data    = find_index_file(model_id, token)
        shard_headers: dict[str, tuple[dict, int]] = {}

        if index_data:
            weight_map = index_data["weight_map"]
            log_lines.append(
                f"📦 分片模型，共 {len(set(weight_map.values()))} 个 shard\n"
            )
        else:
            sf_files   = get_safetensor_files(model_id, token)
            weight_map = None
            log_lines.append(f"📦 单文件：{sf_files}\n")
    except requests.exceptions.HTTPError as e:
        return _http_error_msg(e, model_id), None

    # ── 探测第一个 shard ──────────────────────────
    progress(0.08, desc="识别层结构...")
    try:
        if index_data:
            first_shard = sorted(set(index_data["weight_map"].values()))[0]
        else:
            first_shard = sf_files[0]
        first_url = get_file_url(model_id, first_shard)
        first_header, first_hsize = read_safetensors_header(first_url, token)
        shard_headers[first_shard] = (first_header, first_hsize)
        all_keys = list(first_header.keys())
    except Exception as e:
        return f"❌ 读取 shard header 失败：{e}", None

    # [改动1] 区分文本层 key 和视觉层 key
    text_keys   = [k for k in all_keys if not is_vision_key(k)]
    vision_keys = [k for k in all_keys if is_vision_key(k)]
    log_lines.append(
        f"🔑 总 key 数：{len(all_keys)}  "
        f"（文本层：{len(text_keys)}，视觉层跳过：{len(vision_keys)}）\n"
    )

    # 识别 Q/K/V key 命名规则（只在文本 key 中识别）
    q_candidates = [k for k in text_keys if any(
        p in k for p in ["q_proj.weight","query.weight","q.weight","wq.weight"]
    )]
    if not q_candidates:
        sample = "\n".join(text_keys[:30])
        return f"⚠️ 无法识别文本层 Q/K/V key，前 30 个文本 key：\n{sample}", None

    sample_q = q_candidates[0]
    if   "q_proj"  in sample_q: q_sfx, k_sfx, v_sfx = "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"
    elif "query"   in sample_q: q_sfx, k_sfx, v_sfx = "attention.query.weight",  "attention.key.weight",     "attention.value.weight"
    elif "wq"      in sample_q: q_sfx, k_sfx, v_sfx = "attention.wq.weight",     "attention.wk.weight",      "attention.wv.weight"
    else:
        q_sfx = sample_q.split("layers.0.")[-1]
        k_sfx = q_sfx.replace("q.", "k.")
        v_sfx = q_sfx.replace("q.", "v.")

    log_lines.append(f"🔑 Q suffix：{q_sfx}\n")
    log_lines.append(f"🔑 K suffix：{k_sfx}\n")
    log_lines.append(f"🔑 V suffix：{v_sfx}\n")    # [改动3]
    log_lines.append(f"{'═'*80}\n")

    # ── 辅助：查找 key 所在 shard ─────────────────
    def get_shard_for_key(key: str) -> str | None:
        if index_data:
            return index_data["weight_map"].get(key)
        for sf in sf_files:
            if sf not in shard_headers:
                h, hs = read_safetensors_header(get_file_url(model_id, sf), token)
                shard_headers[sf] = (h, hs)
            if key in shard_headers[sf][0]:
                return sf
        return None

    # ── 逐层分析 ─────────────────────────────────
    gqa_logged = False

    for layer_idx in range(int(max_layers)):
        progress(
            0.10 + 0.85 * layer_idx / int(max_layers),
            desc=f"第 {layer_idx} 层..."
        )

        q_key = f"model.layers.{layer_idx}.{q_sfx}"
        k_key = f"model.layers.{layer_idx}.{k_sfx}"
        v_key = f"model.layers.{layer_idx}.{v_sfx}"    # [改动3]

        q_shard = get_shard_for_key(q_key)
        k_shard = get_shard_for_key(k_key)
        v_shard = get_shard_for_key(v_key)              # [改动3]

        if q_shard is None or k_shard is None:
            log_lines.append(
                f"\nLayer {layer_idx}: Q/K 未找到，分析结束（共 {layer_idx} 层）\n"
            )
            break

        # [改动3] V 找不到时降级处理（不阻断整体分析）
        if v_shard is None:
            log_lines.append(
                f"Layer {layer_idx}: ⚠️ V 未找到，跳过该层\n"
            )
            continue

        for shard in {q_shard, k_shard, v_shard}:
            if shard not in shard_headers:
                h, hs = read_safetensors_header(get_file_url(model_id, shard), token)
                shard_headers[shard] = (h, hs)

        try:
            W_q = load_tensor_remote(
                get_file_url(model_id, q_shard), q_key,
                *shard_headers[q_shard], token
            )
            W_k = load_tensor_remote(
                get_file_url(model_id, k_shard), k_key,
                *shard_headers[k_shard], token
            )
            W_v = load_tensor_remote(                   # [改动3]
                get_file_url(model_id, v_shard), v_key,
                *shard_headers[v_shard], token
            )
        except ValueError as e:
            log_lines.append(f"Layer {layer_idx}: ⚠️ 跳过（{e}）\n")
            continue

        if W_q is None or W_k is None or W_v is None:
            log_lines.append(f"Layer {layer_idx}: ⚠️ tensor 为 None，跳过\n")
            continue

        # [改动1] 判断该层是文本层还是视觉层
        modality = "vision" if is_vision_key(q_key) else "text"
        if modality == "vision":
            log_lines.append(f"Layer {layer_idx}: 🖼️ 视觉层，跳过\n")
            del W_q, W_k, W_v
            continue

        # GQA 推断
        try:
            n_q_heads, n_kv_heads, d_head = infer_gqa_params(W_q, W_k, config)
        except ValueError as e:
            log_lines.append(f"Layer {layer_idx}: ❌ GQA 推断失败：{e}\n")
            del W_q, W_k, W_v
            continue

        if not gqa_logged:
            log_lines.append(
                f"🧠 GQA 结构：n_q={n_q_heads} n_kv={n_kv_heads} "
                f"group={n_q_heads//n_kv_heads} d_head={d_head}\n"
                f"   W_q={list(W_q.shape)} W_k={list(W_k.shape)} "
                f"W_v={list(W_v.shape)}\n"   # [改动3]
                f"{'═'*80}\n"
            )
            gqa_logged = True

        # 逐头全指标计算
        records, layer_log = analyze_layer_heads(
            W_q, W_k, W_v,              # [改动3]
            layer_idx,
            n_q_heads, n_kv_heads, d_head,
            modality=modality            # [改动4]
        )
        all_records.extend(records)
        log_lines.append(layer_log)

        del W_q, W_k, W_v

    # ── 全局汇总 ──────────────────────────────────
    if all_records:
        df = pd.DataFrame(all_records)

        # [改动5] 分模态统计
        def stat_block(arr: np.ndarray, name: str) -> str:
            return (
                f"  {name:<14}"
                f" Median={np.median(arr):.6f}"
                f" Mean={np.mean(arr):.6f}"
                f" Min={np.min(arr):.6f}"
                f" Max={np.max(arr):.6f}\n"
            )

        text_df = df[df["modality"] == "text"]

        summary_lines = [
            f"\n{'═'*80}\n",
            f"📊 王氏五定律全局汇总 — {model_id}\n",
            f"{'═'*80}\n",
            f"文本层记录：{len(text_df)} 条  "
            f"（{text_df['layer'].nunique()} 层 × "
            f"{text_df.groupby('layer').size().iloc[0] if len(text_df)>0 else 0} 头/层）\n\n",

            f"【第一定律 — Pearson r（→ 1）】\n",
            stat_block(text_df["pearson_QK"].values, "Q-K:"),
            stat_block(text_df["pearson_QV"].values, "Q-V:"),   # [改动3]
            stat_block(text_df["pearson_KV"].values, "K-V:"),   # [改动3]

            f"\n【第二定律 — SSR（→ 0）】\n",
            stat_block(text_df["ssr_QK"].values, "Q-K:"),
            stat_block(text_df["ssr_QV"].values, "Q-V:"),       # [改动3]
            stat_block(text_df["ssr_KV"].values, "K-V:"),       # [改动3]

            f"\n【第四定律 — cosU 输出子空间（Q-K≈1/√d，Q-V<1/√d 超正交）】\n",
            stat_block(text_df["cosU_QK"].values, "cosU Q-K:"),
            stat_block(text_df["cosU_QV"].values, "cosU Q-V:"), # [改动3]
            stat_block(text_df["cosU_KV"].values, "cosU K-V:"), # [改动3]

            f"\n【第五定律 — cosV 输入子空间（≈1/√d_model 全局随机正交）】\n",  # [改动2]
            stat_block(text_df["cosV_QK"].values, "cosV Q-K:"),
            stat_block(text_df["cosV_QV"].values, "cosV Q-V:"),
            stat_block(text_df["cosV_KV"].values, "cosV K-V:"),

            f"\n【第三定律 — 条件数（越小越稳定）】\n",              # [改动3]
            stat_block(text_df["cond_Q"].values,  "cond Q:"),
            stat_block(text_df["cond_K"].values,  "cond K:"),
            stat_block(text_df["cond_V"].values,  "cond V:"),

            f"\n⚡ 理论极值：Pearson→1, SSR→0, cosU(QV)<1/√d_head\n",
            f"{'═'*80}\n",
        ]
        log_lines.extend(summary_lines)

        return "".join(log_lines), df
    else:
        return "".join(log_lines) + "\n❌ 未获得任何有效结果\n", None


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────

with gr.Blocks(title="Wang's Five Laws — LLM Spectral Analyzer") as demo:

    gr.Markdown("""
    # 🔬 Wang's Five Laws — LLM Spectral Analyzer
    **Mathematical Foundations of Large Language Models (MF-LLM)**

    通过 **HTTP Range Request** 直接读取 HF 权重，**无需下载整个模型**。
    支持 GQA + 多模态（自动跳过视觉层）。逐头计算全部五定律指标：

    | 定律 | 指标 | 理论极值 | 对象 |
    |------|------|---------|------|
    | 第一定律 | Pearson r / Spearman r | → 1 | Q-K |
    | 第二定律 | SSR | → 0 | Q-K, Q-V, K-V |
    | 第三定律 | 条件数 κ | 越小越好 | Q, K, V |
    | 第四定律 | cosU(Uq,Uk) | ≈1/√d_head；cosU(Uq,Uv)<1/√d_head | Q-K, Q-V, K-V |
    | 第五定律 | cosV(Vq,Vk) | ≈1/√d_model（随机正交） | Q-K, Q-V, K-V |

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
                label="最大分析层数",
                minimum=1, maximum=100, value=4, step=1
            )
            analyze_btn = gr.Button("🚀 开始分析", variant="primary")

        # [改动6] 更新推荐模型列表
        with gr.Column(scale=1):
            gr.Markdown("""
            ### ✅ 推荐模型
            ```
            Qwen/Qwen2.5-14B-Instruct        (GQA 8Q/2K)
            meta-llama/Llama-3-8B            (GQA)
            google/gemma-4-e2b               (MHA 多模态)
            google/gemma-4-e4b-it            (MHA 多模态)
            deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
            ```
            ### GQA 典型结构
            | 模型 | Q头 | KV头 | 每组 |
            |------|-----|------|------|
            | Qwen2.5-7B | 28 | 4 | 7 |
            | LLaMA-3-8B | 32 | 8 | 4 |
            | Qwen2.5-14B | 40 | 8 | 5 |
            | Gemma-4-E2B | 8 | 4 | 2 |

            ### 🖼️ 多模态说明
            - 视觉层自动跳过
            - 仅分析文本 Transformer 层
            - 跳过关键词：`vision / visual / vit / patch_embed`
            """)

    log_output = gr.Textbox(
        label="分析日志（逐头详情）",
        lines=35, max_lines=100
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