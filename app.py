import gradio as gr
import requests
import struct
import json
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
    return json.loads(r.content), header_size


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
# 量化三重检测
# ─────────────────────────────────────────────

def check_quantization(model_id: str, token: str = None) -> tuple[bool, str]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    warnings = []

    # 检测 1：config.json
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
                return True, (f"❌ 检测到 GPTQ {bits}bit 量化\n"
                               f"   请改用原始 BF16 版本。")
            if "awq" in qt:
                return True, "❌ 检测到 AWQ 量化，请改用原始 BF16 版本。"
            if "bitsandbytes" in qt or "bnb" in qt:
                warnings.append("⚠️  检测到 bitsandbytes 量化，结果可能失真")
    except Exception:
        warnings.append("⚠️  无法读取 config.json")

    # 检测 2：文件名 / 模型名关键词
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

    # 检测 3：header key 签名
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
            return True, (f"❌ 检测到量化 key：{bad_keys[:3]}\n"
                           f"   请使用原始 BF16 版本。")
        dtypes = {hdr[k].get("dtype","") for k in all_keys[:20]}
        good = dtypes - UNSUPPORTED_SVD_DTYPES
        if good:
            warnings.append(f"✅ 权重格式：{good}")
    except Exception as e:
        warnings.append(f"⚠️  header 检测失败：{e}")

    msg = "\n".join(warnings) if warnings else "✅ 未检测到量化，可以正常分析"
    return False, msg


# ─────────────────────────────────────────────
# GQA 参数自动推断
# ─────────────────────────────────────────────

def infer_gqa_params(W_q: torch.Tensor, W_k: torch.Tensor, config: dict | None) -> tuple[int,int,int]:
    """
    自动推断：
    - n_q_heads  : Q 头数量
    - n_kv_heads : KV 头数量（GQA）
    - d_head     : 每个头的维度

    权重 shape 约定（最常见）：
      W_q : (n_q_heads  * d_head, d_model)  → shape[0] = n_q * d_h
      W_k : (n_kv_heads * d_head, d_model)  → shape[0] = n_kv * d_h

    d_head 优先从 config.json 读取，其次用常见默认值猜测。
    """
    q_rows, d_model = W_q.shape[0], W_q.shape[1]
    k_rows          = W_k.shape[0]

    # 从 config.json 读取 d_head
    d_head = None
    if config:
        d_head = (
            config.get("head_dim") or
            config.get("kv_channels") or
            config.get("hidden_size", 0) // config.get("num_attention_heads", 1)
        )
        if d_head == 0:
            d_head = None

    # 如果 config 没给，用常见值探测（64, 80, 96, 128, 256）
    if not d_head:
        for candidate in [256, 128, 96, 80, 64]:
            if q_rows % candidate == 0 and k_rows % candidate == 0:
                d_head = candidate
                break

    if not d_head:
        raise ValueError(
            f"无法推断 d_head：W_q.shape={W_q.shape}, W_k.shape={W_k.shape}\n"
            f"请在 config.json 中确认 head_dim 字段。"
        )

    n_q_heads  = q_rows // d_head
    n_kv_heads = k_rows // d_head

    if n_q_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_q_heads={n_q_heads} 不能被 n_kv_heads={n_kv_heads} 整除，"
            f"请检查 d_head 推断是否正确。"
        )

    return n_q_heads, n_kv_heads, d_head


# ─────────────────────────────────────────────
# 逐头 SVD 指标计算
# ─────────────────────────────────────────────

def compute_pearson_corr_torch(s_q: torch.Tensor, s_k: torch.Tensor) -> float:
    sq = s_q.cpu().numpy()
    sk = s_k.cpu().numpy()
    r, _ = pearsonr(sq, sk)
    return float(r)


def compute_singular_value_ratio(
    s_q: torch.Tensor, s_k: torch.Tensor
) -> tuple[float, float]:
    """
    估计尺度因子 α = median(s_q / s_k)
    残差 = mean|s_q - α * s_k| / mean(s_q)
    """
    min_len = min(s_q.shape[0], s_k.shape[0])
    sq = s_q[:min_len]
    sk = s_k[:min_len]
    ratio = sq / (sk + 1e-10)
    alpha = float(ratio.median())
    residual = float((sq - alpha * sk).abs().mean() / (sq.mean() + 1e-10))
    return alpha, residual


def compute_left_vector_alignment(
    U_q: torch.Tensor, U_k: torch.Tensor
) -> float:
    """
    第四定律：左奇异向量（输出子空间）对齐度
    cos_u = mean_i |<u_q_i, u_k_i>|
    """
    min_len = min(U_q.shape[1], U_k.shape[1])
    U_q = U_q[:, :min_len]
    U_k = U_k[:, :min_len]
    cos_vals = (U_q * U_k).sum(dim=0).abs()
    return float(cos_vals.mean())


def compute_covariance_alignment(
    W_q: torch.Tensor, W_k: torch.Tensor, alpha: float
) -> float:
    """
    协方差矩阵对齐误差：
    err = ||W_q W_q^T - α² W_k W_k^T||_F / ||W_k W_k^T||_F
    """
    cov_q = W_q @ W_q.T
    cov_k = W_k @ W_k.T
    diff  = cov_q - (alpha ** 2) * cov_k
    err   = float(torch.norm(diff, p='fro') / (torch.norm(cov_k, p='fro') + 1e-10))
    return err


def compute_ssr(s_q: torch.Tensor, s_k: torch.Tensor) -> float:
    """
    第二定律：归一化谱形状残差
    SSR = mean_i |s̃_q_i - s̃_k_i|
    """
    min_len = min(s_q.shape[0], s_k.shape[0])
    sq = s_q[:min_len].cpu().numpy()
    sk = s_k[:min_len].cpu().numpy()
    sq_n = sq / (np.linalg.norm(sq) + 1e-10)
    sk_n = sk / (np.linalg.norm(sk) + 1e-10)
    return float(np.mean(np.abs(sq_n - sk_n)))


def analyze_layer_heads(
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    layer_idx: int,
    n_q_heads: int,
    n_kv_heads: int,
    d_head: int,
) -> tuple[list[dict], str]:
    """
    GQA 逐头分析：
    - 每个 KV 头对应 group_size = n_q_heads // n_kv_heads 个 Q 头
    - 每个 Q 头分别与其对应的 K 头做 SVD 指标计算
    """
    group_size = n_q_heads // n_kv_heads
    records    = []
    log_lines  = []

    log_lines.append(
        f"\n{'─'*70}\n"
        f"Layer {layer_idx:3d}  "
        f"[n_q={n_q_heads}, n_kv={n_kv_heads}, "
        f"group={group_size}, d_head={d_head}]\n"
        f"{'─'*70}\n"
    )
    log_lines.append(
        f"  {'KV头':>4}  {'Q头':>4}  "
        f"{'Pearson':>8}  {'Spearman':>9}  "
        f"{'α':>7}  {'α残差':>8}  "
        f"{'cos(Uq,Uk)':>10}  {'协方差误差':>10}  {'SSR':>10}\n"
    )

    for kv_h in range(n_kv_heads):
        # ── 提取 K 头矩阵 (d_head × d_model) ──
        k_tensor = W_k[kv_h * d_head : (kv_h + 1) * d_head, :]
        U_k, s_k, _ = torch.linalg.svd(k_tensor, full_matrices=False)

        for q_offset in range(group_size):
            h_idx = kv_h * group_size + q_offset

            # ── 提取 Q 头矩阵 (d_head × d_model) ──
            q_tensor = W_q[h_idx * d_head : (h_idx + 1) * d_head, :]
            U_q, s_q, _ = torch.linalg.svd(q_tensor, full_matrices=False)

            # 1. Pearson r（第一定律）
            min_len    = min(s_q.shape[0], s_k.shape[0])
            pearson_r  = compute_pearson_corr_torch(s_q[:min_len], s_k[:min_len])

            # 2. Spearman r（排名相关，对异常值更鲁棒）
            spearman_r, _ = spearmanr(
                s_q[:min_len].cpu().numpy(),
                s_k[:min_len].cpu().numpy()
            )

            # 3. 尺度因子 α 与残差
            alpha, alpha_res = compute_singular_value_ratio(s_q, s_k)

            # 4. 左奇异向量对齐（第四定律）
            cos_u = compute_left_vector_alignment(U_q, U_k)

            # 5. 协方差矩阵对齐误差
            cov_err = compute_covariance_alignment(q_tensor, k_tensor, alpha)

            # 6. SSR（第二定律）
            ssr = compute_ssr(s_q, s_k)

            records.append({
                "Layer":      layer_idx,
                "KV_head":    kv_h,
                "Q_head":     h_idx,
                "Pearson_r":  round(pearson_r,  6),
                "Spearman_r": round(float(spearman_r), 6),
                "Alpha":      round(alpha,       4),
                "Alpha_res":  round(alpha_res,   6),
                "cos_Uq_Uk":  round(cos_u,       6),
                "Cov_err":    round(cov_err,      6),
                "SSR":        round(ssr,          6),
            })

            log_lines.append(
                f"  KV={kv_h:>3d}  Q={h_idx:>3d}  "
                f"{pearson_r:>+8.4f}  {float(spearman_r):>+9.4f}  "
                f"{alpha:>7.4f}  {alpha_res:>8.2e}  "
                f"{cos_u:>10.4f}  {cov_err:>10.4f}  {ssr:>10.6f}\n"
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
    log_lines = [f"🔍 分析模型：{model_id}\n{'═'*70}\n"]
    all_records: list[dict] = []

    # ── 量化检测 ─────────────────────────────────
    progress(0.02, desc="量化检测...")
    is_blocked, quant_msg = check_quantization(model_id, token)
    log_lines.append(f"【量化检测】\n{quant_msg}\n{'─'*70}\n")
    if is_blocked:
        return "".join(log_lines), None

    # ── 读取 config.json（用于推断 d_head）────────
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
                f"   hidden_size       = {config.get('hidden_size')}\n"
                f"   num_attention_heads = {config.get('num_attention_heads')}\n"
                f"   num_key_value_heads = {config.get('num_key_value_heads')}\n"
                f"   head_dim          = {config.get('head_dim')}\n"
                f"{'─'*70}\n"
            )
    except Exception:
        log_lines.append("⚠️  无法读取 config.json，将从 weight shape 自动推断\n")

    # ── 获取分片索引 ─────────────────────────────
    progress(0.05, desc="读取模型索引...")
    try:
        index_data   = find_index_file(model_id, token)
        shard_headers: dict[str, tuple[dict, int]] = {}

        if index_data:
            weight_map = index_data["weight_map"]
            log_lines.append(
                f"📦 分片模型，共 {len(set(weight_map.values()))} 个 shard\n"
            )
        else:
            sf_files = get_safetensor_files(model_id, token)
            if not sf_files:
                return "❌ 未找到 .safetensors 文件", None
            weight_map = None
            log_lines.append(f"📦 单文件：{sf_files}\n")
    except requests.exceptions.HTTPError as e:
        return _http_error_msg(e, model_id), None

    # ── 探测第一个 shard，识别 Q/K key 命名 ──────
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

    # 识别 Q/K key 命名规则
    q_candidates = [k for k in all_keys if any(
        p in k for p in ["q_proj.weight", "query.weight", "q.weight", "wq.weight"]
    )]
    if not q_candidates:
        sample = "\n".join(all_keys[:30])
        return f"⚠️ 无法识别 Q/K key，前 30 个 key：\n{sample}", None

    sample_q = q_candidates[0]
    if "q_proj"  in sample_q: q_suffix, k_suffix = "self_attn.q_proj.weight", "self_attn.k_proj.weight"
    elif "query" in sample_q: q_suffix, k_suffix = "attention.query.weight",  "attention.key.weight"
    elif "wq"    in sample_q: q_suffix, k_suffix = "attention.wq.weight",     "attention.wk.weight"
    else:
        q_suffix = sample_q.split("layers.0.")[-1]
        k_suffix = q_suffix.replace("q.", "k.")

    log_lines.append(f"🔑 Q suffix：{q_suffix}\n")
    log_lines.append(f"🔑 K suffix：{k_suffix}\n")
    log_lines.append(f"{'═'*70}\n")

    # ── 辅助：查找 key 所在 shard ────────────────
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
    gqa_inferred = False   # 只打印一次 GQA 信息

    for layer_idx in range(int(max_layers)):
        progress(
            0.10 + 0.85 * layer_idx / int(max_layers),
            desc=f"第 {layer_idx} 层..."
        )

        q_key = f"model.layers.{layer_idx}.{q_suffix}"
        k_key = f"model.layers.{layer_idx}.{k_suffix}"

        q_shard = get_shard_for_key(q_key)
        k_shard = get_shard_for_key(k_key)

        if q_shard is None or k_shard is None:
            log_lines.append(f"\nLayer {layer_idx}: Q/K 未找到，分析结束（共 {layer_idx} 层）\n")
            break

        for shard in {q_shard, k_shard}:
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
        except ValueError as e:
            log_lines.append(f"Layer {layer_idx}: ⚠️ 跳过（{e}）\n")
            continue

        if W_q is None or W_k is None:
            log_lines.append(f"Layer {layer_idx}: ⚠️ tensor 为 None，跳过\n")
            continue

        # ── GQA 参数推断（只做一次，后续复用）───
        try:
            n_q_heads, n_kv_heads, d_head = infer_gqa_params(W_q, W_k, config)
        except ValueError as e:
            log_lines.append(f"Layer {layer_idx}: ❌ GQA 推断失败：{e}\n")
            del W_q, W_k
            continue

        if not gqa_inferred:
            group_size = n_q_heads // n_kv_heads
            log_lines.append(
                f"🧠 GQA 结构：n_q_heads={n_q_heads}, "
                f"n_kv_heads={n_kv_heads}, "
                f"group_size={group_size}, "
                f"d_head={d_head}\n"
                f"   W_q shape: {list(W_q.shape)}, "
                f"W_k shape: {list(W_k.shape)}\n"
                f"{'═'*70}\n"
            )
            gqa_inferred = True

        # ── 逐头计算 ────────────────────────────
        records, layer_log = analyze_layer_heads(
            W_q, W_k, layer_idx,
            n_q_heads, n_kv_heads, d_head
        )
        all_records.extend(records)
        log_lines.append(layer_log)

        del W_q, W_k  # 立即释放内存

    # ── 全局汇总统计 ─────────────────────────────
    if all_records:
        df = pd.DataFrame(all_records)

        pearson_vals  = df["Pearson_r"].values
        spearman_vals = df["Spearman_r"].values
        ssr_vals      = df["SSR"].values
        cos_vals      = df["cos_Uq_Uk"].values
        cov_vals      = df["Cov_err"].values

        summary = (
            f"\n{'═'*70}\n"
            f"📊 王氏五定律全局汇总 — {model_id}\n"
            f"{'═'*70}\n"
            f"总分析：{len(df['Layer'].unique())} 层 × "
            f"每层 {df.groupby('Layer').size().iloc[0]} 个 Q 头 "
            f"= {len(all_records)} 条记录\n\n"

            f"【第一定律 — Pearson r（→ 1）】\n"
            f"  Median={np.median(pearson_vals):.6f}  "
            f"Mean={np.mean(pearson_vals):.6f}  "
            f"Min={np.min(pearson_vals):.6f}  "
            f"Max={np.max(pearson_vals):.6f}\n\n"

            f"【第一定律 — Spearman r（→ 1）】\n"
            f"  Median={np.median(spearman_vals):.6f}  "
            f"Mean={np.mean(spearman_vals):.6f}\n\n"

            f"【第二定律 — SSR（→ 0）】\n"
            f"  Median={np.median(ssr_vals):.8f}  "
            f"Mean={np.mean(ssr_vals):.8f}  "
            f"Min={np.min(ssr_vals):.8f}  "
            f"Max={np.max(ssr_vals):.8f}\n\n"

            f"【第四定律 — cos(Uq,Uk) 输出子空间对齐】\n"
            f"  Median={np.median(cos_vals):.6f}  "
            f"Mean={np.mean(cos_vals):.6f}  "
            f"（随机基准 ≈ 1/√d_head）\n\n"

            f"【协方差对齐误差（越小越好）】\n"
            f"  Median={np.median(cov_vals):.6f}  "
            f"Mean={np.mean(cov_vals):.6f}\n"

            f"{'═'*70}\n"
        )
        log_lines.append(summary)

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
    支持 GQA（Grouped Query Attention）：对每个 Q 头分别与其对应 K 头做 SVD 分析。

    | 定律 | 指标 | 理论极值 |
    |------|------|---------|
    | 第一定律 | Pearson r / Spearman r | → 1 |
    | 第二定律 | SSR | → 0 |
    | 第四定律 | cos(Uq, Uk) | ≈ 1/√d_head（随机正交）|

    [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19707844-blue)](https://doi.org/10.5281/zenodo.19707844)
    [![HAL](https://img.shields.io/badge/HAL-hal--05609398-red)](https://hal.science/hal-05609398)
    """)

    with gr.Row():
        with gr.Column(scale=2):
            model_input = gr.Textbox(
                label="HuggingFace 模型 ID",
                placeholder="Qwen/Qwen2.5-14B-Instruct",
                value="Qwen/Qwen2.5-14B-Instruct"
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

        with gr.Column(scale=1):
            gr.Markdown("""
            ### ✅ 推荐模型
            ```
            Qwen/Qwen2.5-7B-Instruct        (GQA 8Q/2K)
            meta-llama/Llama-3.2-1B         (GQA)
            google/gemma-2-2b               (MHA)
            deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
            ```
            ### GQA 典型结构
            | 模型 | Q头 | KV头 | 每组 |
            |------|-----|------|------|
            | Qwen2.5-7B | 28 | 4 | 7 |
            | LLaMA-3-8B | 32 | 8 | 4 |
            | Qwen2.5-14B | 40 | 8 | 5 |
            | Gemma-2-2B | 8 | 4 | 2 |
            """)

    log_output = gr.Textbox(
        label="分析日志（逐头详情）",
        lines=35, max_lines=80
    )

    table_output = gr.Dataframe(
        label="逐头结果表",
        headers=[
            "Layer","KV_head","Q_head",
            "Pearson_r","Spearman_r",
            "Alpha","Alpha_res",
            "cos_Uq_Uk","Cov_err","SSR"
        ]
    )

    analyze_btn.click(
        fn=analyze_model,
        inputs=[model_input, token_input, max_layers_input],
        outputs=[log_output, table_output]
    )

if __name__ == "__main__":
    demo.launch()