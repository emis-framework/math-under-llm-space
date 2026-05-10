# core/fetcher.py
"""
HTTP Range Request 读取 safetensors 权重
零下载，直接从 HuggingFace 远程读取
"""

import struct
import json
import requests
import torch
from huggingface_hub import list_repo_files

# ─────────────────────────────────────────────
# dtype 映射
# ─────────────────────────────────────────────

DTYPE_MAP = {
    "F32":  (torch.float32, 4),
    "F16":  (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "F64":  (torch.float64, 8),
    "I32":  (torch.int32, 4),
    "I64":  (torch.int64, 8),
    "I8":   (torch.int8, 1),
    "U8":   (torch.uint8, 1),
}
try:
    DTYPE_MAP["F8_E4M3"] = (torch.float8_e4m3fn, 1)
    DTYPE_MAP["F8_E5M2"] = (torch.float8_e5m2, 1)
except AttributeError:
    pass

UNSUPPORTED_SVD_DTYPES = {"I8", "U8", "I32", "I64", "F8_E4M3", "F8_E5M2"}
QUANTIZED_KEY_SIGNATURES = ["qweight", "qzeros", "scales", "g_idx", "packed_weight"]


# ─────────────────────────────────────────────
# URL 工具
# ─────────────────────────────────────────────

def get_file_url(model_id: str, filename: str) -> str:
    return f"https://huggingface.co/{model_id}/resolve/main/{filename}"


def http_error_msg(e: requests.exceptions.HTTPError, model_id: str) -> str:
    code = e.response.status_code
    if code == 401: return "❌ 401 未授权：请填写有效的 HF Access Token"
    if code == 403: return f"❌ 403 禁止访问：请先接受 {model_id} 的使用协议"
    if code == 404: return f"❌ 404 未找到：模型 {model_id} 不存在"
    return f"❌ HTTP {code}：{e}"


# ─────────────────────────────────────────────
# safetensors header 读取
# ─────────────────────────────────────────────

def read_safetensors_header(url: str, token: str = None) -> tuple[dict, int]:
    """读取 safetensors 文件头，返回 (header_dict, header_size)"""
    hdrs = {"Authorization": f"Bearer {token}"} if token else {}

    r = requests.get(url, headers={**hdrs, "Range": "bytes=0-7"}, timeout=30)
    r.raise_for_status()
    header_size = struct.unpack("<Q", r.content)[0]

    r = requests.get(
        url,
        headers={**hdrs, "Range": f"bytes=8-{8 + header_size - 1}"},
        timeout=30
    )
    r.raise_for_status()
    raw = json.loads(r.content)
    raw.pop("__metadata__", None)
    return raw, header_size


def load_tensor_remote(
    url: str,
    tensor_name: str,
    header: dict,
    header_size: int,
    token: str = None
) -> torch.Tensor | None:
    """远程加载单个 tensor，返回 float32"""
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

    if torch_dtype == torch.bfloat16:
        tensor = torch.frombuffer(
            bytearray(r.content), dtype=torch.int16
        ).view(torch.bfloat16)
    else:
        tensor = torch.frombuffer(bytearray(r.content), dtype=torch_dtype)

    return tensor.reshape(shape).float()


# ─────────────────────────────────────────────
# 文件列表
# ─────────────────────────────────────────────

def get_safetensor_files(model_id: str, token: str = None) -> list[str]:
    kwargs = {"token": token} if token else {}
    return sorted(
        f for f in list_repo_files(model_id, **kwargs)
        if f.endswith(".safetensors")
    )


def find_index_file(model_id: str, token: str = None) -> dict | None:
    url = f"https://huggingface.co/{model_id}/resolve/main/model.safetensors.index.json"
    hdrs = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(url, headers=hdrs, timeout=15)
    return r.json() if r.status_code == 200 else None


def get_all_shard_files(model_id: str, token: str = None) -> list[str]:
    """获取所有 shard 文件名列表"""
    index = find_index_file(model_id, token)
    if index:
        return sorted(set(index["weight_map"].values()))
    return get_safetensor_files(model_id, token)


def load_all_shard_headers(
    model_id: str,
    token: str = None
) -> dict[str, tuple[dict, int]]:
    """
    读取所有 shard 的 header
    返回：{ shard_filename: (header_dict, header_size) }
    """
    shard_files = get_all_shard_files(model_id, token)
    result = {}
    for sf in shard_files:
        url = get_file_url(model_id, sf)
        h, hs = read_safetensors_header(url, token)
        result[sf] = (h, hs)
    return result


# ─────────────────────────────────────────────
# 量化检测
# ─────────────────────────────────────────────

def check_quantization(model_id: str, token: str = None) -> tuple[bool, str]:
    """
    三重量化检测
    返回 (is_blocked, message)
    """
    hdrs = {"Authorization": f"Bearer {token}"} if token else {}
    warnings = []

    # 检测1：config.json
    try:
        r = requests.get(
            f"https://huggingface.co/{model_id}/resolve/main/config.json",
            headers=hdrs, timeout=15
        )
        if r.status_code == 200:
            cfg  = r.json()
            qcfg = cfg.get("quantization_config", {}) or {}
            qt   = (
                qcfg.get("quant_type", "") or
                qcfg.get("quant_method", "") or
                cfg.get("quantization", "")
            ).lower()
            if "gptq" in qt:
                return True, f"❌ GPTQ {qcfg.get('bits','?')}bit，请用原始 BF16 版本。"
            if "awq" in qt:
                return True, "❌ AWQ 量化，请用原始 BF16 版本。"
            if "bitsandbytes" in qt or "bnb" in qt:
                warnings.append("⚠️  bitsandbytes 量化，结果可能失真")
    except Exception:
        warnings.append("⚠️  无法读取 config.json")

    # 检测2：模型名关键词
    for kw in ["gptq", "awq", "gguf"]:
        if kw in model_id.lower():
            return True, f"❌ 模型名含 '{kw.upper()}'，请使用原始 BF16 版本。"

    # 检测3：文件级别
    try:
        all_files = list(list_repo_files(model_id, token=token))
        if any(f.endswith(".gguf") for f in all_files):
            return True, "❌ 检测到 .gguf 文件，不支持该格式。"
        if not any(f.endswith(".safetensors") for f in all_files):
            return True, "❌ 未找到 .safetensors 文件。"
    except Exception as e:
        warnings.append(f"⚠️  文件列表检测失败：{e}")

    # 检测4：header 内容
    try:
        shard_files = get_all_shard_files(model_id, token)
        hdr, _ = read_safetensors_header(
            get_file_url(model_id, shard_files[0]), token
        )
        bad = [k for k in hdr if any(s in k for s in QUANTIZED_KEY_SIGNATURES)]
        if bad:
            return True, f"❌ 量化 key：{bad[:3]}"
        good = {hdr[k].get("dtype", "") for k in list(hdr)[:20]} - UNSUPPORTED_SVD_DTYPES
        if good:
            warnings.append(f"✅ 权重格式：{good}")
    except Exception as e:
        warnings.append(f"⚠️  header 检测失败：{e}")

    return False, "\n".join(warnings) if warnings else "✅ 未检测到量化，可以正常分析"