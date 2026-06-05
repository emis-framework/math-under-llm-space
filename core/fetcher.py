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
from core.debug import dprint


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

    torch_dtype, bytes_per_elem = DTYPE_MAP[dtype_str]
    abs_start = 8 + header_size + offsets[0]
    abs_end   = 8 + header_size + offsets[1] - 1

    # ── 调试：打印偏移信息 ────────────────────────
    expected_bytes = offsets[1] - offsets[0]
    expected_elems = 1
    for d in shape:
        expected_elems *= d
    dprint(
        f"[FETCH] {tensor_name}\n"
        f"  shape={shape} dtype={dtype_str}\n"
        f"  data_offsets={offsets}\n"
        f"  abs_start={abs_start} abs_end={abs_end}\n"
        f"  expected_bytes={expected_bytes} "
        f"expected_elems={expected_elems} "
        f"bytes_per_elem={bytes_per_elem}\n"
        f"  check: {expected_elems * bytes_per_elem} == {expected_bytes} "
        f"{'✅' if expected_elems * bytes_per_elem == expected_bytes else '❌ 不匹配!'}\n"
    )

    req_headers = {"Range": f"bytes={abs_start}-{abs_end}"}
    if token:
        req_headers["Authorization"] = f"Bearer {token}"

    r = requests.get(url, headers=req_headers, timeout=120)
    r.raise_for_status()

    # ── 调试：打印实际收到的字节数 ────────────────
    actual_bytes = len(r.content)
    dprint(
        f"  actual_bytes={actual_bytes} "
        f"{'✅' if actual_bytes == expected_bytes else '❌ 字节数不匹配!'}\n"
        f"  前8字节(hex)={r.content[:8].hex()}\n"
    )

    if torch_dtype == torch.bfloat16:
        tensor = torch.frombuffer(
            bytearray(r.content), dtype=torch.int16
        ).view(torch.bfloat16)
    else:
        tensor = torch.frombuffer(bytearray(r.content), dtype=torch_dtype)

    result = tensor.reshape(shape).float()

    # ── 调试：打印结果首行 ────────────────────────
    dprint(f"  result[0,:5]={result[0,:5].tolist()}\n")

    return result


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

# ─────────────────────────────────────────────────────────────────────────────
# 追加到 core/fetcher.py 末尾
# 新增函数：load_tensors_batch()
# 作用：一次 Range Request 读取多个连续 tensor，用于 Pythia 多层 QKV 批量读取
# 依赖：DTYPE_MAP, dprint（已在 fetcher.py 中定义）
# ─────────────────────────────────────────────────────────────────────────────

def load_tensors_batch(
    url: str,
    tensor_names: list,
    header: dict,
    header_size: int,
    token: str = None,
) -> dict:
    """
    合并多个 tensor 的字节范围，发一次 Range Request 批量读取。

    参数：
        url          : safetensors 文件的完整 URL
        tensor_names : 要读取的 tensor key 列表
        header       : read_safetensors_header() 返回的 header dict
        header_size  : header 字节数（read_safetensors_header() 第二个返回值）
        token        : HF Access Token（可选）

    返回：
        {tensor_name: torch.Tensor (float32)}
        不在 header 中的 key 静默跳过。

    注意：
        假设 tensor_names 在文件中字节范围连续（Pythia QKV 层满足此条件）。
        合并范围 = [min_start, max_end]，中间若有空隙也一并读入（通常 < 1KB）。
    """
    import time

    hdrs = {"Authorization": f"Bearer {token}"} if token else {}

    # ── 收集每个 tensor 的绝对字节范围 ──────────────────────────────────────
    infos = []
    for name in tensor_names:
        if name not in header:
            continue
        entry = header[name]
        dtype_str = entry["dtype"]
        if dtype_str not in DTYPE_MAP:
            dprint(f"[BATCH] 跳过未知 dtype={dtype_str}: {name}")
            continue
        start, end = entry["data_offsets"]
        abs_start = 8 + header_size + start
        abs_end   = 8 + header_size + end - 1
        infos.append((name, abs_start, abs_end, entry["shape"], dtype_str))

    if not infos:
        return {}

    # ── 合并成一个连续区间 ────────────────────────────────────────────────────
    total_start = min(i[1] for i in infos)
    total_end   = max(i[2] for i in infos)
    total_mb    = (total_end - total_start + 1) / 1024 / 1024

    dprint(f"[BATCH] {len(infos)} tensors  {total_mb:.1f} MB  "
           f"bytes={total_start}-{total_end}")

    # ── 带重试的单次 Range Request ────────────────────────────────────────────
    blob = None
    for attempt in range(5):
        try:
            r = requests.get(
                url,
                headers={**hdrs, "Range": f"bytes={total_start}-{total_end}"},
                timeout=300,
            )
            r.raise_for_status()
            expected = total_end - total_start + 1
            if len(r.content) != expected:
                dprint(f"[BATCH] 数据不完整 {len(r.content)}/{expected}，"
                       f"重试 {attempt+1}/5")
                time.sleep(3 * (attempt + 1))
                continue
            blob = r.content
            break
        except Exception as e:
            dprint(f"[BATCH] 失败({attempt+1}/5): {e}")
            time.sleep(3 * (attempt + 1))

    if blob is None:
        raise RuntimeError(f"load_tensors_batch 重试 5 次失败: {url}")

    # ── 从 blob 切片还原每个 tensor ───────────────────────────────────────────
    result = {}
    for name, abs_start, abs_end, shape, dtype_str in infos:
        torch_dtype, _ = DTYPE_MAP[dtype_str]
        offset = abs_start - total_start
        size   = abs_end - abs_start + 1
        chunk  = blob[offset: offset + size]

        if torch_dtype == torch.bfloat16:
            t = torch.frombuffer(bytearray(chunk), dtype=torch.int16
                                 ).view(torch.bfloat16)
        else:
            t = torch.frombuffer(bytearray(chunk), dtype=torch_dtype)

        result[name] = t.reshape(shape).float()
        # DEBUG
        print(f"[BATCH_DEBUG] {name} shape={result[name].shape} mean={result[name].mean():.6f} sample={float(result[name][0,0]):.6f}", flush=True)
        dprint(f"[BATCH]   {name} {list(shape)} OK")

    return result