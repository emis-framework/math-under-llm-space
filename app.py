import gradio as gr
import requests
import struct
import json
import numpy as np
import torch
from scipy import stats
from huggingface_hub import list_repo_files

# ─────────────────────────────────────────────
# 核心：HTTP Range Request 读取单个 tensor
# ─────────────────────────────────────────────

DTYPE_MAP = {
    "F32":  (torch.float32,  4),
    "F16":  (torch.float16,  2),
    "BF16": (torch.bfloat16, 2),
    "F64":  (torch.float64,  8),
    "I32":  (torch.int32,    4),
    "I64":  (torch.int64,    8),
}

def get_file_url(model_id: str, filename: str) -> str:
    """生成 HuggingFace 直链 URL"""
    return f"https://huggingface.co/{model_id}/resolve/main/{filename}"

def read_safetensors_header(url: str, token: str = None) -> dict:
    """
    只读取 safetensors 文件头部（几KB），
    获取所有 tensor 的 offset、dtype、shape
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    # 第一步：读前 8 bytes → 获取 header_size
    r = requests.get(url, headers={**headers, "Range": "bytes=0-7"}, timeout=30)
    r.raise_for_status()
    header_size = struct.unpack("<Q", r.content)[0]
    
    # 第二步：读 header JSON
    r = requests.get(
        url,
        headers={**headers, "Range": f"bytes=8-{8 + header_size - 1}"},
        timeout=30
    )
    r.raise_for_status()
    return json.loads(r.content), header_size

def load_tensor_remote(url: str, tensor_name: str, header: dict,
                        header_size: int, token: str = None) -> torch.Tensor:
    """
    只下载指定 tensor 的字节数据（Range Request），
    完全不缓存整个文件
    """
    if tensor_name not in header:
        return None
    
    info = header[tensor_name]
    dtype_str = info["dtype"]
    shape = info["shape"]
    offsets = info["data_offsets"]  # [start, end] 相对于数据区
    
    if dtype_str not in DTYPE_MAP:
        raise ValueError(f"不支持的 dtype: {dtype_str}")
    
    torch_dtype, _ = DTYPE_MAP[dtype_str]
    
    # 计算文件中的绝对字节位置
    # safetensors 文件布局：8字节(header_size) + header_size字节(header) + 数据区
    abs_start = 8 + header_size + offsets[0]
    abs_end   = 8 + header_size + offsets[1] - 1
    
    req_headers = {"Range": f"bytes={abs_start}-{abs_end}"}
    if token:
        req_headers["Authorization"] = f"Bearer {token}"
    
    r = requests.get(url, headers=req_headers, timeout=120)
    r.raise_for_status()
    
    # 转换为 tensor（BF16 需特殊处理）
    raw = r.content
    if torch_dtype == torch.bfloat16:
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.int16).view(torch.bfloat16)
    else:
        tensor = torch.frombuffer(bytearray(raw), dtype=torch_dtype)
    
    return tensor.reshape(shape).float()  # 统一转 float32 做 SVD

# ─────────────────────────────────────────────
# 查找模型的 safetensors 文件列表
# ─────────────────────────────────────────────

def get_safetensor_files(model_id: str, token: str = None) -> list:
    """列出模型 repo 中的所有 .safetensors 文件"""
    kwargs = {"token": token} if token else {}
    all_files = list(list_repo_files(model_id, **kwargs))
    sf_files = [f for f in all_files if f.endswith(".safetensors")]
    return sorted(sf_files)

def find_index_file(model_id: str, token: str = None):
    """检查是否有 model.safetensors.index.json（分片模型）"""
    url = f"https://huggingface.co/{model_id}/resolve/main/model.safetensors.index.json"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(url, headers=headers, timeout=15)
    if r.status_code == 200:
        return r.json()
    return None

# ─────────────────────────────────────────────
# 王氏五定律计算核心
# ─────────────────────────────────────────────

def compute_svd_metrics(W_q: torch.Tensor, W_k: torch.Tensor):
    """对一层的 Q/K 矩阵计算 SVD，返回 Pearson r 和 SSR"""
    _, sq, _ = torch.linalg.svd(W_q, full_matrices=False)
    _, sk, _ = torch.linalg.svd(W_k, full_matrices=False)
    
    sq = sq.numpy()
    sk = sk.numpy()
    
    # 第一定律：Pearson r
    r, _ = stats.pearsonr(sq, sk)
    
    # 第二定律：SSR（谱形状残差）
    sq_norm = sq / (np.linalg.norm(sq) + 1e-10)
    sk_norm = sk / (np.linalg.norm(sk) + 1e-10)
    ssr = np.mean(np.abs(sq_norm - sk_norm))
    
    return float(r), float(ssr)

# ─────────────────────────────────────────────
# 主分析函数：扫描所有层
# ─────────────────────────────────────────────

def analyze_model(model_id: str, hf_token: str, max_layers: int, progress=gr.Progress()):
    """
    主函数：
    1. 找到所有 safetensors 文件
    2. 逐层用 Range Request 读取 Q/K tensor
    3. 计算 SVD，输出 Pearson r 和 SSR
    """
    if not model_id.strip():
        return "❌ 请输入模型 ID，例如：Qwen/Qwen2.5-14B-Instruct", None
    
    token = hf_token.strip() if hf_token.strip() else None
    results = []
    log_lines = [f"🔍 分析模型：{model_id}\n"]
    
    try:
        # Step 1: 获取 tensor 名称 → 文件的映射
        progress(0.05, desc="读取模型索引...")
        
        # 尝试分片索引
        index_data = find_index_file(model_id, token)
        
        # 收集所有 shard 的 header
        shard_headers = {}  # filename → (header_dict, header_size)
        
        if index_data:
            weight_map = index_data.get("weight_map", {})
            log_lines.append(f"📦 分片模型，共 {len(set(weight_map.values()))} 个 shard 文件\n")
        else:
            # 单文件模型
            sf_files = get_safetensor_files(model_id, token)
            if not sf_files:
                return "❌ 未找到 .safetensors 文件，请检查模型 ID 或 token", None
            weight_map = {}
            for f in sf_files:
                log_lines.append(f"📦 单文件模型：{f}\n")
        
        # Step 2: 检测层数和 Q/K key 命名规则
        progress(0.1, desc="检测层结构...")
        
        # 先读第一个 shard 来探测 key 命名
        first_shard = None
        if index_data:
            first_shard = list(set(index_data["weight_map"].values()))[0]
        else:
            first_shard = sf_files[0]
        
        first_url = get_file_url(model_id, first_shard)
        first_header, first_hsize = read_safetensors_header(first_url, token)
        shard_headers[first_shard] = (first_header, first_hsize)
        
        # 自动检测 Q/K key 命名模式
        all_keys = list(first_header.keys())
        q_keys_sample = [k for k in all_keys if any(
            p in k for p in ["q_proj.weight", "query.weight", "q.weight", "wq.weight"]
        )]
        
        if not q_keys_sample:
            # 展示所有 key 供用户参考
            sample_keys = "\n".join(all_keys[:30])
            return f"⚠️ 无法自动识别 Q/K key，前30个 key：\n{sample_keys}", None
        
        # 判断命名模式
        sample_q = q_keys_sample[0]
        if "q_proj" in sample_q:
            q_pattern = "self_attn.q_proj.weight"
            k_pattern = "self_attn.k_proj.weight"
        elif "query" in sample_q:
            q_pattern = "attention.query.weight"
            k_pattern = "attention.key.weight"
        else:
            q_pattern = sample_q.split(".")[-3] + ".q.weight"
            k_pattern = sample_q.split(".")[-3] + ".k.weight"
        
        log_lines.append(f"🔑 Q key 模式：{q_pattern}\n")
        log_lines.append(f"🔑 K key 模式：{k_pattern}\n\n")
        
        # Step 3: 逐层计算
        max_layers = int(max_layers)
        layer_idx = 0
        pearson_list = []
        ssr_list = []
        
        while layer_idx < max_layers:
            progress(0.1 + 0.85 * layer_idx / max_layers,
                     desc=f"处理第 {layer_idx} 层...")
            
            # 构建 key 名称（支持常见命名方式）
            q_key = f"model.layers.{layer_idx}.{q_pattern}"
            k_key = f"model.layers.{layer_idx}.{k_pattern}"
            
            # 找到对应的 shard
            def get_shard_for_key(key):
                if index_data:
                    return index_data["weight_map"].get(key)
                else:
                    # 遍历所有 shard header 查找
                    for sf in sf_files:
                        if sf not in shard_headers:
                            url = get_file_url(model_id, sf)
                            h, hs = read_safetensors_header(url, token)
                            shard_headers[sf] = (h, hs)
                        h, _ = shard_headers[sf]
                        if key in h:
                            return sf
                return None
            
            q_shard = get_shard_for_key(q_key)
            k_shard = get_shard_for_key(k_key)
            
            if q_shard is None or k_shard is None:
                log_lines.append(f"Layer {layer_idx}: ⚠️ 未找到 Q/K，停止\n")
                break
            
            # 加载对应 shard 的 header
            for shard in [q_shard, k_shard]:
                if shard not in shard_headers:
                    url = get_file_url(model_id, shard)
                    h, hs = read_safetensors_header(url, token)
                    shard_headers[shard] = (h, hs)
            
            # Range Request 只下载 Q 和 K tensor
            q_url = get_file_url(model_id, q_shard)
            k_url = get_file_url(model_id, k_shard)
            
            q_header, q_hsize = shard_headers[q_shard]
            k_header, k_hsize = shard_headers[k_shard]
            
            W_q = load_tensor_remote(q_url, q_key, q_header, q_hsize, token)
            W_k = load_tensor_remote(k_url, k_key, k_header, k_hsize, token)
            
            if W_q is None or W_k is None:
                log_lines.append(f"Layer {layer_idx}: ⚠️ tensor 读取失败\n")
                break
            
            r, ssr = compute_svd_metrics(W_q, W_k)
            pearson_list.append(r)
            ssr_list.append(ssr)
            results.append({
                "Layer": layer_idx,
                "Pearson_r": round(r, 6),
                "SSR": round(ssr, 6)
            })
            
            log_lines.append(
                f"Layer {layer_idx:3d} | Q shape: {list(W_q.shape)} "
                f"| Pearson r = {r:.4f} | SSR = {ssr:.6f}\n"
            )
            
            # 释放内存
            del W_q, W_k
            layer_idx += 1
        
        # Step 4: 汇总统计
        if pearson_list:
            summary = (
                f"\n{'='*50}\n"
                f"📊 王氏五定律分析结果 — {model_id}\n"
                f"{'='*50}\n"
                f"总层数分析: {len(pearson_list)} 层\n\n"
                f"【第一定律 - 谱线性对齐 Pearson r】\n"
                f"  Median: {np.median(pearson_list):.4f}  "
                f"  Mean: {np.mean(pearson_list):.4f}\n"
                f"  Min: {np.min(pearson_list):.4f}  "
                f"  Max: {np.max(pearson_list):.4f}\n\n"
                f"【第二定律 - 谱形状保真 SSR】\n"
                f"  Median: {np.median(ssr_list):.6f}  "
                f"  Mean: {np.mean(ssr_list):.6f}\n"
                f"  Min: {np.min(ssr_list):.6f}  "
                f"  Max: {np.max(ssr_list):.6f}\n\n"
                f"⚡ 理论值：Pearson r → 1，SSR → 0\n"
                f"{'='*50}\n"
            )
            log_lines.append(summary)
        
        # 生成图表数据
        import pandas as pd
        df = pd.DataFrame(results)
        
        return "".join(log_lines), df
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return "❌ 401 未授权：该模型需要 HF Token，请填写 Access Token", None
        elif e.response.status_code == 403:
            return "❌ 403 禁止访问：请确认已在 HF 接受该模型的使用协议", None
        elif e.response.status_code == 404:
            return f"❌ 404 未找到：模型 {model_id} 不存在或文件路径错误", None
        else:
            return f"❌ HTTP 错误：{e}", None
    except Exception as e:
        return f"❌ 错误：{str(e)}", None

# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────

with gr.Blocks(title="Wang's Five Laws — LLM Spectral Analyzer") as demo:
    gr.Markdown("""
    # 🔬 Wang's Five Laws — LLM Spectral Analyzer
    **Mathematical Foundations of Large Language Models (MF-LLM)**
    
    通过 HTTP Range Request 直接读取 HuggingFace 模型的 Q/K 权重 tensor，
    **无需下载完整模型**，计算王氏五定律的核心指标：
    - 📐 **第一定律**：Pearson r → 1（谱线性对齐）
    - 📏 **第二定律**：SSR → 0（谱形状保真）
    
    [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19707844-blue)](https://doi.org/10.5281/zenodo.19707844)
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            model_input = gr.Textbox(
                label="HuggingFace 模型 ID",
                placeholder="例如：Qwen/Qwen2.5-14B-Instruct",
                value="Qwen/Qwen2.5-14B-Instruct"
            )
            token_input = gr.Textbox(
                label="HF Access Token（公开模型可留空）",
                placeholder="hf_xxxxxxxxxxxx",
                type="password"
            )
            max_layers_input = gr.Slider(
                label="最大分析层数",
                minimum=1, maximum=100, value=32, step=1
            )
            analyze_btn = gr.Button("🚀 开始分析", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("""
            ### 💡 快速测试模型
            - `meta-llama/Llama-3.2-1B`
            - `Qwen/Qwen2.5-7B-Instruct`
            - `google/gemma-2-2b`
            - `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
            
            ### ⚙️ 运行环境
            - CPU Only（无 GPU）
            - 每层约 5-30 秒（取决于网速和矩阵大小）
            - **零缓存**：仅下载 Q/K tensor 字节
            """)
    
    with gr.Row():
        log_output = gr.Textbox(
            label="分析日志",
            lines=25,
            max_lines=50
        )
    
    with gr.Row():
        table_output = gr.Dataframe(
            label="逐层结果（Pearson r & SSR）",
            headers=["Layer", "Pearson_r", "SSR"]
        )
    
    analyze_btn.click(
        fn=analyze_model,
        inputs=[model_input, token_input, max_layers_input],
        outputs=[log_output, table_output]
    )

if __name__ == "__main__":
    demo.launch()