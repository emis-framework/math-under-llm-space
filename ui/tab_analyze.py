# ui/tab_analyze.py
"""
Tab2：分析单个模型
- 使用 LayerProfile 自动推断结构
- start_layer / end_layer 按原始层号过滤
- 逐头计算五定律全指标
- 结果写入 SQLite（Phase 2 完成后接入）
"""

import gradio as gr
import requests
import pandas as pd
import numpy as np

from core.fetcher import (
    load_all_shard_headers,
    load_tensor_remote,
    get_file_url,
    check_quantization,
    http_error_msg,
)
from core.layer_profile import (
    scan_model_structure,
    extract_config_params,
)
from core.metrics import analyze_layer, summarize_records


SIDEBAR_MD = """
### ✅ 推荐模型
google/gemma-4-e2b  
google/gemma-4-e4b-it  
google/gemma-4-31b-it  
Qwen/Qwen2.5-14B-Instruct  
deepseek-ai/DeepSeek-R1-Distill-Qwen-14B  
meta-llama/Meta-Llama-3-8B  


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

### 示例：Gemma-4-31B
| 组件 | 层范围 |
|------|--------|
| language(局部层) | 0~59 |
| language(全局层) | 5,11,17...59 |
| vision_tower | 0~26 |
"""


def run_analysis(
    model_id:    str,
    hf_token:    str,
    start_layer: int,
    end_layer:   int,
    progress=gr.Progress()
) -> tuple[str, pd.DataFrame]:

    if not model_id.strip():
        return "❌ 请输入模型 ID", None

    token   = hf_token.strip() or None
    start_l = int(start_layer)
    end_l   = int(end_layer)
    log     = [
        f"🔍 分析：{model_id}  层 {start_l}~{end_l}\n"
        f"{'═'*80}\n"
    ]
    all_records: list[dict] = []

    # ── 量化检测 ─────────────────────────────────
    progress(0.02, desc="量化检测...")
    blocked, qmsg = check_quantization(model_id, token)
    log.append(f"【量化检测】\n{qmsg}\n{'─'*80}\n")
    if blocked:
        return "".join(log), None

    # ── config.json ───────────────────────────────
    progress(0.05, desc="读取 config...")
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
                f"head_dim={config_params.get('head_dim')}\n"
                f"{'─'*80}\n"
            )
    except Exception:
        log.append("⚠️  无法读取 config.json\n")

    # ── 读取所有 shard headers ────────────────────
    progress(0.08, desc="读取 shard headers...")
    try:
        all_headers = load_all_shard_headers(model_id, token)
    except requests.exceptions.HTTPError as e:
        return http_error_msg(e, model_id), None
    except Exception as e:
        return "".join(log) + f"❌ 读取失败：{e}\n", None

    log.append(
        f"📦 shard 数：{len(all_headers)}  "
        f"总 key：{sum(len(h) for h,_ in all_headers.values())}\n"
    )

    # ── 扫描层结构 ────────────────────────────────
    progress(0.12, desc="扫描层结构...")
    profiles = scan_model_structure(all_headers, config_params)

    if not profiles:
        return "".join(log) + "⚠️ 未发现任何 Q/K/V 层\n", None

    # ── 按原始层号过滤 ────────────────────────────
    filtered = {
        (pfx, idx): prof
        for (pfx, idx), prof in profiles.items()
        if start_l <= idx <= end_l and prof.complete
    }

    if not filtered:
        # 打印实际层号供参考
        by_pfx: dict[str, list] = {}
        for (pfx, idx) in profiles:
            by_pfx.setdefault(pfx, []).append(idx)
        info = "\n".join(
            f"  '{p}': {sorted(v)}"
            for p, v in sorted(by_pfx.items())
        )
        return (
            "".join(log) +
            f"⚠️ 层 {start_l}~{end_l} 内无完整层\n"
            f"实际层号：\n{info}\n", None
        )

    # 打印将分析的层
    by_pfx2: dict[str, list] = {}
    for (pfx, idx) in filtered:
        by_pfx2.setdefault(pfx, []).append(idx)
    log.append(f"📐 将分析：\n")
    for pfx, idxs in sorted(by_pfx2.items()):
        log.append(f"  '{pfx}' → 层 {sorted(idxs)}\n")
    log.append(f"{'═'*80}\n")

    # ── 逐层分析 ─────────────────────────────────
    sorted_items = sorted(filtered.items(), key=lambda x: (x[0][0], x[0][1]))
    total = len(sorted_items)

    for i, ((pfx, idx), prof) in enumerate(sorted_items):
        progress(
            0.15 + 0.80 * i / max(total, 1),
            desc=f"{pfx.split('.')[-2] if '.' in pfx else pfx} L{idx}..."
        )

        # 加载 Q/K/V
        try:
            q_url = get_file_url(model_id, prof.q.shard)
            k_url = get_file_url(model_id, prof.k.shard)

            q_hdr, q_hs = all_headers[prof.q.shard]
            k_hdr, k_hs = all_headers[prof.k.shard]

            W_q = load_tensor_remote(q_url, prof.q.key, q_hdr, q_hs, token)
            W_k = load_tensor_remote(k_url, prof.k.key, k_hdr, k_hs, token)

            if prof.kv_shared:
                # K=V 共享：直接复用
                W_v = W_k.clone()
            else:
                v_url = get_file_url(model_id, prof.v.shard)
                v_hdr, v_hs = all_headers[prof.v.shard]
                W_v = load_tensor_remote(v_url, prof.v.key, v_hdr, v_hs, token)

        except Exception as e:
            log.append(f"[{pfx}] Layer {idx}: ❌ 加载失败：{e}\n")
            continue

        if W_q is None or W_k is None or W_v is None:
            log.append(f"[{pfx}] Layer {idx}: ⚠️ tensor 为 None\n")
            continue

        # 计算五定律
        try:
            records, layer_log = analyze_layer(W_q, W_k, W_v, prof)
            all_records.extend(records)
            log.append(layer_log)
        except Exception as e:
            log.append(f"[{pfx}] Layer {idx}: ❌ 计算失败：{e}\n")
        finally:
            del W_q, W_k, W_v

    # ── 汇总 ─────────────────────────────────────
    if not all_records:
        return "".join(log) + "\n❌ 未获得任何有效结果\n", None

    summary = summarize_records(all_records, model_id)
    log.append(summary)

    df = pd.DataFrame(all_records)
    return "".join(log), df


# ─────────────────────────────────────────────
# Tab2 UI 组件
# ─────────────────────────────────────────────

def build_tab_analyze():
    with gr.Tab("📊 分析"):
        gr.Markdown("""
        **第二步：选择层范围，计算王氏五定律全指标**
        层号 = safetensors key 中 `layers.{N}` 的原始 N，K=V 共享层自动处理。
        """)

        with gr.Row():
            with gr.Column(scale=3):
                model_id_input = gr.Textbox(
                    label="HuggingFace 模型 ID",
                    placeholder="google/gemma-4-e2b",
                    value="google/gemma-4-e2b"
                )
                token_input = gr.Textbox(
                    label="HF Access Token（公开模型可留空）",
                    type="password"
                )
                with gr.Row():
                    start_input = gr.Number(
                        label="起始层号（含）",
                        value=0, minimum=0, maximum=9999, precision=0
                    )
                    end_input = gr.Number(
                        label="结束层号（含）",
                        value=5, minimum=0, maximum=9999, precision=0
                    )
                analyze_btn = gr.Button("🚀 开始分析", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown(SIDEBAR_MD)

        analyze_log = gr.Textbox(
            label="分析日志（逐头详情）",
            lines=35, max_lines=300
        )
        analyze_table = gr.Dataframe(
            label="逐头全指标结果表",
            headers=[
                "prefix", "layer", "kv_head", "q_head", "kv_shared",
                "pearson_QK", "spearman_QK", "pearson_QV", "pearson_KV",
                "ssr_QK", "ssr_QV", "ssr_KV",
                "cosU_QK", "cosU_QV", "cosU_KV",
                "cosV_QK", "cosV_QV", "cosV_KV",
                "alpha_QK", "alpha_QV", "alpha_KV",
                "alpha_res_QK", "alpha_res_QV", "alpha_res_KV",
                "sigma_max_Q", "sigma_min_Q",
                "sigma_max_K", "sigma_min_K",
                "sigma_max_V", "sigma_min_V",
                "cond_Q", "cond_K", "cond_V",
                "head_dim", "d_model", "n_q_heads", "n_kv_heads",
            ]
        )

        analyze_btn.click(
            fn=run_analysis,
            inputs=[model_id_input, token_input, start_input, end_input],
            outputs=[analyze_log, analyze_table]
        )

    return model_id_input, token_input