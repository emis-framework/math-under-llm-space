# ui/tab_inspect.py
"""
Tab1：模型结构探测
- 读取所有 shard header
- 展示原始 key 结构
- 自动构建 LayerProfile 并展示推断结果
"""

import gradio as gr
import requests
import pandas as pd

from core.fetcher import (
    load_all_shard_headers,
    get_file_url,
    check_quantization,
    http_error_msg,
)
from core.layer_profile import (
    scan_model_structure,
    summarize_structure,
    extract_config_params,
)


def inspect_model(
    model_id: str,
    hf_token: str,
    progress=gr.Progress()
) -> tuple[str, pd.DataFrame]:
    """
    返回 (结构文本日志, 组件概览DataFrame)
    """
    if not model_id.strip():
        return "❌ 请输入模型 ID", None

    token = hf_token.strip() or None
    log   = [f"🔬 结构探测：{model_id}\n{'═'*80}\n"]

    # ── 量化检测 ─────────────────────────────────
    progress(0.05, desc="量化检测...")
    blocked, qmsg = check_quantization(model_id, token)
    log.append(f"【量化检测】\n{qmsg}\n{'─'*80}\n")
    if blocked:
        return "".join(log), None

    # ── config.json ───────────────────────────────
    progress(0.10, desc="读取 config...")
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
                f"📋 config：\n"
                f"   model_type = {config_params.get('model_type')}\n"
                f"   hidden     = {config_params.get('hidden_size')}\n"
                f"   n_heads    = {config_params.get('num_attention_heads')}\n"
                f"   n_kv       = {config_params.get('num_key_value_heads')}\n"
                f"   head_dim   = {config_params.get('head_dim')}\n"
                f"{'─'*80}\n"
            )
    except Exception as e:
        log.append(f"⚠️  config.json 读取失败：{e}\n")

    # ── 读取所有 shard headers ────────────────────
    progress(0.20, desc="读取 shard headers...")
    try:
        all_headers = load_all_shard_headers(model_id, token)
    except requests.exceptions.HTTPError as e:
        return http_error_msg(e, model_id), None
    except Exception as e:
        return "".join(log) + f"❌ 读取失败：{e}\n", None

    total_keys = sum(len(h) for h, _ in all_headers.values())
    log.append(
        f"📦 shard 数：{len(all_headers)}  "
        f"总 key 数：{total_keys}\n"
        f"{'─'*80}\n"
    )

    # ── 扫描层结构 ────────────────────────────────
    progress(0.50, desc="扫描层结构...")
    profiles = scan_model_structure(all_headers, config_params)

    if not profiles:
        # 打印前30个 key 辅助调试
        sample = []
        for h, _ in list(all_headers.values())[:1]:
            sample = list(h.keys())[:30]
        return (
            "".join(log) +
            "⚠️  未发现 Q/K/V 层，前30个 key：\n" +
            "\n".join(sample), None
        )

    # ── 生成结构文本 ──────────────────────────────
    progress(0.80, desc="生成报告...")
    struct_text = summarize_structure(profiles)
    log.append(struct_text)

    # ── 生成概览 DataFrame ────────────────────────
    rows = []
    for (prefix, layer_idx), p in sorted(profiles.items()):
        rows.append({
            "prefix":     prefix,
            "layer":      layer_idx,
            "d_model":    p.d_model,
            "head_dim":   p.head_dim,
            "dim_source": p.head_dim_source,
            "n_q":        p.n_q_heads,
            "n_kv":       p.n_kv_heads,
            "kv_shared":  p.kv_shared,
            "complete":   p.complete,
            "q_shape":    str(p.q.shape) if p.q else "",
            "k_shape":    str(p.k.shape) if p.k else "",
            "v_shape":    str(p.v.shape) if p.v else "K=V",
        })

    df = pd.DataFrame(rows)

    progress(1.0, desc="完成")
    return "".join(log), df


# ─────────────────────────────────────────────
# Tab1 UI 组件
# ─────────────────────────────────────────────

def build_tab_inspect():
    with gr.Tab("🔬 结构探测"):
        gr.Markdown("""
        **第一步：先探测模型结构**
        - 自动识别所有组件（language/vision/audio）
        - 自动推断 head_dim（支持异构层，如 Gemma-4-31B 局部/全局层）
        - 自动检测 K=V 共享层
        - 结果供「分析」Tab 使用
        """)

        with gr.Row():
            with gr.Column(scale=3):
                inspect_model_id = gr.Textbox(
                    label="HuggingFace 模型 ID",
                    placeholder="google/gemma-4-e2b",
                    value="google/gemma-4-e2b"
                )
            with gr.Column(scale=2):
                inspect_token = gr.Textbox(
                    label="HF Access Token（公开模型可留空）",
                    type="password"
                )
            with gr.Column(scale=1):
                inspect_btn = gr.Button(
                    "🔍 探测结构", variant="secondary", size="lg"
                )

        inspect_log = gr.Textbox(
            label="结构探测日志",
            lines=30, max_lines=200
        )
        inspect_table = gr.Dataframe(
            label="层结构概览表",
            headers=[
                "prefix", "layer", "d_model", "head_dim", "dim_source",
                "n_q", "n_kv", "kv_shared", "complete",
                "q_shape", "k_shape", "v_shape"
            ]
        )

        inspect_btn.click(
            fn=inspect_model,
            inputs=[inspect_model_id, inspect_token],
            outputs=[inspect_log, inspect_table]
        )

    return inspect_model_id, inspect_token