# ui/tab_database.py
"""
Tab4: Database Browser
- Model list (Plan A: aggregated by modality)
- Model detail (Plan B: raw components rows, expandable)
- Per-head raw data query (modality + layer_type as two independent filters)
- DB stats
"""

import gradio as gr
import pandas as pd

from db.schema import init_db, get_db_stats
from db.reader import (
    get_analyzed_models,
    get_model_components,
    get_model_summary,
    get_layer_metrics,
    get_resume_status,
)


def load_db_stats() -> str:
    conn  = init_db()
    stats = get_db_stats(conn)
    return (
        f"Database Statistics\n"
        f"{'─'*40}\n"
        f"  Models:            {stats.get('models', 0)}\n"
        f"  Components:        {stats.get('components', 0)}\n"
        f"  Layer-head records:{stats.get('layer_head_metrics', 0)}\n"
        f"  Summary rows:      {stats.get('model_summary', 0)}\n"
        f"  DB size:           {stats.get('db_size_mb', 0)} MB\n"
    )


def load_model_list() -> pd.DataFrame:
    """
    方案A：按 modality 聚合层数
    language_layers 含 standard + global（同一prefix下全部层）
    vision/audio 为 0 时显示 0
    """
    conn = init_db()
    df   = get_analyzed_models(conn)
    if df.empty:
        return pd.DataFrame(columns=[
            "model_id", "model_type", "analyzed_at", "analyze_sec",
            "n_components", "language_layers", "vision_layers", "audio_layers"
        ])
    # vision/audio 为 0 时替换为空字符串，更美观
    for col in ["vision_layers", "audio_layers"]:
        df[col] = df[col].apply(lambda x: "" if x == 0 else x)
    return df


def load_model_detail(
    model_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    返回：
    1. 方案B：原始 components 行（prefix/modality/n_layers/head_dim等）
    2. model_summary 汇总统计
    3. 断点续传状态文本
    """
    if not model_id.strip():
        return pd.DataFrame(), pd.DataFrame(), "Please enter a model ID."

    conn = init_db()
    mid  = model_id.strip()

    # 方案B：原始 components
    comp_df    = get_model_components(conn, mid)

    # 汇总统计
    summary_df = get_model_summary(conn, mid)

    # 断点续传状态
    status_lines = [f"Resume Status: {mid}\n{'─'*50}\n"]
    if not comp_df.empty:
        for pfx in comp_df["prefix"].tolist():
            rs = get_resume_status(conn, mid, pfx)
            status_lines.append(
                f"  [{pfx}]\n"
                f"    Done layers : {rs['total_done']}\n"
                f"    Layer index : {sorted(rs['done_layers'])}\n"
            )
    else:
        status_lines.append("  No data yet.\n")

    return comp_df, summary_df, "".join(status_lines)


def load_layer_data(
    model_id:   str,
    modality:   str,
    layer_type: str,
    start_layer:int,
    end_layer:  int,
) -> tuple[pd.DataFrame, str]:
    """
    逐头原始数据查询
    modality 和 layer_type 两个维度独立过滤
    """
    if not model_id.strip():
        return pd.DataFrame(), "Please enter a model ID."

    conn = init_db()
    mod  = modality   if modality   != "all" else None
    lt   = layer_type if layer_type != "all" else None

    df = get_layer_metrics(
        conn,
        model_id    = model_id.strip(),
        modality    = mod,
        layer_type  = lt,
        start_layer = int(start_layer),
        end_layer   = int(end_layer),
    )

    if df.empty:
        return pd.DataFrame(), (
            f"No data found: model={model_id} "
            f"modality={mod or 'all'} layer_type={lt or 'all'}"
        )

    status = (
        f"✅ {len(df)} records  "
        f"| layers {df['layer'].min()}~{df['layer'].max()}  "
        f"| modality={mod or 'all'}  layer_type={lt or 'all'}"
    )
    return df, status


# ─────────────────────────────────────────────
# Tab4 UI
# ─────────────────────────────────────────────

def build_tab_database():
    with gr.Tab("🗄️ Database"):
        gr.Markdown(
            "## Database Browser\n"
            "View analyzed models, raw per-head data, and resume status.\n\n"
            "> 查看已分析模型、逐头原始数据及断点续传状态。"
        )

        # ── DB Stats ────────────────────────────────────────
        with gr.Row():
            stats_text = gr.Textbox(
                label="Database Statistics",
                value="Click Refresh to load.",
                lines=7,
                interactive=False,
                scale=2,
            )
            refresh_stats_btn = gr.Button(
                "🔄 Refresh Stats", scale=1, variant="secondary"
            )
        refresh_stats_btn.click(fn=load_db_stats, outputs=stats_text)

        gr.Markdown("---")

        # ── Model List（方案A）──────────────────────────────
        gr.Markdown(
            "### Analyzed Models\n"
            "Layers are split by modality. "
            "`language_layers` includes both standard and global layers.\n\n"
            "> 层数按模态拆分。`language_layers` 含 standard 和 global 层。"
        )
        refresh_models_btn = gr.Button(
            "🔄 Refresh Model List", variant="secondary"
        )
        models_table = gr.Dataframe(
            label="Analyzed Models",
            headers=[
                "model_id", "model_type", "analyzed_at", "analyze_sec",
                "n_components", "language_layers", "vision_layers", "audio_layers"
            ],
            interactive=False,
        )
        refresh_models_btn.click(fn=load_model_list, outputs=models_table)

        gr.Markdown("---")

        # ── Model Detail（方案B展开）────────────────────────
        gr.Markdown(
            "### Model Detail & Resume Status\n"
            "Expand raw component rows and check which layers are done.\n\n"
            "> 查看原始组件信息及断点续传进度。"
        )
        with gr.Row():
            detail_model_id = gr.Textbox(
                label="Model ID",
                placeholder="google/gemma-4-e2b",
                scale=3,
            )
            load_detail_btn = gr.Button(
                "📋 Load Detail", variant="secondary", scale=1
            )

        resume_status_text = gr.Textbox(
            label="Resume Status",
            lines=8,
            interactive=False,
        )
        # 方案B：原始 components 行
        components_table = gr.Dataframe(
            label="Components (raw) — prefix / modality / n_layers / head_dim",
            headers=[
                "prefix", "modality", "n_layers",
                "head_dim_min", "head_dim_max",
                "has_kv_shared", "has_global", "d_model"
            ],
            interactive=False,
        )
        summary_table = gr.Dataframe(
            label="Model Summary (all / standard / global)",
            interactive=False,
        )

        load_detail_btn.click(
            fn=load_model_detail,
            inputs=[detail_model_id],
            outputs=[components_table, summary_table, resume_status_text],
        )

        gr.Markdown("---")

        # ── Raw Data Query ──────────────────────────────────
        gr.Markdown(
            "### Per-head Raw Data Query\n"
            "`Modality` and `Layer Type` are two independent filter dimensions.\n\n"
            "> Modality（模态）和 Layer Type（层结构类型）是两个独立过滤维度，可组合使用。"
        )
        with gr.Row():
            raw_model_id = gr.Textbox(
                label="Model ID",
                placeholder="google/gemma-4-e2b",
                scale=2,
            )
            raw_modality = gr.Dropdown(
                label="Modality",
                choices=["all", "language", "vision", "audio"],
                value="language",
                scale=1,
                info="Filter by component modality | 按模态过滤",
            )
            raw_layer_type = gr.Dropdown(
                label="Layer Type",
                choices=["all", "standard", "global"],
                value="all",
                scale=1,
                info=(
                    "standard = normal layers  |  "
                    "global = K=V shared layers (e.g. Gemma global)"
                ),
            )
        with gr.Row():
            raw_start = gr.Number(
                label="Start Layer", value=0,  precision=0, scale=1
            )
            raw_end = gr.Number(
                label="End Layer",   value=10, precision=0, scale=1
            )
            load_raw_btn = gr.Button(
                "🔍 Query Data", variant="secondary", scale=1
            )

        raw_status = gr.Textbox(
            label="Query Status", lines=1, interactive=False
        )
        raw_table = gr.Dataframe(
            label="Per-head Raw Data",
            interactive=False,
            wrap=False,
        )

        load_raw_btn.click(
            fn=load_layer_data,
            inputs=[
                raw_model_id, raw_modality, raw_layer_type,
                raw_start, raw_end
            ],
            outputs=[raw_table, raw_status],
        )