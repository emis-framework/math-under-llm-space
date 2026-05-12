# ui/tab_database.py
"""
Tab4：数据库浏览
- 查看已分析模型列表
- 查看某模型的逐层原始数据
- 数据库统计信息
"""

import gradio as gr
import pandas as pd

from db.schema import init_db, get_db_stats
from db.reader import (
    get_analyzed_models,
    get_model_summary,
    get_layer_metrics,
    get_resume_status,
)


def load_db_stats() -> str:
    """获取数据库统计信息"""
    conn  = init_db()
    stats = get_db_stats(conn)
    return (
        f"📊 数据库统计\n"
        f"{'─'*40}\n"
        f"  模型数：     {stats.get('models', 0)}\n"
        f"  组件数：     {stats.get('components', 0)}\n"
        f"  层头记录数： {stats.get('layer_head_metrics', 0)}\n"
        f"  汇总行数：   {stats.get('model_summary', 0)}\n"
        f"  数据库大小： {stats.get('db_size_mb', 0)} MB\n"
    )


def load_model_list() -> pd.DataFrame:
    """加载已分析模型列表"""
    conn = init_db()
    df   = get_analyzed_models(conn)
    if df.empty:
        return pd.DataFrame(
            columns=["model_id", "model_type", "analyzed_at",
                     "analyze_sec", "n_components", "total_layers"]
        )
    return df


def load_model_detail(model_id: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    加载模型详情
    返回 (summary_df, 断点续传状态文本)
    """
    if not model_id.strip():
        return pd.DataFrame(), pd.DataFrame(), "请输入模型 ID"

    conn = init_db()

    # 汇总统计
    summary_df = get_model_summary(conn, model_id.strip())

    # 断点续传状态（按前缀）
    status_lines = [f"📍 断点续传状态：{model_id}\n{'─'*50}\n"]
    if not summary_df.empty:
        for pfx in summary_df["prefix"].unique():
            rs = get_resume_status(conn, model_id.strip(), pfx)
            status_lines.append(
                f"  [{pfx}]\n"
                f"    已完成层数：{rs['total_done']}\n"
                f"    层号：{sorted(rs['done_layers'])}\n"
            )
    else:
        status_lines.append("  暂无数据\n")

    return summary_df, "".join(status_lines)


def load_layer_data(
    model_id:    str,
    prefix:      str,
    layer_type:  str,
    start_layer: int,
    end_layer:   int,
) -> tuple[pd.DataFrame, str]:
    """加载逐头原始数据"""
    if not model_id.strip():
        return pd.DataFrame(), "请输入模型 ID"

    conn = init_db()
    lt   = layer_type if layer_type != "all" else None
    pfx  = prefix.strip() or None

    df = get_layer_metrics(
        conn,
        model_id    = model_id.strip(),
        prefix      = pfx,
        layer_type  = lt,
        start_layer = int(start_layer),
        end_layer   = int(end_layer),
    )

    if df.empty:
        return pd.DataFrame(), f"⚠️ 无数据：model={model_id} prefix={pfx} layer_type={lt}"

    status = (
        f"✅ {len(df)} 条记录  "
        f"| 层 {df['layer'].min()}~{df['layer'].max()}  "
        f"| prefix={pfx or '全部'}"
    )
    return df, status


# ─────────────────────────────────────────────
# Tab4 UI
# ─────────────────────────────────────────────

def build_tab_database():
    with gr.Tab("🗄️ 数据库"):
        gr.Markdown("## 数据库浏览  \n查看已分析模型的原始数据和汇总统计。")

        # ── 数据库统计 ──────────────────────────
        with gr.Row():
            stats_text = gr.Textbox(
                label="数据库统计",
                value="点击刷新",
                lines=7,
                interactive=False,
                scale=2,
            )
            refresh_stats_btn = gr.Button(
                "🔄 刷新统计", scale=1, variant="secondary"
            )

        refresh_stats_btn.click(
            fn=load_db_stats,
            outputs=stats_text,
        )

        gr.Markdown("---")

        # ── 已分析模型列表 ──────────────────────
        gr.Markdown("### 已分析模型")
        with gr.Row():
            refresh_models_btn = gr.Button(
                "🔄 刷新模型列表", variant="secondary"
            )

        models_table = gr.Dataframe(
            label="已分析模型",
            interactive=False,
        )

        refresh_models_btn.click(
            fn=load_model_list,
            outputs=models_table,
        )

        gr.Markdown("---")

        # ── 模型详情 ────────────────────────────
        gr.Markdown("### 模型详情 & 断点续传状态")
        with gr.Row():
            detail_model_id = gr.Textbox(
                label="模型 ID",
                placeholder="google/gemma-4-e2b",
                scale=3,
            )
            load_detail_btn = gr.Button(
                "📋 查看详情", variant="secondary", scale=1
            )

        resume_status_text = gr.Textbox(
            label="断点续传状态",
            lines=8,
            interactive=False,
        )
        summary_table = gr.Dataframe(
            label="模型汇总统计（all/standard/global 三行）",
            interactive=False,
        )

        load_detail_btn.click(
            fn=load_model_detail,
            inputs=[detail_model_id],
            outputs=[summary_table, resume_status_text],
        )

        gr.Markdown("---")

        # ── 逐头原始数据 ────────────────────────
        gr.Markdown("### 逐头原始数据查询")
        with gr.Row():
            raw_model_id = gr.Textbox(
                label="模型 ID",
                placeholder="google/gemma-4-e2b",
                scale=2,
            )
            raw_prefix = gr.Textbox(
                label="组件前缀（留空=全部）",
                placeholder="model.language_model.",
                scale=2,
            )
            raw_layer_type = gr.Dropdown(
                label="层类型",
                choices=["all", "standard", "global"],
                value="all",
                scale=1,
            )
        with gr.Row():
            raw_start = gr.Number(
                label="起始层号", value=0, precision=0, scale=1
            )
            raw_end = gr.Number(
                label="结束层号", value=10, precision=0, scale=1
            )
            load_raw_btn = gr.Button(
                "🔍 查询数据", variant="secondary", scale=1
            )

        raw_status = gr.Textbox(
            label="查询状态", lines=1, interactive=False
        )
        raw_table = gr.Dataframe(
            label="逐头原始数据",
            interactive=False,
            wrap=False,
        )

        load_raw_btn.click(
            fn=load_layer_data,
            inputs=[raw_model_id, raw_prefix, raw_layer_type,
                    raw_start, raw_end],
            outputs=[raw_table, raw_status],
        )