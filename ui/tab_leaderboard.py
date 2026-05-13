# ui/tab_leaderboard.py
"""
Tab3: Wang's Five Laws Leaderboard
- Ranked by wang_score (= 1 − median SSR_QK, standard layers only)
- Filter by modality (default: language)
- Filter by layer_type (default: standard)
"""

import gradio as gr
import pandas as pd
import numpy as np

from db.schema import init_db
from db.reader import get_leaderboard


def _format_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["model_name"] = df["model_id"].apply(
        lambda x: x.split("/")[-1] if "/" in x else x
    )
    df["wang_score_pct"] = df["wang_score"].apply(
        lambda x: f"{x*100:.3f}" if pd.notna(x) else "N/A"
    )
    for col in ["median_pearson_QK", "median_ssr_QK", "mean_ssr_QK"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: f"{x:.6f}" if pd.notna(x) else "N/A"
            )

    display_cols = [
        "model_name", "modality", "layer_type",
        "wang_score_pct",
        "median_pearson_QK", "median_ssr_QK", "mean_ssr_QK",
        "median_cosU_QK", "median_cosU_QV", "median_cosV_QK",
        "n_layers", "n_records", "model_id",
    ]
    existing = [c for c in display_cols if c in df.columns]
    return df[existing]


def load_leaderboard(
    modality:   str,
    layer_type: str,
) -> tuple[pd.DataFrame, str]:
    conn = init_db()
    lt   = layer_type if layer_type != "all" else "standard"
    mod  = modality

    df = get_leaderboard(conn, modality=mod, layer_type=lt, limit=100)

    if df.empty:
        return pd.DataFrame(), (
            f"No data yet. Please analyze at least one model first.\n"
            f"(modality='{mod}', layer_type='{lt}')\n\n"
            f"暂无数据，请先在「Analyze」Tab 分析至少一个模型。"
        )

    formatted = _format_leaderboard(df)
    status = (
        f"✅ {len(formatted)} entries  "
        f"| modality={mod}  layer_type={lt}"
    )
    return formatted, status


def build_tab_leaderboard():
    with gr.Tab("🏆 Leaderboard"):
        gr.Markdown(r"""
        ## Wang's Five Laws — Model Leaderboard

        **Wang Score = 1 − median(SSR\_QK)**  Higher is better. Theoretical max = 1.  
        Computed from `standard` layers only (global/KV-shared layers excluded).

        > 王氏评分 = 1 − median(SSR_QK)，越高越好，理论极值=1。
        > 仅基于 standard 层计算（排除 K=V 共享的全局层干扰）。
        """)

        with gr.Row():
            modality_input = gr.Dropdown(
                label="Modality",
                choices=["language", "vision", "audio", "all"],
                value="language",
                scale=1,
                info="language = text LLM components | 通常选 language",
            )
            layer_type_input = gr.Dropdown(
                label="Layer Type",
                choices=["standard", "global", "all"],
                value="standard",
                scale=1,
                info=(
                    "standard = normal layers  |  "
                    "global = K=V shared (Gemma global layers)"
                ),
            )
            refresh_btn = gr.Button(
                "🔄 Refresh Leaderboard", variant="primary", scale=1
            )

        status_text = gr.Textbox(
            label="Status",
            value="Click Refresh to load leaderboard.",
            lines=1,
            interactive=False,
        )

        leaderboard_table = gr.Dataframe(
            label="Wang Score Leaderboard (sorted by Wang Score ↓)",
            headers=[
                "model_name", "modality", "layer_type",
                "wang_score_pct",
                "median_pearson_QK", "median_ssr_QK", "mean_ssr_QK",
                "median_cosU_QK", "median_cosU_QV", "median_cosV_QK",
                "n_layers", "n_records", "model_id",
            ],
            interactive=False,
            wrap=True,
        )

        gr.Markdown(r"""
        ### Metric Reference | 指标说明

        | Metric | Description | Better |
        |--------|-------------|--------|
        | Wang Score | 1 − median(SSR\_QK)，综合推理能力评分 | ↑ Higher |
        | median\_pearson\_QK | Q/K spectral Pearson correlation (Law 1) | ↑ Higher |
        | median\_ssr\_QK | Q/K normalized spectral mismatch (Law 2) | ↓ Lower |
        | median\_cosU\_QK | Q/K output subspace alignment (Law 4, ≈ random orthogonal) | ≈ 1/√d |
        | median\_cosU\_QV | Q/V output subspace (Law 4, super-orthogonal) | ↓ Lower |
        | median\_cosV\_QK | Q/K input subspace (Law 5, ≈ random orthogonal) | ≈ 1/√D |
        """)

        refresh_btn.click(
            fn=load_leaderboard,
            inputs=[modality_input, layer_type_input],
            outputs=[leaderboard_table, status_text],
        )