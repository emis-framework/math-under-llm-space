# ui/tab_olmoe.py
"""
Tab: OLMoE Training Dynamics
================================
扫描 allenai/OLMoE-1B-7B-0924 的训练 checkpoint，
展示 SSR / UniIso / eff_rank 随训练步数（tokens数）的动态变化。

和 tab_pythia.py 的区别：
  x轴用tokens_B（更有物理意义，和grokking文献对齐）
  标注来自 arxiv 2506.21551 的 grokking 区间
"""

import os
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from core.olmoe_scanner import (
    scan_olmoe, OLMOE_CONFIGS, DEFAULT_STEPS_OLMOE, DATA_DIR
)

# ── grokking区间（来自arxiv 2506.21551）────────────────────────────────────────
# 单位：tokens（B）
GROKKING_REGIONS = {
    "Common sense":  (210,  420,  "#2166AC"),   # step50K-100K
    "Code":          (420,  840,  "#D6604D"),   # step100K-200K
    "Math":          (840,  1680, "#762A83"),   # step200K-400K
}

# ── 指标映射 ───────────────────────────────────────────────────────────────────
METRICS = {
    "Q_uni_iso":      "UniIso (Q)",
    "Q_eff_rank":     "eff_rank (Q)",
    "ssr":            "SSR (Q vs K)",
    "Q_sv_max_ratio": "sv_max_ratio σ₁/σ₂ (Q)",
    "Q_sv_entropy":   "sv_entropy (Q)",
    "K_uni_iso":      "UniIso (K)",
}

# ── 画图函数 ───────────────────────────────────────────────────────────────────

def plot_training_curves(
    df: pd.DataFrame,
    metric: str,
    show_grokking: bool = True,
) -> go.Figure:
    """
    训练动态折线图。
    x轴：tokens_B（和grokking文献对齐）
    pseudo-bulk：每层16头取median
    标注grokking区间（来自arxiv 2506.21551）
    """
    if df.empty or metric not in df.columns:
        return go.Figure().update_layout(title="无数据")

    cfg      = list(OLMOE_CONFIGS.values())[0]
    n_layers = cfg["n_layers"]
    colors   = px.colors.sample_colorscale("Viridis", n_layers)

    # pseudo-bulk
    pb = df.groupby(["tokens_B", "layer"])[metric].median().reset_index()

    fig = go.Figure()

    # grokking区间标注
    if show_grokking:
        for domain, (t_start, t_end, color) in GROKKING_REGIONS.items():
            fig.add_vrect(
                x0=t_start, x1=t_end,
                fillcolor=color, opacity=0.08,
                layer="below", line_width=0,
                annotation_text=domain,
                annotation_position="top left",
                annotation_font_size=9,
            )

    # 各层曲线
    for layer in range(n_layers):
        ld = pb[pb["layer"] == layer].sort_values("tokens_B")
        if ld.empty:
            continue
        fig.add_trace(go.Scatter(
            x=ld["tokens_B"],
            y=ld[metric],
            mode="lines+markers",
            name=f"L{layer}",
            line=dict(color=colors[layer], width=1.5),
            marker=dict(size=4),
        ))

    fig.update_layout(
        title=(f"OLMoE-1B-7B: {METRICS.get(metric, metric)} vs Training Tokens<br>"
               f"<sup>Shaded regions: grokking intervals from arxiv 2506.21551</sup>"),
        xaxis=dict(title="Training Tokens (B)", type="linear"),
        yaxis=dict(title=METRICS.get(metric, metric)),
        legend=dict(title="Layer", font=dict(size=9)),
        height=500,
        margin=dict(l=60, r=20, t=70, b=50),
    )
    return fig


def plot_heatmap(
    df: pd.DataFrame,
    metric: str,
    tokens_B: float,
) -> go.Figure:
    """层×头热力图：选定tokens数的快照"""
    # 找最近的tokens_B
    available = sorted(df["tokens_B"].unique())
    if not available:
        return go.Figure().update_layout(title="无数据")
    nearest = min(available, key=lambda t: abs(t - tokens_B))
    sub = df[df["tokens_B"] == nearest]

    cfg      = list(OLMOE_CONFIGS.values())[0]
    n_layers = cfg["n_layers"]
    n_heads  = cfg["n_heads"]

    mat = np.full((n_layers, n_heads), np.nan)
    for _, row in sub.iterrows():
        l, h = int(row["layer"]), int(row["head"])
        if 0 <= l < n_layers and 0 <= h < n_heads:
            mat[l, h] = row[metric]

    branch = sub["branch"].iloc[0] if len(sub) else ""
    fig = go.Figure(data=go.Heatmap(
        z=mat,
        x=[f"h{i}" for i in range(n_heads)],
        y=[f"L{i}" for i in range(n_layers)],
        colorscale="Viridis",
        colorbar=dict(title=METRICS.get(metric, metric)),
    ))
    fig.update_layout(
        title=f"OLMoE  {branch}  ({nearest}B tokens)  {METRICS.get(metric, metric)}",
        xaxis_title="Head",
        yaxis_title="Layer",
        height=400,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


# ── CSV 加载 ───────────────────────────────────────────────────────────────────

def load_latest_csv() -> pd.DataFrame:
    if not os.path.exists(DATA_DIR):
        return pd.DataFrame()
    files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.startswith("olmoe_OLMoE") and f.endswith(".csv")
    ])
    if not files:
        return pd.DataFrame()
    try:
        return pd.read_csv(os.path.join(DATA_DIR, files[-1]))
    except Exception:
        return pd.DataFrame()


# ── Gradio Tab ─────────────────────────────────────────────────────────────────

def build_tab_olmoe():
    with gr.Tab("🧠 OLMoE"):
        gr.Markdown("""
### OLMoE-1B-7B Checkpoint Scan
扫描 [allenai/OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924) 
训练过程中的 attention 矩阵谱指标。

阴影区域标注来自 **arxiv 2506.21551** 的 grokking 时间窗口。
> 数据持久化至 `/data/olmoe_OLMoE-1B-7B-0924_ssr_{ts}.csv`，支持断点续跑。
        """)

        with gr.Row():
            dd_metric = gr.Dropdown(
                choices=list(METRICS.keys()),
                value="ssr",
                label="Metric",
            )
            cb_grok = gr.Checkbox(
                value=True,
                label="显示grokking区间（arxiv 2506.21551）",
            )
            btn_scan = gr.Button("🚀 Run Scan", variant="primary")
            btn_load = gr.Button("📂 Load Existing CSV")
            btn_dl   = gr.DownloadButton("⬇️ Download CSV", visible=False)

        status = gr.Textbox(label="Status", lines=3, interactive=False)

        with gr.Row():
            plot_curve = gr.Plot(label="Training Dynamics (pseudo-bulk per layer)")

        with gr.Row():
            sl_tokens = gr.Slider(
                minimum=20, maximum=5117, step=20, value=1000,
                label="Heatmap: select tokens (B)",
            )
            plot_heat = gr.Plot(label="Layer × Head Heatmap")

        state_df  = gr.State(pd.DataFrame())
        state_csv = gr.State("")

        # ── Run Scan ──────────────────────────────────────────────────────────
        def on_scan(metric, show_grok, progress=gr.Progress()):
            log_lines = []
            def prog_fn(cur, total, desc):
                log_lines.append(f"[{cur}/{total}] {desc}")
                progress(cur / total, desc=desc)

            try:
                csv_path = scan_olmoe(
                    model_name="OLMoE-1B-7B-0924",
                    steps=DEFAULT_STEPS_OLMOE,
                    token=None,
                    progress_fn=prog_fn,
                )
            except Exception as e:
                return (f"❌ 扫描失败: {e}", pd.DataFrame(), "",
                        go.Figure(), go.Figure(), gr.update(visible=False))

            df = load_latest_csv()
            if df.empty:
                return ("⚠️ CSV为空", df, csv_path,
                        go.Figure(), go.Figure(), gr.update(visible=False))

            fig_c = plot_training_curves(df, metric, show_grok)
            fig_h = plot_heatmap(df, metric, float(df["tokens_B"].max()))
            summary = "\n".join(log_lines[-8:])
            summary += f"\n\n✅ 完成  {df['step'].nunique()} steps  {len(df)} rows"
            return (summary, df, csv_path,
                    fig_c, fig_h, gr.update(visible=True, value=csv_path))

        btn_scan.click(
            fn=on_scan,
            inputs=[dd_metric, cb_grok],
            outputs=[status, state_df, state_csv,
                     plot_curve, plot_heat, btn_dl],
        )

        # ── Load CSV ──────────────────────────────────────────────────────────
        def on_load(metric, show_grok):
            df = load_latest_csv()
            if df.empty:
                return ("⚠️ /data 下未找到OLMoE CSV",
                        df, "", go.Figure(), go.Figure(),
                        gr.update(visible=False))
            csv_path = os.path.join(DATA_DIR, sorted([
                f for f in os.listdir(DATA_DIR)
                if f.startswith("olmoe_OLMoE") and f.endswith(".csv")
            ])[-1])
            fig_c = plot_training_curves(df, metric, show_grok)
            fig_h = plot_heatmap(df, metric, float(df["tokens_B"].max()))
            return (f"✅ {df['step'].nunique()} steps  {len(df)} rows\n→ {csv_path}",
                    df, csv_path,
                    fig_c, fig_h, gr.update(visible=True, value=csv_path))

        btn_load.click(
            fn=on_load,
            inputs=[dd_metric, cb_grok],
            outputs=[status, state_df, state_csv,
                     plot_curve, plot_heat, btn_dl],
        )

        # ── Metric / Tokens 变更重绘 ──────────────────────────────────────────
        def on_metric(df, metric, show_grok):
            if df.empty:
                return go.Figure()
            return plot_training_curves(df, metric, show_grok)

        def on_tokens(df, tokens_B, metric):
            if df.empty:
                return go.Figure()
            return plot_heatmap(df, metric, tokens_B)

        dd_metric.change(
            fn=on_metric,
            inputs=[state_df, dd_metric, cb_grok],
            outputs=[plot_curve],
        )
        sl_tokens.change(
            fn=on_tokens,
            inputs=[state_df, sl_tokens, dd_metric],
            outputs=[plot_heat],
        )
        cb_grok.change(
            fn=on_metric,
            inputs=[state_df, dd_metric, cb_grok],
            outputs=[plot_curve],
        )
