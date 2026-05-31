# ui/tab_pythia.py
"""
Tab: Pythia Training Dynamics
================================
扫描 EleutherAI/pythia 多个训练 checkpoint，
实时展示 SSR / UniIso / eff_rank 随训练步数的动态变化。

两个子视图：
  1. 训练曲线（SSR vs step）：pseudo-bulk per-layer median，log x 轴
  2. 层×头热力图（选定 step 的 Q_uni_iso 快照）
"""

import os
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from core.pythia_scanner import scan_pythia, DEFAULT_STEPS, PYTHIA_CONFIGS, DATA_DIR

# ── 常量 ───────────────────────────────────────────────────────────────────────
METRICS = {
    "Q_uni_iso":       "UniIso (Q)",
    "Q_eff_rank":      "eff_rank (Q)",
    "Q_ssr":           "SSR (Q vs K)",
    "Q_sv_max_ratio":  "sv_max_ratio σ₁/σ₂ (Q)",
    "Q_sv_entropy":    "sv_entropy (Q)",
    "K_uni_iso":       "UniIso (K)",
}

# ── 画图函数 ───────────────────────────────────────────────────────────────────

def plot_training_curves(df: pd.DataFrame, metric: str, model_size: str) -> go.Figure:
    """
    训练动态折线图。
    pseudo-bulk: 每层 12/16 个头取 median → 每层一条曲线。
    x 轴 log scale（早期变化密集）。
    """
    if df.empty or metric not in df.columns:
        return go.Figure().update_layout(title="无数据")

    cfg      = PYTHIA_CONFIGS[model_size]
    n_layers = cfg["n_layers"]

    # pseudo-bulk：step × layer 的 median
    pb = df.groupby(["step", "layer"])[metric].median().reset_index()
    pb.columns = ["step", "layer", "value"]

    # 层深度颜色渐变
    colors = px.colors.sample_colorscale("Viridis", n_layers)

    fig = go.Figure()
    for layer in range(n_layers):
        ld = pb[pb["layer"] == layer].sort_values("step")
        if ld.empty:
            continue
        fig.add_trace(go.Scatter(
            x=ld["step"],
            y=ld["value"],
            mode="lines+markers",
            name=f"layer {layer}",
            line=dict(color=colors[layer], width=1.5),
            marker=dict(size=4),
        ))

    # 理论固定点参考线（仅对 uni_iso 有意义）
    if metric == "Q_uni_iso" and not df.empty:
        d_h = cfg["d_head"]
        # 用最后一个 step 的 median eff_rank 估算理论值
        last_step = df["step"].max()
        last_rk   = df[df["step"] == last_step]["Q_eff_rank"].median()
        if not np.isnan(last_rk):
            import math
            theory = math.sqrt(max(0, 2 - 2*math.sqrt(last_rk)/math.sqrt(d_h)))
            fig.add_hline(
                y=theory,
                line_dash="dash",
                line_color="red",
                annotation_text=f"theory k={last_rk:.0f}: {theory:.4f}",
                annotation_position="bottom right",
            )

    fig.update_layout(
        title=f"Pythia-{model_size}: {METRICS.get(metric, metric)} vs Training Step",
        xaxis=dict(title="Training Step", type="log"),
        yaxis=dict(title=METRICS.get(metric, metric)),
        legend=dict(title="Layer", font=dict(size=9)),
        height=480,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


def plot_heatmap(df: pd.DataFrame, metric: str, step: int, model_size: str) -> go.Figure:
    """
    层 × 头热力图：选定 step 的指标快照。
    """
    sub = df[df["step"] == step]
    if sub.empty:
        return go.Figure().update_layout(title=f"step {step} 无数据")

    cfg      = PYTHIA_CONFIGS[model_size]
    n_layers = cfg["n_layers"]
    n_heads  = cfg["n_heads"]

    # 构造 matrix [n_layers, n_heads]
    mat = np.full((n_layers, n_heads), np.nan)
    for _, row in sub.iterrows():
        l, h = int(row["layer"]), int(row["head"])
        if 0 <= l < n_layers and 0 <= h < n_heads:
            mat[l, h] = row[metric]

    fig = go.Figure(data=go.Heatmap(
        z=mat,
        x=[f"h{i}" for i in range(n_heads)],
        y=[f"L{i}" for i in range(n_layers)],
        colorscale="Viridis",
        colorbar=dict(title=METRICS.get(metric, metric)),
    ))
    fig.update_layout(
        title=f"Pythia-{model_size}  step={step}  {METRICS.get(metric, metric)}",
        xaxis_title="Head",
        yaxis_title="Layer",
        height=420,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


# ── CSV 加载 ───────────────────────────────────────────────────────────────────

def load_latest_csv(model_size: str) -> pd.DataFrame:
    """加载 /data 下最新的该 model_size 的 CSV"""
    if not os.path.exists(DATA_DIR):
        return pd.DataFrame()
    files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.startswith(f"pythia_{model_size}_ssr_") and f.endswith(".csv")
    ])
    if not files:
        return pd.DataFrame()
    path = os.path.join(DATA_DIR, files[-1])
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# ── Gradio Tab ─────────────────────────────────────────────────────────────────

def build_tab_pythia():
    with gr.Tab("📈 Pythia Training Dynamics"):
        gr.Markdown("""
### Pythia Checkpoint Scan
扫描 [EleutherAI/pythia](https://huggingface.co/EleutherAI) 训练过程中的 attention 矩阵谱指标。
每个 checkpoint 对应一个训练阶段，展示 UniIso / SSR / eff_rank 的动态演化。
> **数据持久化**：扫描结果保存至 `/data/pythia_{size}_ssr_{ts}.csv`，支持断点续跑。
        """)

        # ── 控制面板 ──────────────────────────────────────────────────────────
        with gr.Row():
            dd_size = gr.Dropdown(
                choices=["160m", "410m"],
                value="160m",
                label="Model Size",
            )
            dd_metric = gr.Dropdown(
                choices=list(METRICS.keys()),
                value="Q_uni_iso",
                label="Metric",
            )
            cb_steps = gr.CheckboxGroup(
                choices=[str(s) for s in DEFAULT_STEPS],
                value=[str(s) for s in DEFAULT_STEPS],
                label="Checkpoints to scan",
            )

        with gr.Row():
            btn_scan  = gr.Button("🚀 Run Scan", variant="primary")
            btn_load  = gr.Button("📂 Load Existing CSV")
            btn_dl    = gr.DownloadButton("⬇️ Download CSV", visible=False)

        status = gr.Textbox(label="Status", lines=4, interactive=False)

        # ── 图表区 ────────────────────────────────────────────────────────────
        with gr.Row():
            plot_curve = gr.Plot(label="Training Dynamics (pseudo-bulk per layer)")
        with gr.Row():
            sl_step = gr.Slider(
                minimum=1, maximum=143000, step=1, value=143000,
                label="Heatmap: select step",
            )
            plot_heat = gr.Plot(label="Layer × Head Heatmap")

        # ── 数据状态（存在 State 里，避免重复 IO）────────────────────────────
        state_df  = gr.State(pd.DataFrame())
        state_csv = gr.State("")

        # ── 事件：Run Scan ────────────────────────────────────────────────────
        def on_scan(model_size, steps_str_list, metric, progress=gr.Progress()):
            steps = [int(s) for s in steps_str_list]

            log_lines = []
            def prog_fn(cur, total, desc):
                log_lines.append(f"[{cur}/{total}] {desc}")
                progress(cur / total, desc=desc)

            try:
                csv_path = scan_pythia(
                    model_size=model_size,
                    steps=steps,
                    token=None,
                    progress_fn=prog_fn,
                )
            except Exception as e:
                return (
                    f"❌ 扫描失败: {e}",
                    pd.DataFrame(),
                    csv_path if 'csv_path' in dir() else "",
                    go.Figure(),
                    go.Figure(),
                    gr.update(visible=False),
                )

            df = load_latest_csv(model_size)
            if df.empty:
                return ("⚠️ 扫描完成但 CSV 为空", df, csv_path,
                        go.Figure(), go.Figure(), gr.update(visible=False))

            done_steps = sorted(df["step"].unique())
            fig_curve  = plot_training_curves(df, metric, model_size)
            last_step  = done_steps[-1]
            fig_heat   = plot_heatmap(df, metric, last_step, model_size)

            summary = "\n".join(log_lines[-10:])  # 最后 10 行
            summary += f"\n\n✅ 完成  {len(done_steps)} steps  {len(df)} rows\n→ {csv_path}"

            return (summary, df, csv_path,
                    fig_curve, fig_heat, gr.update(visible=True, value=csv_path))

        btn_scan.click(
            fn=on_scan,
            inputs=[dd_size, cb_steps, dd_metric],
            outputs=[status, state_df, state_csv,
                     plot_curve, plot_heat, btn_dl],
        )

        # ── 事件：Load Existing CSV ────────────────────────────────────────────
        def on_load(model_size, metric):
            df = load_latest_csv(model_size)
            if df.empty:
                return ("⚠️ /data 下未找到该模型的 CSV，请先运行扫描",
                        df, "", go.Figure(), go.Figure(), gr.update(visible=False))

            done_steps = sorted(df["step"].unique())
            csv_path   = os.path.join(DATA_DIR, sorted([
                f for f in os.listdir(DATA_DIR)
                if f.startswith(f"pythia_{model_size}_ssr_")
            ])[-1])
            fig_curve = plot_training_curves(df, metric, model_size)
            fig_heat  = plot_heatmap(df, metric, done_steps[-1], model_size)
            return (f"✅ 加载 {len(done_steps)} steps，{len(df)} rows\n→ {csv_path}",
                    df, csv_path,
                    fig_curve, fig_heat, gr.update(visible=True, value=csv_path))

        btn_load.click(
            fn=on_load,
            inputs=[dd_size, dd_metric],
            outputs=[status, state_df, state_csv,
                     plot_curve, plot_heat, btn_dl],
        )

        # ── 事件：Metric / Step 变更时重绘 ────────────────────────────────────
        def on_metric_change(df, metric, model_size):
            if df.empty:
                return go.Figure()
            return plot_training_curves(df, metric, model_size)

        def on_step_change(df, step, metric, model_size):
            if df.empty:
                return go.Figure()
            available = sorted(df["step"].unique())
            # 找最近的已有 step
            nearest = min(available, key=lambda s: abs(s - step))
            return plot_heatmap(df, metric, nearest, model_size)

        dd_metric.change(
            fn=on_metric_change,
            inputs=[state_df, dd_metric, dd_size],
            outputs=[plot_curve],
        )
        sl_step.change(
            fn=on_step_change,
            inputs=[state_df, sl_step, dd_metric, dd_size],
            outputs=[plot_heat],
        )
