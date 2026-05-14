# ui/tab_plot.py
"""
Tab5: Plot — Publication-quality figure generation
Data pulled from SQLite DB.
Supports: single model (4×3) and two-model comparison (4×3).
Export: PNG (300 dpi) / PDF / SVG.
Engine: matplotlib (publication) + optional Plotly (interactive).
"""

import os
import tempfile
import zipfile

import gradio as gr
import pandas as pd
import numpy as np

from db.schema import init_db
from db.reader import get_layer_metrics, get_analyzed_models
from core.plotter import (
    plot_single_model,
    plot_compare_models,
    save_figure,
    fig_to_plotly,
)

# ── Output directory ──────────────────────────────────────────────────────────
_OUT_DIR = "/tmp/wang_plots"
os.makedirs(_OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_model_choices() -> list[str]:
    try:
        conn = init_db()
        df   = get_analyzed_models(conn)
        if df.empty:
            return []
        return df["model_id"].tolist()
    except Exception:
        return []


def _load_df(model_id: str, modality: str,
             start_layer: int, end_layer: int) -> pd.DataFrame:
    conn = init_db()
    df = get_layer_metrics(
        conn,
        model_id    = model_id,
        modality    = modality if modality != "all" else None,
        layer_type  = None,
        start_layer = int(start_layer),
        end_layer   = int(end_layer),
    )
    return df


def _infer_dims(df: pd.DataFrame) -> tuple[int, int]:
    """Try to read head_dim and d_model from the dataframe."""
    head_dim = 128
    d_model  = 5120
    if not df.empty:
        if "head_dim" in df.columns:
            v = df["head_dim"].dropna()
            if len(v):
                head_dim = int(v.median())
        if "d_model" in df.columns:
            v = df["d_model"].dropna()
            if len(v):
                d_model = int(v.median())
    return head_dim, d_model


def _short_name(model_id: str) -> str:
    return model_id.split("/")[-1] if "/" in model_id else model_id


def _safe_base_path(name: str) -> str:
    safe = name.replace("/", "_").replace(" ", "_")
    return os.path.join(_OUT_DIR, safe)


# ─────────────────────────────────────────────────────────────────────────────
# Main generation functions
# ─────────────────────────────────────────────────────────────────────────────

def generate_single(
    model_id:    str,
    modality:    str,
    start_layer: int,
    end_layer:   int,
    show_band:   bool,
    progress=gr.Progress()
) -> tuple:
    """
    Returns: (status_str, png_path, [png_path, pdf_path, svg_path], plotly_fig)
    """
    if not model_id or not model_id.strip():
        return "❌ Please select a model.", None, None, None

    progress(0.1, desc="Loading data from DB...")
    df = _load_df(model_id, modality, start_layer, end_layer)

    if df.empty:
        return (
            f"❌ No data found for {model_id} "
            f"(modality={modality}, layers {start_layer}~{end_layer}).\n"
            f"Please run analysis first in Tab 2.",
            None, None, None
        )

    progress(0.35, desc="Inferring dimensions...")
    head_dim, d_model = _infer_dims(df)
    n_layers  = df["layer"].nunique()
    n_records = len(df)

    progress(0.50, desc="Generating matplotlib figure...")
    name = _short_name(model_id)
    fig  = plot_single_model(
        df, model_name=name,
        show_band=show_band,
        head_dim=head_dim,
        d_model=d_model,
    )

    progress(0.75, desc="Saving PNG / PDF / SVG...")
    base  = _safe_base_path(f"single_{name}_L{start_layer}-{end_layer}")
    paths = save_figure(fig, base)

    progress(0.90, desc="Generating Plotly preview...")
    plotly_fig = fig_to_plotly(fig)

    import matplotlib.pyplot as plt
    plt.close(fig)

    status = (
        f"✅ {model_id}  |  modality={modality}  "
        f"|  layers {start_layer}~{end_layer}  "
        f"|  {n_layers} layers  {n_records} head-records\n"
        f"   head_dim={head_dim}  d_model={d_model}\n"
        f"   Saved: {', '.join(os.path.basename(p) for p in paths)}"
    )
    png_path = paths[0]
    return status, png_path, paths, plotly_fig


def generate_compare(
    model_a:     str,
    model_b:     str,
    modality:    str,
    start_layer: int,
    end_layer:   int,
    show_band:   bool,
    show_delta:  bool,
    progress=gr.Progress()
) -> tuple:
    if not model_a or not model_b:
        return "❌ Please select both models.", None, None, None
    if model_a == model_b:
        return "❌ Please select two different models.", None, None, None

    progress(0.10, desc="Loading Model A from DB...")
    df_a = _load_df(model_a, modality, start_layer, end_layer)
    progress(0.25, desc="Loading Model B from DB...")
    df_b = _load_df(model_b, modality, start_layer, end_layer)

    if df_a.empty:
        return f"❌ No data for Model A ({model_a}).", None, None, None
    if df_b.empty:
        return f"❌ No data for Model B ({model_b}).", None, None, None

    head_dim_a, d_model_a = _infer_dims(df_a)
    head_dim_b, d_model_b = _infer_dims(df_b)
    head_dim = int((head_dim_a + head_dim_b) / 2)
    d_model  = int((d_model_a + d_model_b) / 2)

    progress(0.50, desc="Generating comparison figure...")
    name_a = _short_name(model_a)
    name_b = _short_name(model_b)
    fig = plot_compare_models(
        df_a, df_b,
        name_a=name_a, name_b=name_b,
        show_band=show_band,
        show_delta=show_delta,
        head_dim=head_dim,
        d_model=d_model,
    )

    progress(0.80, desc="Saving PNG / PDF / SVG...")
    base  = _safe_base_path(f"compare_{name_a}_vs_{name_b}_L{start_layer}-{end_layer}")
    paths = save_figure(fig, base)

    progress(0.92, desc="Generating Plotly preview...")
    plotly_fig = fig_to_plotly(fig)

    import matplotlib.pyplot as plt
    plt.close(fig)

    status = (
        f"✅ {name_a}  vs  {name_b}\n"
        f"   modality={modality}  layers {start_layer}~{end_layer}\n"
        f"   Model A: {len(df_a)} records  |  Model B: {len(df_b)} records\n"
        f"   head_dim≈{head_dim}  d_model≈{d_model}\n"
        f"   Saved: {', '.join(os.path.basename(p) for p in paths)}"
    )
    return status, paths[0], paths, plotly_fig


def make_zip(file_paths: list) -> str | None:
    """Bundle all exported files into a single ZIP for download."""
    if not file_paths:
        return None
    zip_path = os.path.join(_OUT_DIR, "wang_laws_figures.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in file_paths:
            if p and os.path.exists(p):
                zf.write(p, os.path.basename(p))
    return zip_path


# ─────────────────────────────────────────────────────────────────────────────
# Tab5 UI
# ─────────────────────────────────────────────────────────────────────────────

def build_tab_plot():
    with gr.Tab("📈 Plot"):
        gr.Markdown("""
        ## Wang's Five Laws — Publication-Quality Figures
        Data is loaded directly from the SQLite database (Tab 2 must be run first).

        **4×3 grid layout** (12 subplots, one figure):
        | Row | Content | Laws |
        |-----|---------|------|
        | 1 | pearson_QK · SSR_QK · α_QK | Law 1 & 2 |
        | 2 | σ_max(Q) · σ_max(K) · κ(Q) & κ(K) | Law 3 |
        | 3 | cosU QK · QV · KV + random baseline | Law 4 |
        | 4 | cosV QK · QV · KV + random baseline | Law 5 |

        Export: **PNG 300 dpi** · **PDF (vector)** · **SVG (vector)**
        """)

        # ── Shared controls ───────────────────────────────────────────────────
        with gr.Row():
            modality_sel = gr.Dropdown(
                label="Modality",
                choices=["language", "vision", "audio", "all"],
                value="language",
                scale=1,
            )
            start_l = gr.Number(
                label="Start Layer", value=0,  precision=0, scale=1
            )
            end_l = gr.Number(
                label="End Layer",   value=47, precision=0, scale=1
            )
            show_band_chk = gr.Checkbox(
                label="Show 25%–75% band (head consistency)",
                value=True, scale=1
            )

        gr.Markdown("---")

        # ══ Mode 1: Single model ══════════════════════════════════════════════
        with gr.Accordion("📊 Single Model", open=True):
            with gr.Row():
                model_choices = _get_model_choices()
                single_model = gr.Dropdown(
                    label="Model",
                    choices=model_choices,
                    value=model_choices[0] if model_choices else None,
                    allow_custom_value=True,
                    scale=3,
                    info="Refresh the page after analyzing new models to update this list."
                )
                single_btn = gr.Button(
                    "🎨 Generate Figure", variant="primary", scale=1
                )

            single_status = gr.Textbox(
                label="Status", lines=3, interactive=False
            )

            with gr.Tabs():
                with gr.Tab("🖼️ Preview (PNG)"):
                    single_img = gr.Image(
                        label="Figure preview",
                        type="filepath",
                        height=600,
                    )
                with gr.Tab("📉 Interactive (Plotly)"):
                    single_plotly = gr.Plot(label="Plotly interactive")

            with gr.Row():
                dl_single_png = gr.File(label="⬇ PNG (300 dpi)")
                dl_single_pdf = gr.File(label="⬇ PDF (vector)")
                dl_single_svg = gr.File(label="⬇ SVG (vector)")
                dl_single_zip = gr.File(label="⬇ ZIP (all formats)")

        gr.Markdown("---")

        # ══ Mode 2: Two-model comparison ══════════════════════════════════════
        with gr.Accordion("📊 Two-Model Comparison", open=False):
            with gr.Row():
                model_a = gr.Dropdown(
                    label="Model A (solid line)",
                    choices=model_choices,
                    value=model_choices[0] if len(model_choices) > 0 else None,
                    allow_custom_value=True,
                    scale=2,
                )
                model_b = gr.Dropdown(
                    label="Model B (dashed line)",
                    choices=model_choices,
                    value=model_choices[1] if len(model_choices) > 1 else None,
                    allow_custom_value=True,
                    scale=2,
                )
                show_delta_chk = gr.Checkbox(
                    label="Show Δ (B − A) fill",
                    value=True, scale=1
                )
                compare_btn = gr.Button(
                    "🎨 Generate Comparison", variant="primary", scale=1
                )

            compare_status = gr.Textbox(
                label="Status", lines=3, interactive=False
            )

            with gr.Tabs():
                with gr.Tab("🖼️ Preview (PNG)"):
                    compare_img = gr.Image(
                        label="Comparison figure preview",
                        type="filepath",
                        height=600,
                    )
                with gr.Tab("📉 Interactive (Plotly)"):
                    compare_plotly = gr.Plot(label="Plotly interactive")

            with gr.Row():
                dl_cmp_png = gr.File(label="⬇ PNG (300 dpi)")
                dl_cmp_pdf = gr.File(label="⬇ PDF (vector)")
                dl_cmp_svg = gr.File(label="⬇ SVG (vector)")
                dl_cmp_zip = gr.File(label="⬇ ZIP (all formats)")

        gr.Markdown("""
        ---
        **Tips**
        - Band = 25%–75% quantile across attention heads per layer.
          Narrow band → heads behave consistently → model is "well-organized".
        - Vertical dotted lines mark **global layers** (K=V shared, e.g. Gemma-4).
        - Dashed horizontal lines = theoretical ideals or random baselines.
        - For Law 4 & 5 panels, Q–V and K–V cosU values **below** the random baseline
          indicate **super-orthogonality** — a key signature of pretraining convergence.
        """)

        # ── Wire up single model ──────────────────────────────────────────────
        _single_file_state = gr.State([])

        def _run_single(model_id, modality, start, end, band, progress=gr.Progress()):
            status, png, paths, plotly_fig = generate_single(
                model_id, modality, int(start), int(end), band, progress
            )
            if paths is None:
                return status, None, None, None, None, None, None, []
            zip_p = make_zip(paths)
            png_p = paths[0] if len(paths) > 0 else None
            pdf_p = paths[1] if len(paths) > 1 else None
            svg_p = paths[2] if len(paths) > 2 else None
            return (status, png, plotly_fig,
                    png_p, pdf_p, svg_p, zip_p, paths)

        single_btn.click(
            fn=_run_single,
            inputs=[single_model, modality_sel, start_l, end_l, show_band_chk],
            outputs=[
                single_status, single_img, single_plotly,
                dl_single_png, dl_single_pdf, dl_single_svg, dl_single_zip,
                _single_file_state,
            ]
        )

        # ── Wire up comparison ────────────────────────────────────────────────
        _compare_file_state = gr.State([])

        def _run_compare(ma, mb, modality, start, end, band, delta,
                         progress=gr.Progress()):
            status, png, paths, plotly_fig = generate_compare(
                ma, mb, modality, int(start), int(end), band, delta, progress
            )
            if paths is None:
                return status, None, None, None, None, None, None, []
            zip_p = make_zip(paths)
            png_p = paths[0] if len(paths) > 0 else None
            pdf_p = paths[1] if len(paths) > 1 else None
            svg_p = paths[2] if len(paths) > 2 else None
            return (status, png, plotly_fig,
                    png_p, pdf_p, svg_p, zip_p, paths)

        compare_btn.click(
            fn=_run_compare,
            inputs=[model_a, model_b, modality_sel,
                    start_l, end_l, show_band_chk, show_delta_chk],
            outputs=[
                compare_status, compare_img, compare_plotly,
                dl_cmp_png, dl_cmp_pdf, dl_cmp_svg, dl_cmp_zip,
                _compare_file_state,
            ]
        )