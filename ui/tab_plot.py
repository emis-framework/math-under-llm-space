# ui/tab_plot.py 
"""
Tab5: Plot — Publication-quality figure generation
- Plotly native  → interactive browser preview (12×1, full-width, fast)
- matplotlib     → PNG / PDF / SVG export (300 dpi, paper-ready)

NO nested gr.Tabs() — avoids Gradio rendering bugs with Accordion+Tabs nesting.
Two side-by-side buttons instead: ⚡ Interactive  |  🖨️ Export
"""

import os
import zipfile

import gradio as gr
import pandas as pd

from db.schema import init_db
from db.reader import get_layer_metrics, get_analyzed_models
from core.plotter import plot_single_model, plot_compare_models, save_figure
from core.plotter_plotly import plotly_single, plotly_compare

_OUT_DIR = "/tmp/wang_plots"
os.makedirs(_OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_model_choices() -> list[str]:
    try:
        conn = init_db()
        df   = get_analyzed_models(conn)
        return df["model_id"].tolist() if not df.empty else []
    except Exception:
        return []


def _load_df(model_id, modality, start_layer, end_layer) -> pd.DataFrame:
    conn = init_db()
    return get_layer_metrics(
        conn,
        model_id    = model_id,
        modality    = modality if modality != "all" else None,
        layer_type  = None,
        start_layer = int(start_layer),
        end_layer   = int(end_layer),
    )


def _infer_dims(df: pd.DataFrame) -> tuple[int, int]:
    head_dim = 128
    d_model  = 5120
    if not df.empty:
        if "head_dim" in df.columns and df["head_dim"].notna().any():
            head_dim = int(df["head_dim"].dropna().median())
        if "d_model" in df.columns and df["d_model"].notna().any():
            d_model  = int(df["d_model"].dropna().median())
    return head_dim, d_model


def _short(model_id: str) -> str:
    return model_id.split("/")[-1] if "/" in model_id else model_id


def _safe_path(tag: str) -> str:
    return os.path.join(_OUT_DIR, tag.replace("/", "_").replace(" ", "_"))


def _make_zip(paths: list) -> str | None:
    valid = [p for p in paths if p and os.path.exists(p)]
    if not valid:
        return None
    zp = os.path.join(_OUT_DIR, "wang_laws_figures.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in valid:
            zf.write(p, os.path.basename(p))
    return zp


# ─────────────────────────────────────────────────────────────────────────────
# Single-model handlers
# ─────────────────────────────────────────────────────────────────────────────

def gen_single_plotly(model_id, modality, start_l, end_l, show_band,
                      progress=gr.Progress()):
    if not model_id:
        return None, "Please select a model."
    progress(0.2, desc="Loading data from DB...")
    df = _load_df(model_id, modality, start_l, end_l)
    if df.empty:
        return None, f"No data for {model_id}. Run Tab 2 analysis first."
    progress(0.7, desc="Building Plotly figure...")
    fig = plotly_single(df, _short(model_id), show_band=show_band)
    status = (
        f"✅ {model_id}  |  {df['layer'].nunique()} layers  "
        f"{len(df)} head-records  |  modality={modality}"
    )
    progress(1.0)
    return fig, status


def gen_single_export(model_id, modality, start_l, end_l, show_band,
                      progress=gr.Progress()):
    if not model_id:
        return "Please select a model.", None, None, None, None, None
    progress(0.15, desc="Loading data from DB...")
    df = _load_df(model_id, modality, start_l, end_l)
    if df.empty:
        return f"No data for {model_id}.", None, None, None, None, None
    head_dim, d_model = _infer_dims(df)
    progress(0.40, desc="Rendering matplotlib figure (18×20 in, 300 dpi)...")
    import matplotlib.pyplot as plt
    fig = plot_single_model(
        df, _short(model_id),
        show_band=show_band,
        head_dim=head_dim, d_model=d_model,
    )
    progress(0.78, desc="Saving PNG / PDF / SVG...")
    base  = _safe_path(f"single_{_short(model_id)}_L{int(start_l)}-{int(end_l)}")
    paths = save_figure(fig, base)
    plt.close(fig)
    zip_p  = _make_zip(paths)
    status = (
        f"✅ Exported: {', '.join(os.path.basename(p) for p in paths)}\n"
        f"   head_dim={head_dim}  d_model={d_model}"
    )
    progress(1.0)
    png = paths[0] if len(paths) > 0 else None
    pdf = paths[1] if len(paths) > 1 else None
    svg = paths[2] if len(paths) > 2 else None
    # 6 values: status, preview(=png), png_dl, pdf_dl, svg_dl, zip
    return status, png, png, pdf, svg, zip_p


# ─────────────────────────────────────────────────────────────────────────────
# Compare handlers
# ─────────────────────────────────────────────────────────────────────────────

def gen_compare_plotly(model_a, model_b, modality, start_l, end_l,
                       show_band, show_delta, progress=gr.Progress()):
    if not model_a or not model_b:
        return None, "Please select both models."
    if model_a == model_b:
        return None, "Please select two different models."
    progress(0.15, desc="Loading Model A...")
    df_a = _load_df(model_a, modality, start_l, end_l)
    progress(0.35, desc="Loading Model B...")
    df_b = _load_df(model_b, modality, start_l, end_l)
    if df_a.empty:
        return None, f"No data for Model A ({model_a})."
    if df_b.empty:
        return None, f"No data for Model B ({model_b})."
    progress(0.65, desc="Building Plotly comparison figure...")
    fig = plotly_compare(df_a, df_b, _short(model_a), _short(model_b),
                         show_band=show_band, show_delta=show_delta)
    status = (
        f"✅ {_short(model_a)} vs {_short(model_b)}  |  "
        f"A: {len(df_a)} records  B: {len(df_b)} records  |  modality={modality}"
    )
    progress(1.0)
    return fig, status


def gen_compare_export(model_a, model_b, modality, start_l, end_l,
                       show_band, show_delta, progress=gr.Progress()):
    if not model_a or not model_b or model_a == model_b:
        return "Select two different models.", None, None, None, None, None
    progress(0.10, desc="Loading data...")
    df_a = _load_df(model_a, modality, start_l, end_l)
    df_b = _load_df(model_b, modality, start_l, end_l)
    if df_a.empty or df_b.empty:
        return "Missing data for one or both models.", None, None, None, None, None
    head_dim_a, d_model_a = _infer_dims(df_a)
    head_dim_b, d_model_b = _infer_dims(df_b)
    head_dim = (head_dim_a + head_dim_b) // 2
    d_model  = (d_model_a  + d_model_b)  // 2
    progress(0.40, desc="Rendering matplotlib figure...")
    import matplotlib.pyplot as plt
    fig = plot_compare_models(
        df_a, df_b, _short(model_a), _short(model_b),
        show_band=show_band, show_delta=show_delta,
        head_dim=head_dim, d_model=d_model,
    )
    progress(0.78, desc="Saving PNG / PDF / SVG...")
    base  = _safe_path(
        f"compare_{_short(model_a)}_vs_{_short(model_b)}_L{int(start_l)}-{int(end_l)}"
    )
    paths = save_figure(fig, base)
    plt.close(fig)
    zip_p  = _make_zip(paths)
    status = (
        f"✅ Exported: {', '.join(os.path.basename(p) for p in paths)}\n"
        f"   head_dim≈{head_dim}  d_model≈{d_model}"
    )
    progress(1.0)
    png = paths[0] if len(paths) > 0 else None
    pdf = paths[1] if len(paths) > 1 else None
    svg = paths[2] if len(paths) > 2 else None
    # 6 values: status, preview(=png), png_dl, pdf_dl, svg_dl, zip
    return status, png, png, pdf, svg, zip_p


# ─────────────────────────────────────────────────────────────────────────────
# Tab5 UI  —  NO nested gr.Tabs() inside Accordion
# ─────────────────────────────────────────────────────────────────────────────

def build_tab_plot():
    with gr.Tab("📈 Plot"):
        gr.Markdown("""
        ## Wang's Five Laws — Figures

        | Button | Engine | Speed | Output |
        |--------|--------|-------|--------|
        | ⚡ **Interactive** | Native Plotly 12×1 full-width | ~2 s | In-page, hover/zoom |
        | 🖨️ **Export** | Matplotlib 18×20 in @ 300 dpi | ~30 s | PNG · PDF · SVG download |

        > Run **Tab 2 (Analyze)** first to populate the database.
        """)

        # ── Shared controls ───────────────────────────────────────────────────
        with gr.Row():
            modality_sel = gr.Dropdown(
                ["language", "vision", "audio", "all"],
                value="language", label="Modality", scale=1,
            )
            start_l = gr.Number(value=0,  precision=0, label="Start Layer", scale=1)
            end_l   = gr.Number(value=47, precision=0, label="End Layer",   scale=1)
            show_band_chk = gr.Checkbox(
                value=True, label="Show IQR band", scale=1
            )

        gr.Markdown("---")

        # ══ Single model ══════════════════════════════════════════════════════
        with gr.Accordion("📊 Single Model", open=True):

            choices = _get_model_choices()
            single_model = gr.Dropdown(
                choices=choices,
                value=choices[0] if choices else None,
                allow_custom_value=True,
                label="Model",
                info="Refresh page after new analysis to update this list.",
            )

            # Two side-by-side buttons — no nested Tabs
            with gr.Row():
                single_plotly_btn = gr.Button(
                    "⚡ Interactive (Plotly)", variant="primary", scale=1
                )
                single_export_btn = gr.Button(
                    "🖨️ Export PNG / PDF / SVG", variant="secondary", scale=1
                )

            single_status = gr.Textbox(
                lines=2, interactive=False, label="Status"
            )

            # Interactive output — always visible, populated on demand
            single_plotly_fig = gr.Plot(label="Interactive figure")

            # Export outputs — always visible, populated on demand
            gr.Markdown("#### 🖨️ Export outputs")
            single_preview = gr.Image(
                type="filepath", label="PNG preview (click to enlarge)", height=350
            )
            with gr.Row():
                dl_s_png = gr.File(label="⬇ PNG (300 dpi)")
                dl_s_pdf = gr.File(label="⬇ PDF (vector)")
                dl_s_svg = gr.File(label="⬇ SVG (vector)")
                dl_s_zip = gr.File(label="⬇ ZIP (all formats)")

        gr.Markdown("---")

        # ══ Two-model comparison ══════════════════════════════════════════════
        with gr.Accordion("📊 Two-Model Comparison", open=False):

            with gr.Row():
                model_a = gr.Dropdown(
                    choices=choices,
                    value=choices[0] if len(choices) > 0 else None,
                    allow_custom_value=True,
                    label="Model A  (solid line)", scale=2,
                )
                model_b = gr.Dropdown(
                    choices=choices,
                    value=choices[1] if len(choices) > 1 else None,
                    allow_custom_value=True,
                    label="Model B  (dashed line)", scale=2,
                )
                show_delta_chk = gr.Checkbox(
                    value=True, label="Show Δ fill (B − A)", scale=1
                )

            with gr.Row():
                cmp_plotly_btn = gr.Button(
                    "⚡ Interactive (Plotly)", variant="primary", scale=1
                )
                cmp_export_btn = gr.Button(
                    "🖨️ Export PNG / PDF / SVG", variant="secondary", scale=1
                )

            cmp_status = gr.Textbox(
                lines=2, interactive=False, label="Status"
            )

            cmp_plotly_fig = gr.Plot(label="Interactive comparison figure")

            gr.Markdown("#### 🖨️ Export outputs")
            cmp_preview = gr.Image(
                type="filepath", label="PNG preview", height=350
            )
            with gr.Row():
                dl_c_png = gr.File(label="⬇ PNG (300 dpi)")
                dl_c_pdf = gr.File(label="⬇ PDF (vector)")
                dl_c_svg = gr.File(label="⬇ SVG (vector)")
                dl_c_zip = gr.File(label="⬇ ZIP (all formats)")

        gr.Markdown("""
        ---
        **Reading the figures**
        - **IQR band** — 25%–75% quantile across attention heads per layer.
          Narrow band → heads are consistent → model is well-organized.
        - **Dotted vertical lines** — global (K=V shared) layers (Gemma-4 only).
        - **Dashed horizontal lines** — theoretical ideals (r=1, SSR=0, α=1)
          or random baselines (cosU: 1/√d_head · cosV: 1/√d_model).
        - **Super-orthogonality** (Law 4) — cosU(Q–V) and cosU(K–V) sit *below*
          the random baseline; pretraining actively pushes V away from Q/K.
        """)

        # ── Wiring ────────────────────────────────────────────────────────────

        single_plotly_btn.click(
            fn=gen_single_plotly,
            inputs=[single_model, modality_sel, start_l, end_l, show_band_chk],
            outputs=[single_plotly_fig, single_status],
        )
        single_export_btn.click(
            fn=gen_single_export,
            inputs=[single_model, modality_sel, start_l, end_l, show_band_chk],
            outputs=[single_status, single_preview,
                     dl_s_png, dl_s_pdf, dl_s_svg, dl_s_zip],
        )
        cmp_plotly_btn.click(
            fn=gen_compare_plotly,
            inputs=[model_a, model_b, modality_sel,
                    start_l, end_l, show_band_chk, show_delta_chk],
            outputs=[cmp_plotly_fig, cmp_status],
        )
        cmp_export_btn.click(
            fn=gen_compare_export,
            inputs=[model_a, model_b, modality_sel,
                    start_l, end_l, show_band_chk, show_delta_chk],
            outputs=[cmp_status, cmp_preview,
                     dl_c_png, dl_c_pdf, dl_c_svg, dl_c_zip],
        )