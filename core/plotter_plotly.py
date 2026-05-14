# core/plotter_plotly.py
"""
Native Plotly interactive figures for Wang's Five Laws.
12 subplots stacked vertically (12×1), full browser width.
Fast: data aggregated once, drawn directly — no matplotlib conversion.

Layout (top → bottom):
  0  pearson_QK      Law 1  Spectral Linear Alignment
  1  ssr_QK          Law 2  Spectral Shape Fidelity
  2  alpha_QK        Law 1+2  Scale Factor α
  3  sigma_max_Q     Law 3  Max Singular Value (Q)
  4  sigma_max_K     Law 3  Max Singular Value (K)
  5  cond_Q + cond_K Law 3  Condition Number κ  (dual line)
  6  cosU_QK         Law 4  Output Subspace Q–K
  7  cosU_QV         Law 4  Output Subspace Q–V
  8  cosU_KV         Law 4  Output Subspace K–V
  9  cosV_QK         Law 5  Input Subspace Q–K
  10 cosV_QV         Law 5  Input Subspace Q–V
  11 cosV_KV         Law 5  Input Subspace K–V
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Color palette (identical to plotter.py) ───────────────────────────────────
C = {
    "Q":   "#2166AC",
    "K":   "#D6604D",
    "V":   "#4DAC26",
    "QK":  "#762A83",
    "QV":  "#01665E",
    "KV":  "#E08214",
    "ref": "#888888",
}

BAND_ALPHA = 0.15   # opacity for IQR band fill

# ── Panel definitions ─────────────────────────────────────────────────────────
# (col, color_key, y_label, title, ideal_value or None)
PANELS = [
    ("pearson_QK",  "QK", "Pearson r",   "Law 1 — Spectral Linear Alignment (Pearson r Q–K)",   1.0),
    ("ssr_QK",      "QK", "SSR",         "Law 2 — Spectral Shape Fidelity (SSR Q–K)",             0.0),
    ("alpha_QK",    "QK", "α",           "Law 1+2 — Scale Factor α (Q–K)",                        1.0),
    ("sigma_max_Q", "Q",  "σ_max",       "Law 3 — Max Singular Value σ_max (Q)",                 None),
    ("sigma_max_K", "K",  "σ_max",       "Law 3 — Max Singular Value σ_max (K)",                 None),
    ("cond_dual",   None, "κ",           "Law 3 — Condition Number κ (Q & K)",                   None),
    ("cosU_QK",     "QK", "cosU",        "Law 4 — Output Subspace cosU (Q–K)",                   None),
    ("cosU_QV",     "QV", "cosU",        "Law 4 — Output Subspace cosU (Q–V)  [super-orth]",     None),
    ("cosU_KV",     "KV", "cosU",        "Law 4 — Output Subspace cosU (K–V)  [super-orth]",     None),
    ("cosV_QK",     "QK", "cosV",        "Law 5 — Input Subspace cosV (Q–K)",                    None),
    ("cosV_QV",     "QV", "cosV",        "Law 5 — Input Subspace cosV (Q–V)",                    None),
    ("cosV_KV",     "KV", "cosV",        "Law 5 — Input Subspace cosV (K–V)",                    None),
]

SUBPLOT_HEIGHT = 280   # px per subplot
TOTAL_HEIGHT   = SUBPLOT_HEIGHT * len(PANELS) + 120   # +header


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _agg(df: pd.DataFrame, col: str):
    """Per-layer median + IQR. Excludes kv_shared rows for KV metrics."""
    kv_cols = {"ssr_KV", "pearson_KV", "cosU_KV", "cosV_KV", "alpha_KV"}
    d = df[df["kv_shared"] == 0] if col in kv_cols and "kv_shared" in df.columns else df
    grp    = d.groupby("layer")[col]
    layers = np.array(sorted(d["layer"].unique()), dtype=int)
    med    = grp.median().reindex(layers).values.astype(float)
    q25    = grp.quantile(0.25).reindex(layers).values.astype(float)
    q75    = grp.quantile(0.75).reindex(layers).values.astype(float)
    return layers, med, q25, q75


def _global_layers(df: pd.DataFrame) -> list[int]:
    if "kv_shared" not in df.columns:
        return []
    return sorted(df[df["kv_shared"] == 1]["layer"].unique().tolist())


def _infer_dims(df: pd.DataFrame) -> tuple[int, int]:
    head_dim = int(df["head_dim"].dropna().median()) if "head_dim" in df.columns and df["head_dim"].notna().any() else 128
    d_model  = int(df["d_model"].dropna().median())  if "d_model"  in df.columns and df["d_model"].notna().any()  else 5120
    return head_dim, d_model


# ─────────────────────────────────────────────────────────────────────────────
# Trace builders
# ─────────────────────────────────────────────────────────────────────────────

def _band_traces(layers, med, q25, q75, color, name,
                 row, dash="solid", show_legend=True):
    """Returns (band_trace, line_trace) for one series."""
    rgba_fill = _hex_to_rgba(color, BAND_ALPHA)

    band = go.Scatter(
        x=np.concatenate([layers, layers[::-1]]).tolist(),
        y=np.concatenate([q75, q25[::-1]]).tolist(),
        fill="toself",
        fillcolor=rgba_fill,
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
        legendgroup=name,
    )
    line = go.Scatter(
        x=layers.tolist(),
        y=med.tolist(),
        mode="lines",
        name=name,
        line=dict(color=color, width=2, dash=dash),
        hovertemplate=f"Layer %{{x}}<br>{name}: %{{y:.5f}}<extra></extra>",
        showlegend=show_legend,
        legendgroup=name,
    )
    return band, line


def _hline_trace(layers, y_val, label, color=None, row=None):
    color = color or C["ref"]
    return go.Scatter(
        x=[layers[0], layers[-1]],
        y=[y_val, y_val],
        mode="lines",
        name=label,
        line=dict(color=color, width=1.2, dash="dash"),
        hoverinfo="skip",
        showlegend=True,
        legendgroup=label,
    )


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _vlines(fig, global_layers, row, x_range):
    for gl in global_layers:
        fig.add_vline(
            x=gl, row=row, col=1,
            line=dict(color="#AAAAAA", width=1, dash="dot"),
            annotation=dict(
                text=f"G{gl}", font=dict(size=8, color="#999999"),
                showarrow=False, yref="paper",
            ) if row == 1 else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Single-model native Plotly figure
# ─────────────────────────────────────────────────────────────────────────────

def plotly_single(
    df:         pd.DataFrame,
    model_name: str,
    show_band:  bool = True,
) -> go.Figure:
    """
    12×1 stacked subplots, full browser width.
    Each subplot: median line + IQR band + reference lines + global-layer markers.
    """
    n_panels = len(PANELS)
    head_dim, d_model = _infer_dims(df)
    baseline_U = 1.0 / np.sqrt(head_dim)
    baseline_V = 1.0 / np.sqrt(d_model)
    gl         = _global_layers(df)

    subtitles = [p[3] for p in PANELS]
    fig = make_subplots(
        rows=n_panels, cols=1,
        subplot_titles=subtitles,
        shared_xaxes=False,
        vertical_spacing=0.03,
    )

    for row_idx, (col, color_key, ylabel, title, ideal) in enumerate(PANELS, start=1):
        color = C[color_key] if color_key else C["Q"]

        # ── special case: cond_dual ──────────────────────────────────────────
        if col == "cond_dual":
            for c_col, c_key, c_name in [
                ("cond_Q", "Q", "κ(Q)"),
                ("cond_K", "K", "κ(K)"),
            ]:
                layers, med, q25, q75 = _agg(df, c_col)
                if len(layers) == 0:
                    continue
                band, line = _band_traces(
                    layers, med, q25, q75, C[c_key], c_name,
                    row=row_idx, show_legend=True
                )
                if show_band:
                    fig.add_trace(band, row=row_idx, col=1)
                fig.add_trace(line, row=row_idx, col=1)
            layers_ref = _agg(df, "cond_Q")[0]

        else:
            layers, med, q25, q75 = _agg(df, col)
            if len(layers) == 0:
                continue
            band, line = _band_traces(
                layers, med, q25, q75, color,
                model_name, row=row_idx, show_legend=(row_idx == 1)
            )
            if show_band:
                fig.add_trace(band, row=row_idx, col=1)
            fig.add_trace(line, row=row_idx, col=1)
            layers_ref = layers

            # ── ideal / baseline reference lines ─────────────────────────────
            if ideal is not None and len(layers_ref):
                fig.add_trace(
                    _hline_trace(layers_ref, ideal, f"Ideal={ideal}",
                                 color=C["ref"]),
                    row=row_idx, col=1
                )

        # ── random baselines for cosU / cosV ────────────────────────────────
        if col.startswith("cosU_") and len(layers_ref):
            fig.add_trace(
                _hline_trace(layers_ref, baseline_U,
                             f"Random 1/√d_h ≈ {baseline_U:.4f}",
                             color="#E07B39"),
                row=row_idx, col=1
            )
        if col.startswith("cosV_") and len(layers_ref):
            fig.add_trace(
                _hline_trace(layers_ref, baseline_V,
                             f"Random 1/√D ≈ {baseline_V:.4f}",
                             color="#E07B39"),
                row=row_idx, col=1
            )

        # ── global layer vertical markers ────────────────────────────────────
        for gl_idx in gl:
            fig.add_vline(
                x=gl_idx, row=row_idx, col=1,
                line=dict(color="#BBBBBB", width=1, dash="dot"),
            )

        # ── y-axis label ─────────────────────────────────────────────────────
        fig.update_yaxes(title_text=ylabel, row=row_idx, col=1,
                         title_font=dict(size=11))
        fig.update_xaxes(title_text="Layer index", row=row_idx, col=1,
                         title_font=dict(size=11))

    # ── shared Y for cosU row (panels 6,7,8) ─────────────────────────────────
    _sync_yrange(fig, df, ["cosU_QK", "cosU_QV", "cosU_KV"],
                 rows=[7, 8, 9], pad=0.08)
    # ── shared Y for cosV row (panels 9,10,11) ───────────────────────────────
    _sync_yrange(fig, df, ["cosV_QK", "cosV_QV", "cosV_KV"],
                 rows=[10, 11, 12], pad=0.08)

    # ── layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"<b>Wang's Five Laws — {model_name}</b>",
            font=dict(size=16),
            x=0.5, xanchor="center",
        ),
        height=TOTAL_HEIGHT,
        width=None,          # full browser width
        autosize=True,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right",  x=1,
            font=dict(size=10),
        ),
        margin=dict(l=70, r=30, t=80, b=40),
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        font=dict(family="Arial, sans-serif", size=11),
        hovermode="x unified",
    )
    fig.update_annotations(font_size=11)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Two-model comparison native Plotly figure
# ─────────────────────────────────────────────────────────────────────────────

def plotly_compare(
    df_a:       pd.DataFrame,
    df_b:       pd.DataFrame,
    name_a:     str,
    name_b:     str,
    show_band:  bool = True,
    show_delta: bool = True,
) -> go.Figure:
    """
    12×1 stacked subplots.
    Model A: solid lines.  Model B: dashed lines.
    Δ = B − A shown as light gray fill when show_delta=True.
    """
    n_panels = len(PANELS)
    head_dim_a, d_model_a = _infer_dims(df_a)
    head_dim_b, d_model_b = _infer_dims(df_b)
    head_dim   = (head_dim_a + head_dim_b) // 2
    d_model    = (d_model_a  + d_model_b)  // 2
    baseline_U = 1.0 / np.sqrt(head_dim)
    baseline_V = 1.0 / np.sqrt(d_model)
    gl = sorted(set(_global_layers(df_a)) | set(_global_layers(df_b)))

    subtitles = [p[3] for p in PANELS]
    fig = make_subplots(
        rows=n_panels, cols=1,
        subplot_titles=subtitles,
        shared_xaxes=False,
        vertical_spacing=0.03,
    )

    for row_idx, (col, color_key, ylabel, title, ideal) in enumerate(PANELS, start=1):
        color = C[color_key] if color_key else C["Q"]

        if col == "cond_dual":
            for c_col, c_key, c_name in [
                ("cond_Q", "Q", "κ(Q)"),
                ("cond_K", "K", "κ(K)"),
            ]:
                for df_, nm, dash in [(df_a, name_a, "solid"),
                                      (df_b, name_b, "dash")]:
                    layers, med, q25, q75 = _agg(df_, c_col)
                    if len(layers) == 0:
                        continue
                    label = f"{c_name} {nm}"
                    band, line = _band_traces(
                        layers, med, q25, q75, C[c_key], label,
                        row=row_idx, dash=dash, show_legend=True
                    )
                    if show_band:
                        fig.add_trace(band, row=row_idx, col=1)
                    fig.add_trace(line, row=row_idx, col=1)
            layers_ref = _agg(df_a, "cond_Q")[0]

        else:
            layers_a, med_a, q25_a, q75_a = _agg(df_a, col)
            layers_b, med_b, q25_b, q75_b = _agg(df_b, col)

            for layers, med, q25, q75, nm, dash in [
                (layers_a, med_a, q25_a, q75_a, name_a, "solid"),
                (layers_b, med_b, q25_b, q75_b, name_b, "dash"),
            ]:
                if len(layers) == 0:
                    continue
                show_leg = (row_idx == 1)
                band, line = _band_traces(
                    layers, med, q25, q75, color, nm,
                    row=row_idx, dash=dash, show_legend=show_leg
                )
                if show_band:
                    fig.add_trace(band, row=row_idx, col=1)
                fig.add_trace(line, row=row_idx, col=1)

            # Delta fill
            if show_delta and len(layers_a) and len(layers_b):
                common = np.intersect1d(layers_a, layers_b)
                if len(common) > 1:
                    idx_a = np.isin(layers_a, common)
                    idx_b = np.isin(layers_b, common)
                    delta = med_b[idx_b] - med_a[idx_a]
                    zero  = np.zeros_like(delta)
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([common, common[::-1]]).tolist(),
                        y=np.concatenate([delta, zero[::-1]]).tolist(),
                        fill="toself",
                        fillcolor="rgba(160,160,160,0.20)",
                        line=dict(color="rgba(0,0,0,0)"),
                        hoverinfo="skip",
                        showlegend=(row_idx == 1),
                        name=f"Δ ({name_b}−{name_a})",
                        legendgroup="delta",
                    ), row=row_idx, col=1)

            layers_ref = layers_a if len(layers_a) else layers_b

            # Reference lines
            if ideal is not None and len(layers_ref):
                fig.add_trace(
                    _hline_trace(layers_ref, ideal, f"Ideal={ideal}", C["ref"]),
                    row=row_idx, col=1
                )

        if col.startswith("cosU_") and len(layers_ref):
            fig.add_trace(
                _hline_trace(layers_ref, baseline_U,
                             f"Random 1/√d_h ≈ {baseline_U:.4f}", "#E07B39"),
                row=row_idx, col=1
            )
        if col.startswith("cosV_") and len(layers_ref):
            fig.add_trace(
                _hline_trace(layers_ref, baseline_V,
                             f"Random 1/√D ≈ {baseline_V:.4f}", "#E07B39"),
                row=row_idx, col=1
            )

        for gl_idx in gl:
            fig.add_vline(
                x=gl_idx, row=row_idx, col=1,
                line=dict(color="#BBBBBB", width=1, dash="dot"),
            )

        fig.update_yaxes(title_text=ylabel, row=row_idx, col=1,
                         title_font=dict(size=11))
        fig.update_xaxes(title_text="Layer index", row=row_idx, col=1,
                         title_font=dict(size=11))

    _sync_yrange_compare(fig, df_a, df_b,
                         ["cosU_QK", "cosU_QV", "cosU_KV"], [7, 8, 9])
    _sync_yrange_compare(fig, df_a, df_b,
                         ["cosV_QK", "cosV_QV", "cosV_KV"], [10, 11, 12])

    fig.update_layout(
        title=dict(
            text=f"<b>Wang's Five Laws — {name_a}  vs  {name_b}</b>",
            font=dict(size=16),
            x=0.5, xanchor="center",
        ),
        height=TOTAL_HEIGHT,
        width=None,
        autosize=True,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right",  x=1,
            font=dict(size=10),
        ),
        margin=dict(l=70, r=30, t=80, b=40),
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        font=dict(family="Arial, sans-serif", size=11),
        hovermode="x unified",
    )
    fig.update_annotations(font_size=11)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Shared Y-axis helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sync_yrange(fig, df, cols, rows, pad=0.08):
    """Force identical y-range for a set of rows (single model)."""
    vals = []
    for col in cols:
        try:
            _, med, q25, q75 = _agg(df, col)
            vals.extend(q25[~np.isnan(q25)].tolist())
            vals.extend(q75[~np.isnan(q75)].tolist())
        except Exception:
            pass
    if not vals:
        return
    lo = max(0.0, min(vals) * (1 - pad))
    hi = max(vals) * (1 + pad)
    for r in rows:
        fig.update_yaxes(range=[lo, hi], row=r, col=1)


def _sync_yrange_compare(fig, df_a, df_b, cols, rows, pad=0.08):
    """Force identical y-range for a set of rows (two-model comparison)."""
    vals = []
    for col in cols:
        for df_ in [df_a, df_b]:
            try:
                _, med, q25, q75 = _agg(df_, col)
                vals.extend(q25[~np.isnan(q25)].tolist())
                vals.extend(q75[~np.isnan(q75)].tolist())
            except Exception:
                pass
    if not vals:
        return
    lo = max(0.0, min(vals) * (1 - pad))
    hi = max(vals) * (1 + pad)
    for r in rows:
        fig.update_yaxes(range=[lo, hi], row=r, col=1)