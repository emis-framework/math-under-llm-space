# core/plotter.py
"""
Publication-quality figure generation for Wang's Five Laws.
Standards: Nature / PRL / top-conference level.
Canvas: 18×20 inches @ 300 DPI, Arial/Helvetica fonts.

Color system:
  Q-related  → blue  (#2166AC)
  K-related  → red   (#D6604D)
  V-related  → green (#4DAC26)
  QK pair    → purple (#762A83)
  QV pair    → cyan   (#01665E)
  KV pair    → orange (#E08214)
  Model A (base)   → solid line
  Model B (RL)     → dashed line
  Delta            → gray fill
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import io
import os

# ── Font & style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",   # fallback; Arial not always present
    "font.size":          9,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "axes.linewidth":     0.8,
    "grid.linewidth":     0.4,
    "lines.linewidth":    1.5,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "0.7",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# ── Color palette ─────────────────────────────────────────────────────────────
C = {
    "Q":   "#2166AC",   # blue
    "K":   "#D6604D",   # red
    "V":   "#4DAC26",   # green
    "QK":  "#762A83",   # purple
    "QV":  "#01665E",   # cyan/teal
    "KV":  "#E08214",   # orange
    "ref": "#555555",   # reference line (gray)
    "band_alpha": 0.18,
}

BAND_COLORS = {
    "Q":  "#2166AC",
    "K":  "#D6604D",
    "QK": "#762A83",
    "QV": "#01665E",
    "KV": "#E08214",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_by_layer(df: pd.DataFrame, col: str):
    """
    Pseudo-bulk two-step aggregation per layer (Nature Comms 2021).
    Step 1: median across Q heads within each (layer, kv_head) group.
    Step 2: median / q25 / q75 across kv_head groups per layer.
    Avoids pseudoreplication bias in GQA models (e.g. 4Q:1K).
    Excludes kv_shared rows for KV metrics (theoretical-value bias).
    """
    kv_cols = {"ssr_KV", "pearson_KV", "cosU_KV", "cosV_KV", "alpha_KV"}
    if col in kv_cols:
        df = df[df["kv_shared"] == 0] if "kv_shared" in df.columns else df

    layers = np.array(sorted(df["layer"].unique()))
    med_vals, q25_vals, q75_vals = [], [], []

    for layer in layers:
        ldf = df[df["layer"] == layer]
        # Step 1: median within each kv_head group
        if "kv_head" in ldf.columns:
            step1 = ldf.groupby("kv_head")[col].median().values
        else:
            step1 = ldf[col].dropna().values
        step1 = step1[~np.isnan(step1)] if len(step1) > 0 else step1
        # Step 2: statistics across kv_head medians
        med_vals.append(float(np.median(step1)) if len(step1) > 0 else np.nan)
        q25_vals.append(float(np.percentile(step1, 25)) if len(step1) > 0 else np.nan)
        q75_vals.append(float(np.percentile(step1, 75)) if len(step1) > 0 else np.nan)

    return layers, np.array(med_vals), np.array(q25_vals), np.array(q75_vals)


def _global_layers(df: pd.DataFrame):
    """Return list of layer indices where kv_shared==True (Gemma global layers)."""
    if "kv_shared" not in df.columns:
        return []
    return sorted(df[df["kv_shared"] == 1]["layer"].unique().tolist())


# ─────────────────────────────────────────────────────────────────────────────
# Single-subplot drawing primitives
# ─────────────────────────────────────────────────────────────────────────────

def _draw_line(ax, layers, med, q25, q75, color, label, linestyle="-",
               show_band=True, global_layers=None):
    ax.plot(layers, med, color=color, linestyle=linestyle,
            linewidth=1.8, label=label, zorder=3)
    if show_band:
        ax.fill_between(layers, q25, q75, color=color,
                        alpha=C["band_alpha"], zorder=2)
    if global_layers:
        for gl in global_layers:
            ax.axvline(gl, color="#AAAAAA", linewidth=0.7,
                       linestyle=":", zorder=1)


def _add_hline(ax, y, label=None, color=None):
    color = color or C["ref"]
    ax.axhline(y, color=color, linewidth=1.0, linestyle="--",
               alpha=0.75, zorder=1, label=label)


def _finalize_ax(ax, title, ylabel, xlabel="Layer index"):
    ax.set_title(title, fontweight="bold", pad=4)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(loc="best", handlelength=1.5)


# ─────────────────────────────────────────────────────────────────────────────
# The 12-panel 4×3 figure  (single model)
# ─────────────────────────────────────────────────────────────────────────────

def plot_single_model(
    df:         pd.DataFrame,
    model_name: str,
    show_band:  bool = True,
    head_dim:   int  = 128,
    d_model:    int  = 5120,
) -> plt.Figure:
    """
    4×3 grid, 12 subplots.

    Row 1 — Law 1 & 2 (singular value metrics):
      [0,0] pearson_QK   [0,1] ssr_QK      [0,2] alpha_QK

    Row 2 — Law 3 (condition numbers & max singular values):
      [1,0] sigma_max_Q  [1,1] sigma_max_K  [1,2] cond_Q & cond_K (dual line)

    Row 3 — Law 4 (output subspace, left singular vectors U):
      [2,0] cosU_QK      [2,1] cosU_QV      [2,2] cosU_KV
      + random baseline 1/√d_head

    Row 4 — Law 5 (input subspace, right singular vectors V):
      [3,0] cosV_QK      [3,1] cosV_QV      [3,2] cosV_KV
      + random baseline 1/√d_model
    """
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(
        f"Wang's Five Laws — {model_name}",
        fontsize=14, fontweight="bold", y=0.995
    )

    gl = _global_layers(df)
    baseline_U = 1.0 / np.sqrt(head_dim)
    baseline_V = 1.0 / np.sqrt(d_model)

    # ── helper ───────────────────────────────────────────────────────────────
    def draw(ax, col, color, label, linestyle="-"):
        layers, med, q25, q75 = _aggregate_by_layer(df, col)
        _draw_line(ax, layers, med, q25, q75, color, label,
                   linestyle=linestyle, show_band=show_band,
                   global_layers=gl)

    # ── Row 0: Law 1 & 2 ─────────────────────────────────────────────────────
    ax = axes[0, 0]
    draw(ax, "pearson_QK", C["QK"], "Pearson r (Q–K)")
    _add_hline(ax, 1.0, "Ideal = 1")
    _finalize_ax(ax, "Law 1 — Spectral Linear Alignment",
                 "Pearson r (Q, K spectra)")

    ax = axes[0, 1]
    draw(ax, "ssr_QK", C["QK"], "SSR (Q–K)")
    _add_hline(ax, 0.0, "Ideal = 0")
    _finalize_ax(ax, "Law 2 — Spectral Shape Fidelity",
                 "SSR (Q–K normalized)")

    ax = axes[0, 2]
    draw(ax, "alpha_QK", C["QK"], "α (Q–K)")
    _add_hline(ax, 1.0, "Ideal = 1")
    _finalize_ax(ax, "Law 1+2 — Scale Factor α (Q–K)",
                 "Scale factor α")

    # ── Row 1: Law 3 ─────────────────────────────────────────────────────────
    ax = axes[1, 0]
    draw(ax, "sigma_max_Q", C["Q"], "σ_max (Q)")
    _finalize_ax(ax, "Law 3 — Max Singular Value (Q)",
                 "σ_max")

    ax = axes[1, 1]
    draw(ax, "sigma_max_K", C["K"], "σ_max (K)")
    _finalize_ax(ax, "Law 3 — Max Singular Value (K)",
                 "σ_max")

    ax = axes[1, 2]
    draw(ax, "cond_Q", C["Q"], "κ(Q)")
    draw(ax, "cond_K", C["K"], "κ(K)")
    ax.set_yscale("log")
    _finalize_ax(ax, "Law 3 — Condition Number κ  (log scale)",
                 "Condition number κ  (log)")

    # ── Row 2: Law 4 ─────────────────────────────────────────────────────────
    # Share y-axis across this row
    axU = [axes[2, 0], axes[2, 1], axes[2, 2]]
    u_data = {}
    for col in ["cosU_QK", "cosU_QV", "cosU_KV"]:
        _, med, q25, q75 = _aggregate_by_layer(df, col)
        u_data[col] = (med, q25, q75)
    all_u = np.concatenate([np.concatenate([v[1], v[2]]) for v in u_data.values()])
    all_u = all_u[~np.isnan(all_u)]
    if len(all_u) > 0:
        u_ymin = max(0, np.nanmin(all_u) * 0.92)
        u_ymax = np.nanmax(all_u) * 1.08
    else:
        u_ymin, u_ymax = 0, 0.15

    for (col, color, title_suffix), ax in zip(
        [("cosU_QK", C["QK"], "Q–K"),
         ("cosU_QV", C["QV"], "Q–V"),
         ("cosU_KV", C["KV"], "K–V")],
        axU
    ):
        draw(ax, col, color, f"cosU ({title_suffix})")
        _add_hline(ax, baseline_U,
                   f"Random = 1/√d_h ≈ {baseline_U:.4f}")
        ax.set_ylim(u_ymin, u_ymax)
        _finalize_ax(ax, f"Law 4 — Output Subspace cosU ({title_suffix})",
                     "Mean |cos| (left singular vectors)")

    # ── Row 3: Law 5 ─────────────────────────────────────────────────────────
    axV = [axes[3, 0], axes[3, 1], axes[3, 2]]
    v_data = {}
    for col in ["cosV_QK", "cosV_QV", "cosV_KV"]:
        _, med, q25, q75 = _aggregate_by_layer(df, col)
        v_data[col] = (med, q25, q75)
    all_v = np.concatenate([np.concatenate([v[1], v[2]]) for v in v_data.values()])
    all_v = all_v[~np.isnan(all_v)]
    if len(all_v) > 0:
        v_ymin = max(0, np.nanmin(all_v) * 0.92)
        v_ymax = np.nanmax(all_v) * 1.08
    else:
        v_ymin, v_ymax = 0, 0.05

    for (col, color, title_suffix), ax in zip(
        [("cosV_QK", C["QK"], "Q–K"),
         ("cosV_QV", C["QV"], "Q–V"),
         ("cosV_KV", C["KV"], "K–V")],
        axV
    ):
        draw(ax, col, color, f"cosV ({title_suffix})")
        _add_hline(ax, baseline_V,
                   f"Random = 1/√D ≈ {baseline_V:.4f}")
        ax.set_ylim(v_ymin, v_ymax)
        _finalize_ax(ax, f"Law 5 — Input Subspace cosV ({title_suffix})",
                     "Mean |cos| (right singular vectors)")

    # ── Global layer legend ───────────────────────────────────────────────────
    if gl:
        fig.text(
            0.5, 0.001,
            f"Vertical dotted lines mark global (K=V shared) layers: {gl}",
            ha="center", fontsize=8, color="#666666"
        )

    fig.tight_layout(rect=[0, 0.01, 1, 0.995])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Two-model comparison figure  (same 4×3, dual lines + delta subpanels)
# ─────────────────────────────────────────────────────────────────────────────

def plot_compare_models(
    df_a:        pd.DataFrame,
    df_b:        pd.DataFrame,
    name_a:      str,
    name_b:      str,
    show_band:   bool = True,
    show_delta:  bool = True,
    head_dim:    int  = 128,
    d_model:     int  = 5120,
) -> plt.Figure:
    """
    4×3 comparison grid.
    Each subplot: Model A (solid) vs Model B (dashed).
    Delta (B - A) shown as gray fill when show_delta=True.
    """
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(
        f"Wang's Five Laws — {name_a}  vs  {name_b}",
        fontsize=14, fontweight="bold", y=0.995
    )

    gl_a = _global_layers(df_a)
    gl_b = _global_layers(df_b)
    gl   = sorted(set(gl_a) | set(gl_b))

    baseline_U = 1.0 / np.sqrt(head_dim)
    baseline_V = 1.0 / np.sqrt(d_model)

    def draw_pair(ax, col, color, label_a, label_b, hline=None, hline_label=None):
        """Draw Model A (solid) and Model B (dashed) on the same axes."""
        lay_a, med_a, q25_a, q75_a = _aggregate_by_layer(df_a, col)
        lay_b, med_b, q25_b, q75_b = _aggregate_by_layer(df_b, col)

        _draw_line(ax, lay_a, med_a, q25_a, q75_a, color, label_a,
                   linestyle="-", show_band=show_band, global_layers=gl)
        _draw_line(ax, lay_b, med_b, q25_b, q75_b, color, label_b,
                   linestyle="--", show_band=show_band, global_layers=None)

        # Delta fill
        if show_delta:
            common = np.intersect1d(lay_a, lay_b)
            if len(common) > 1:
                idx_a = np.isin(lay_a, common)
                idx_b = np.isin(lay_b, common)
                delta = med_b[idx_b] - med_a[idx_a]
                pos   = np.maximum(delta, 0)
                neg   = np.minimum(delta, 0)
                ax.fill_between(common, 0, pos,
                                color="#AAAAAA", alpha=0.25, zorder=0)
                ax.fill_between(common, 0, neg,
                                color="#AAAAAA", alpha=0.25, zorder=0)

        if hline is not None:
            _add_hline(ax, hline, hline_label)

    # ── Row 0 ────────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    draw_pair(ax, "pearson_QK", C["QK"],
              f"{name_a} Pearson r", f"{name_b} Pearson r", hline=1.0, hline_label="Ideal=1")
    _finalize_ax(ax, "Law 1 — Spectral Linear Alignment", "Pearson r (Q, K)")

    ax = axes[0, 1]
    draw_pair(ax, "ssr_QK", C["QK"],
              f"{name_a} SSR", f"{name_b} SSR", hline=0.0, hline_label="Ideal=0")
    _finalize_ax(ax, "Law 2 — Spectral Shape Fidelity", "SSR (Q–K)")

    ax = axes[0, 2]
    draw_pair(ax, "alpha_QK", C["QK"],
              f"{name_a} α", f"{name_b} α", hline=1.0, hline_label="Ideal=1")
    _finalize_ax(ax, "Law 1+2 — Scale Factor α (Q–K)", "Scale factor α")

    # ── Row 1 ────────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    draw_pair(ax, "sigma_max_Q", C["Q"],
              f"{name_a} σ_max(Q)", f"{name_b} σ_max(Q)")
    _finalize_ax(ax, "Law 3 — Max Singular Value (Q)", "σ_max")

    ax = axes[1, 1]
    draw_pair(ax, "sigma_max_K", C["K"],
              f"{name_a} σ_max(K)", f"{name_b} σ_max(K)")
    _finalize_ax(ax, "Law 3 — Max Singular Value (K)", "σ_max")

    ax = axes[1, 2]
    # cond: draw both Q and K for both models → 4 lines
    lay_a, med_a, q25_a, q75_a = _aggregate_by_layer(df_a, "cond_Q")
    lay_b, med_b, q25_b, q75_b = _aggregate_by_layer(df_b, "cond_Q")
    _draw_line(ax, lay_a, med_a, q25_a, q75_a, C["Q"],
               f"{name_a} κ(Q)", "-", show_band, gl)
    _draw_line(ax, lay_b, med_b, q25_b, q75_b, C["Q"],
               f"{name_b} κ(Q)", "--", show_band, None)
    lay_a, med_a, q25_a, q75_a = _aggregate_by_layer(df_a, "cond_K")
    lay_b, med_b, q25_b, q75_b = _aggregate_by_layer(df_b, "cond_K")
    _draw_line(ax, lay_a, med_a, q25_a, q75_a, C["K"],
               f"{name_a} κ(K)", "-", show_band, None)
    _draw_line(ax, lay_b, med_b, q25_b, q75_b, C["K"],
               f"{name_b} κ(K)", "--", show_band, None)
    ax.set_yscale("log")
    _finalize_ax(ax, "Law 3 — Condition Number κ  (log scale)", "Condition number κ  (log)")

    # ── Row 2: Law 4 ─────────────────────────────────────────────────────────
    u_cols = [("cosU_QK", C["QK"], "Q–K"),
              ("cosU_QV", C["QV"], "Q–V"),
              ("cosU_KV", C["KV"], "K–V")]

    # Compute shared y range
    u_vals = []
    for col, _, _ in u_cols:
        for df_ in [df_a, df_b]:
            _, med, q25, q75 = _aggregate_by_layer(df_, col)
            u_vals.extend(q25[~np.isnan(q25)].tolist())
            u_vals.extend(q75[~np.isnan(q75)].tolist())
    u_ymin = max(0, min(u_vals) * 0.92) if u_vals else 0
    u_ymax = (max(u_vals) * 1.08) if u_vals else 0.15

    for (col, color, suffix), ax in zip(u_cols, axes[2]):
        draw_pair(ax, col, color,
                  f"{name_a}", f"{name_b}",
                  hline=baseline_U,
                  hline_label=f"Random 1/√d_h ≈ {baseline_U:.4f}")
        ax.set_ylim(u_ymin, u_ymax)
        _finalize_ax(ax, f"Law 4 — cosU ({suffix})",
                     "Mean |cos| (U)")

    # ── Row 3: Law 5 ─────────────────────────────────────────────────────────
    v_cols = [("cosV_QK", C["QK"], "Q–K"),
              ("cosV_QV", C["QV"], "Q–V"),
              ("cosV_KV", C["KV"], "K–V")]

    v_vals = []
    for col, _, _ in v_cols:
        for df_ in [df_a, df_b]:
            _, med, q25, q75 = _aggregate_by_layer(df_, col)
            v_vals.extend(q25[~np.isnan(q25)].tolist())
            v_vals.extend(q75[~np.isnan(q75)].tolist())
    v_ymin = max(0, min(v_vals) * 0.92) if v_vals else 0
    v_ymax = (max(v_vals) * 1.08) if v_vals else 0.05

    for (col, color, suffix), ax in zip(v_cols, axes[3]):
        draw_pair(ax, col, color,
                  f"{name_a}", f"{name_b}",
                  hline=baseline_V,
                  hline_label=f"Random 1/√D ≈ {baseline_V:.4f}")
        ax.set_ylim(v_ymin, v_ymax)
        _finalize_ax(ax, f"Law 5 — cosV ({suffix})",
                     "Mean |cos| (V)")

    # ── Legend for line styles ────────────────────────────────────────────────
    solid_patch  = Line2D([0], [0], color="#333333", linewidth=1.8,
                          linestyle="-",  label=f"Solid = {name_a}")
    dashed_patch = Line2D([0], [0], color="#333333", linewidth=1.8,
                          linestyle="--", label=f"Dashed = {name_b}")
    fig.legend(handles=[solid_patch, dashed_patch],
               loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 0.001))

    if gl:
        fig.text(
            0.5, 0.0045,
            f"Vertical dotted lines mark global (K=V shared) layers: {gl}",
            ha="center", fontsize=8, color="#666666"
        )

    fig.tight_layout(rect=[0, 0.015, 1, 0.995])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_figure(fig: plt.Figure, base_path: str):
    """
    Save figure to PNG (300 dpi), PDF (vector), and SVG (vector).
    base_path: path without extension, e.g. "/tmp/wang_laws_gemma"
    Returns list of saved file paths.
    """
    paths = []
    for fmt, kwargs in [
        ("png", {"dpi": 300, "bbox_inches": "tight"}),
        ("pdf", {"bbox_inches": "tight"}),
        ("svg", {"bbox_inches": "tight"}),
    ]:
        p = f"{base_path}.{fmt}"
        fig.savefig(p, format=fmt, **kwargs)
        paths.append(p)
    return paths


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    """Return PNG bytes for Gradio Image component."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# fig_to_plotly removed — use core/plotter_plotly.py for native Plotly figures.