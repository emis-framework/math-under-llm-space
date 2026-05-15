# ui/tab_tables.py
"""
Tab6: Tables — Paper-ready table generation for Wang's Five Laws.
Data: language modality, standard layers only, from SQLite DB.
Output: Gradio DataFrames + LaTeX + Markdown + CSV downloads.
"""

import io
import os
import zipfile
import tempfile

import gradio as gr
import pandas as pd

from db.schema import init_db
from db.reader import get_layer_metrics, get_analyzed_models
from core.table_gen import (
    generate_all_tables,
    format_all_latex,
    format_all_markdown,
    TABLE_META,
)

_OUT_DIR = "/tmp/wang_tables"
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


def _load_all_models(choices: list[str]) -> dict[str, pd.DataFrame]:
    """Load language-modality standard-layer data for all selected models."""
    conn = init_db()
    out  = {}
    for mid in choices:
        df = get_layer_metrics(
            conn,
            model_id   = mid,
            modality   = "language",
            layer_type = None,   # keep both; table_gen filters internally
            start_layer= 0,
            end_layer  = 9999,
        )
        if not df.empty:
            out[mid] = df
    return out


def _parse_groups(text: str) -> list[tuple[int, int]]:
    """
    Parse user-defined layer groups.
    Format: "0-11, 12-23, 24-35, 36-47"
    Returns list of (lo, hi) tuples.
    """
    groups = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            try:
                lo_s, hi_s = part.split("-", 1)
                groups.append((int(lo_s.strip()), int(hi_s.strip())))
            except ValueError:
                continue
    return groups if groups else [(0, 11), (12, 23), (24, 35), (36, 47)]


def _save_csv(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(_OUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    return path


def _make_zip(paths: list[str]) -> str:
    zp = os.path.join(_OUT_DIR, "wang_laws_tables.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            if p and os.path.exists(p):
                zf.write(p, os.path.basename(p))
    return zp


def _save_text(content: str, name: str) -> str:
    path = os.path.join(_OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main generation function
# ─────────────────────────────────────────────────────────────────────────────

def generate_tables(
    selected_models: list[str],
    table2_model_a:  str,
    table2_model_b:  str,
    group_text:      str,
    progress=gr.Progress(),
):
    """
    Returns:
      status, t1_df, t2_df, t3_df, t4_df, t5_df, t6_df,
      latex_text, md_text,
      csv_t1, csv_t2, csv_t3, csv_t4, csv_t5, csv_t6,
      latex_file, md_file, zip_file
    """
    EMPTY = tuple([None] * 17)

    if not selected_models:
        return ("❌ Please select at least one model.",) + EMPTY[1:]

    progress(0.05, desc="Loading data from DB...")
    model_dfs = _load_all_models(selected_models)

    if not model_dfs:
        return ("❌ No language-modality data found. Run Tab 2 analysis first.",) + EMPTY[1:]

    progress(0.25, desc="Parsing layer groups...")
    group_bounds = _parse_groups(group_text)

    # Validate Table 2 model selection
    name_a = table2_model_a if table2_model_a in model_dfs else None
    name_b = table2_model_b if table2_model_b in model_dfs else None
    if name_b == name_a:
        name_b = None

    progress(0.40, desc="Computing tables...")
    tables = generate_all_tables(
        model_dfs    = model_dfs,
        group_bounds = group_bounds,
        name_a       = name_a,
        name_b       = name_b,
    )

    progress(0.65, desc="Formatting LaTeX & Markdown...")
    latex_str = format_all_latex(tables)
    md_str    = format_all_markdown(tables)

    progress(0.80, desc="Saving files...")
    csv_paths = {}
    for key, df in tables.items():
        csv_paths[key] = _save_csv(df, f"wang_laws_{key}")

    latex_file = _save_text(latex_str, "wang_laws_tables.tex")
    md_file    = _save_text(md_str,    "wang_laws_tables.md")
    zip_file   = _make_zip(list(csv_paths.values()) + [latex_file, md_file])

    loaded = list(model_dfs.keys())
    status = (
        f"✅ Generated 6 tables  |  {len(loaded)} models loaded\n"
        f"   Models: {', '.join(loaded)}\n"
        f"   Layer groups (Table 2): {group_bounds}\n"
        f"   Table 2 comparison: {name_a or '—'}  vs  {name_b or '—'}\n"
        f"   Note: language modality, standard layers only (global layers excluded)"
    )

    progress(1.0)
    return (
        status,
        tables["t1"], tables["t2"], tables["t3"],
        tables["t4"], tables["t5"], tables["t6"],
        latex_str, md_str,
        csv_paths["t1"], csv_paths["t2"], csv_paths["t3"],
        csv_paths["t4"], csv_paths["t5"], csv_paths["t6"],
        latex_file, md_file, zip_file,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab6 UI
# ─────────────────────────────────────────────────────────────────────────────

def build_tab_tables():
    with gr.Tab("📋 Tables"):
        gr.Markdown("""
        ## Wang's Five Laws — Paper Tables

        One-click generation of all 6 tables. Data: **language modality, standard layers only**
        (global/K=V-shared layers excluded from all metrics).

        | Table | Content | Law |
        |-------|---------|-----|
        | 1 | Cross-model summary: Pearson r, SSR (Wang Score in Table 6) | 1 & 2 |
        | 2 | SSR layer-group trend (RL effect, user-defined groups) | 2 |
        | 3 | Output subspace cosU: Q–K, Q–V, K–V + random baseline | 4 |
        | 4 | Input subspace cosV: Q–K, Q–V, K–V + random baseline | 5 |
        | 5 | Condition number κ: median all layers + Layer 0 + deep layers | 3 |
        | 6 | Wang Score leaderboard (ranked) | 1 & 2 |

        > Run **Tab 2 (Analyze)** first to populate the database.
        """)

        # ── Model selector with Refresh ───────────────────────────────────────
        def _refresh_choices():
            new_choices = _get_model_choices()
            return (
                gr.CheckboxGroup(choices=new_choices, value=new_choices),
                gr.Dropdown(choices=new_choices),
                gr.Dropdown(choices=new_choices),
            )

        init_choices = _get_model_choices()

        # ── Controls ──────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    gr.Markdown("**Models to include** (all tables use selected models)")
                    refresh_btn = gr.Button("🔄 Refresh", scale=0, min_width=100)
                model_selector = gr.CheckboxGroup(
                    choices  = init_choices,
                    value    = init_choices,
                    label    = "",
                    show_label=False,
                )
            with gr.Column(scale=2):
                gr.Markdown("**Table 2 — SSR Layer-Group Comparison**")
                t2_model_a = gr.Dropdown(
                    choices          = init_choices,
                    value            = init_choices[0] if init_choices else None,
                    allow_custom_value=True,
                    label            = "Model A (base)",
                )
                t2_model_b = gr.Dropdown(
                    choices          = init_choices,
                    value            = init_choices[1] if len(init_choices) > 1 else None,
                    allow_custom_value=True,
                    label            = "Model B (RL-tuned / comparison)",
                    info             = "Leave same as A for single-model view",
                )
                group_input = gr.Textbox(
                    label       = "Layer groups (comma-separated lo-hi pairs)",
                    value       = "0-11, 12-23, 24-35, 36-47",
                    placeholder = "0-11, 12-23, 24-35, 36-47",
                    info        = "Adjust for model depth: 32-layer→0-7,8-15,16-23,24-31  60-layer→0-14,15-29,30-44,45-59",
                )

        generate_btn = gr.Button("🚀 Generate All Tables", variant="primary")
        status_box   = gr.Textbox(lines=4, interactive=False, label="Status")

        gr.Markdown("---")

        # ── Table displays ────────────────────────────────────────────────────
        with gr.Accordion("📊 Table 1 — Cross-Model Summary (Law 1 & 2)", open=True):
            t1_df = gr.Dataframe(interactive=False, wrap=True)
            dl_t1 = gr.File(label="⬇ CSV")

        with gr.Accordion("📊 Table 2 — SSR Layer-Group Trend (Law 2)", open=True):
            t2_df = gr.Dataframe(interactive=False, wrap=True)
            dl_t2 = gr.File(label="⬇ CSV")

        with gr.Accordion("📊 Table 3 — Output Subspace cosU (Law 4)", open=True):
            t3_df = gr.Dataframe(interactive=False, wrap=True)
            dl_t3 = gr.File(label="⬇ CSV")

        with gr.Accordion("📊 Table 4 — Input Subspace cosV (Law 5)", open=True):
            t4_df = gr.Dataframe(interactive=False, wrap=True)
            dl_t4 = gr.File(label="⬇ CSV")

        with gr.Accordion("📊 Table 5 — Condition Number κ (Law 3)", open=True):
            t5_df = gr.Dataframe(interactive=False, wrap=True)
            dl_t5 = gr.File(label="⬇ CSV")

        with gr.Accordion("📊 Table 6 — Wang Score Leaderboard", open=True):
            t6_df = gr.Dataframe(interactive=False, wrap=True)
            dl_t6 = gr.File(label="⬇ CSV")

        gr.Markdown("---")

        # ── Text outputs ──────────────────────────────────────────────────────
        with gr.Accordion("📄 LaTeX Output (paste into .tex)", open=False):
            latex_box = gr.Code(
                language   = "latex",
                label      = "LaTeX tables (booktabs style)",
                interactive= False,
                lines      = 30,
            )

        with gr.Accordion("📄 Markdown Output (paste into README)", open=False):
            md_box = gr.Code(
                language   = "markdown",
                label      = "Markdown tables (GitHub-flavored)",
                interactive= False,
                lines      = 30,
            )

        gr.Markdown("---")

        # ── Bulk downloads ────────────────────────────────────────────────────
        gr.Markdown("### ⬇ Bulk Downloads")
        with gr.Row():
            dl_latex = gr.File(label="⬇ wang_laws_tables.tex")
            dl_md    = gr.File(label="⬇ wang_laws_tables.md")
            dl_zip   = gr.File(label="⬇ ZIP (all CSVs + LaTeX + Markdown)")

        gr.Markdown("""
        ---
        **Notes**
        - All tables use **language modality** and **standard layers only**.
          Global (K=V-shared) layers are excluded from metrics but their count is shown in Table 1.
        - Table 2 layer groups are user-defined. Suggested defaults:
          48-layer models → `0-11, 12-23, 24-35, 36-47`
          32-layer models → `0-7, 8-15, 16-23, 24-31`
          60-layer models → `0-14, 15-29, 30-44, 45-59`
        - LaTeX output uses `booktabs` style (`\\toprule`, `\\midrule`, `\\bottomrule`).
          Add `\\usepackage{booktabs}` to your preamble.
        - Wang Score = 1 − median(SSR_QK). Theoretical maximum = 1.
        """)

        # ── Wiring ────────────────────────────────────────────────────────────
        generate_btn.click(
            fn      = generate_tables,
            inputs  = [model_selector, t2_model_a, t2_model_b, group_input],
            outputs = [
                status_box,
                t1_df, t2_df, t3_df, t4_df, t5_df, t6_df,
                latex_box, md_box,
                dl_t1, dl_t2, dl_t3, dl_t4, dl_t5, dl_t6,
                dl_latex, dl_md, dl_zip,
            ],
        )

        refresh_btn.click(
            fn      = _refresh_choices,
            inputs  = [],
            outputs = [model_selector, t2_model_a, t2_model_b],
        )