# core/table_gen.py
"""
Table generation for Wang's Five Laws — paper-ready output.
Pure computation layer: takes DataFrames from db/reader, returns DataFrames + formatted strings.
No UI, no DB, no side effects.

Tables:
  Table 1 — Cross-model summary (Law 1 & 2): Pearson r, SSR, Wang Score
  Table 2 — SSR layer-group trend (Law 2, RL effect): user-defined groups
  Table 3 — Output subspace cosU (Law 4): QK / QV / KV + random baseline
  Table 4 — Input subspace cosV (Law 5): QK / QV / KV + random baseline
  Table 5 — Condition number κ summary (Law 3): cond_Q, cond_K
  Table 6 — Wang Score leaderboard
"""

import numpy as np
import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _med(series) -> Optional[float]:
    v = series.dropna()
    return float(v.median()) if len(v) > 0 else None


def _mean(series) -> Optional[float]:
    v = series.dropna()
    return float(v.mean()) if len(v) > 0 else None


def _fmt(x, decimals=6) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{decimals}f}"


def _short(model_id: str) -> str:
    return model_id.split("/")[-1] if "/" in model_id else model_id


def _standard_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only standard layers (exclude global/KV-shared layers)."""
    if "kv_shared" in df.columns:
        return df[df["kv_shared"] == 0]
    if "layer_type" in df.columns:
        return df[df["layer_type"] == "standard"]
    return df


def _random_baseline_U(df: pd.DataFrame) -> float:
    if "head_dim" in df.columns and df["head_dim"].notna().any():
        return 1.0 / np.sqrt(float(df["head_dim"].dropna().median()))
    return float("nan")


def _random_baseline_V(df: pd.DataFrame) -> float:
    if "d_model" in df.columns and df["d_model"].notna().any():
        return 1.0 / np.sqrt(float(df["d_model"].dropna().median()))
    return float("nan")


def _n_global(df: pd.DataFrame) -> int:
    if "kv_shared" in df.columns:
        return int(df[df["kv_shared"] == 1]["layer"].nunique())
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX / Markdown helpers
# ─────────────────────────────────────────────────────────────────────────────

def df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Convert DataFrame to a complete LaTeX table."""
    cols = list(df.columns)
    n_cols = len(cols)
    col_fmt = "l" + "r" * (n_cols - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_fmt}}}",
        r"    \toprule",
        "    " + " & ".join(str(c) for c in cols) + r" \\",
        r"    \midrule",
    ]
    for _, row in df.iterrows():
        lines.append("    " + " & ".join(str(v) for v in row.values) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def df_to_markdown(df: pd.DataFrame, caption: str) -> str:
    """Convert DataFrame to GitHub-flavored Markdown table."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    rows   = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row.values) + " |")
    lines = [f"**{caption}**", "", header, sep] + rows
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 — Cross-model summary (Law 1 & 2)
# ─────────────────────────────────────────────────────────────────────────────

def make_table1(
    model_dfs: dict[str, pd.DataFrame],  # {model_id: full_df from DB}
) -> pd.DataFrame:
    """
    One row per model.
    Columns: Model | Layers | Global | Median Pearson | Mean Pearson | Median SSR | Mean SSR | Wang Score
    Uses standard layers only.
    """
    rows = []
    for model_id, df in model_dfs.items():
        std = _standard_only(df)
        if std.empty:
            continue
        n_layers = std["layer"].nunique()
        n_global = _n_global(df)
        rows.append({
            "Model":         _short(model_id),
            "Std Layers":    n_layers,
            "Global Layers": n_global if n_global > 0 else "—",
            "Median Pearson":_fmt(_med(std["pearson_QK"]), 4),
            "Mean Pearson":  _fmt(_mean(std["pearson_QK"]), 4),
            "Median SSR":    _fmt(_med(std["ssr_QK"]), 6),
            "Mean SSR":      _fmt(_mean(std["ssr_QK"]), 6),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Table 2 — SSR layer-group trend (Law 2, RL effect)
# ─────────────────────────────────────────────────────────────────────────────

def make_table2(
    df_a: pd.DataFrame,
    name_a: str,
    df_b: Optional[pd.DataFrame],
    name_b: Optional[str],
    group_bounds: list[tuple[int, int]],  # e.g. [(0,11),(12,23),(24,35),(36,47)]
) -> pd.DataFrame:
    """
    One row per layer group.
    Single model: Model SSR + Layers column.
    Two models: A SSR | B SSR | Improvement %.
    Uses standard layers only.
    """
    std_a = _standard_only(df_a)
    std_b = _standard_only(df_b) if df_b is not None else None

    rows = []
    for lo, hi in group_bounds:
        label = f"{lo}–{hi}"
        grp_a = std_a[(std_a["layer"] >= lo) & (std_a["layer"] <= hi)]
        ssr_a = _med(grp_a["ssr_QK"])

        row = {"Layer Group": label, f"{_short(name_a)} SSR": _fmt(ssr_a, 6)}

        if std_b is not None and name_b:
            grp_b = std_b[(std_b["layer"] >= lo) & (std_b["layer"] <= hi)]
            ssr_b = _med(grp_b["ssr_QK"])
            row[f"{_short(name_b)} SSR"] = _fmt(ssr_b, 6)
            if ssr_a and ssr_b and ssr_a > 0:
                improvement = (ssr_a - ssr_b) / ssr_a * 100
                row["Improvement (%)"] = f"+{improvement:.2f}%" if improvement >= 0 else f"{improvement:.2f}%"
            else:
                row["Improvement (%)"] = "—"

        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Table 3 — Output subspace cosU (Law 4)
# ─────────────────────────────────────────────────────────────────────────────

def make_table3(
    model_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    One row per model.
    Columns: Model | d_h | Random Baseline | cosU(QK) | cosU(QV) | cosU(KV)
    Uses standard layers only.
    """
    rows = []
    for model_id, df in model_dfs.items():
        std = _standard_only(df)
        if std.empty:
            continue
        baseline = _random_baseline_U(std)
        head_dim = int(std["head_dim"].dropna().median()) if "head_dim" in std.columns and std["head_dim"].notna().any() else "—"
        rows.append({
            "Model":           _short(model_id),
            "d_h":             head_dim,
            "Random 1/√d_h":  _fmt(baseline, 4),
            "cosU(Q,K)":      _fmt(_med(std["cosU_QK"]), 4),
            "cosU(Q,V)":      _fmt(_med(std["cosU_QV"]), 4),
            "cosU(K,V)":      _fmt(_med(std["cosU_KV"]), 4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Table 4 — Input subspace cosV (Law 5)
# ─────────────────────────────────────────────────────────────────────────────

def make_table4(
    model_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    One row per model.
    Columns: Model | d_model | Random Baseline | cosV(QK) | cosV(QV) | cosV(KV)
    Uses standard layers only.
    """
    rows = []
    for model_id, df in model_dfs.items():
        std = _standard_only(df)
        if std.empty:
            continue
        baseline = _random_baseline_V(std)
        d_model  = int(std["d_model"].dropna().median()) if "d_model" in std.columns and std["d_model"].notna().any() else "—"
        rows.append({
            "Model":           _short(model_id),
            "d_model":         d_model,
            "Random 1/√D":    _fmt(baseline, 4),
            "cosV(Q,K)":      _fmt(_med(std["cosV_QK"]), 4),
            "cosV(Q,V)":      _fmt(_med(std["cosV_QV"]), 4),
            "cosV(K,V)":      _fmt(_med(std["cosV_KV"]), 4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Table 5 — Condition number κ summary (Law 3)
# ─────────────────────────────────────────────────────────────────────────────

def make_table5(
    model_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    One row per model.
    Columns: Model | Median κ(Q) | Mean κ(Q) | Median κ(K) | Mean κ(K)
    Layer 0 typically has extreme κ — report separately.
    Uses standard layers only.
    """
    rows = []
    for model_id, df in model_dfs.items():
        std = _standard_only(df)
        if std.empty:
            continue
        # Layer 0 stats (typically extreme)
        l0 = std[std["layer"] == std["layer"].min()]
        deep = std[std["layer"] > std["layer"].min()]
        rows.append({
            "Model":            _short(model_id),
            "Median κ(Q) all":  _fmt(_med(std["cond_Q"]), 1),
            "Median κ(K) all":  _fmt(_med(std["cond_K"]), 1),
            "κ(Q) Layer 0":     _fmt(_med(l0["cond_Q"]), 1),
            "κ(K) Layer 0":     _fmt(_med(l0["cond_K"]), 1),
            "Median κ(Q) deep": _fmt(_med(deep["cond_Q"]), 1),
            "Median κ(K) deep": _fmt(_med(deep["cond_K"]), 1),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Table 6 — Wang Score leaderboard
# ─────────────────────────────────────────────────────────────────────────────

def make_table6(
    model_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Ranked by Wang Score descending.
    Columns: Rank | Model | Std Layers | Median Pearson | Median SSR | Wang Score
    """
    rows = []
    for model_id, df in model_dfs.items():
        std = _standard_only(df)
        if std.empty:
            continue
        med_ssr     = _med(std["ssr_QK"])
        wang_score  = 1 - med_ssr if med_ssr is not None else None
        med_pearson = _med(std["pearson_QK"])
        rows.append({
            "Model":          _short(model_id),
            "Std Layers":     std["layer"].nunique(),
            "Median Pearson": _fmt(med_pearson, 4),
            "Median SSR":     _fmt(med_ssr, 6),
            "Wang Score":     wang_score if wang_score is not None else float("nan"),
        })

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out

    df_out = df_out.sort_values("Wang Score", ascending=False).reset_index(drop=True)
    df_out.insert(0, "Rank", range(1, len(df_out) + 1))
    df_out["Wang Score"] = df_out["Wang Score"].apply(lambda x: _fmt(x, 6))
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Master: generate all tables at once
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_tables(
    model_dfs:    dict[str, pd.DataFrame],
    group_bounds: list[tuple[int, int]],
    name_a:       Optional[str] = None,
    name_b:       Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Generate all 6 tables.
    model_dfs: {model_id: per-head DataFrame from DB}
    group_bounds: layer groups for Table 2, e.g. [(0,11),(12,23),(24,35),(36,47)]
    name_a / name_b: model IDs for Table 2 comparison (name_a must be in model_dfs)
    """
    df_a = model_dfs.get(name_a) if name_a else None
    df_b = model_dfs.get(name_b) if name_b else None

    tables = {}
    tables["t1"] = make_table1(model_dfs)
    if df_a is not None:
        tables["t2"] = make_table2(df_a, name_a, df_b, name_b, group_bounds)
    else:
        tables["t2"] = pd.DataFrame({"Note": ["Select at least Model A for Table 2"]})
    tables["t3"] = make_table3(model_dfs)
    tables["t4"] = make_table4(model_dfs)
    tables["t5"] = make_table5(model_dfs)
    tables["t6"] = make_table6(model_dfs)
    return tables


# ─────────────────────────────────────────────────────────────────────────────
# Format all outputs
# ─────────────────────────────────────────────────────────────────────────────

TABLE_META = {
    "t1": ("Table 1 — Cross-Model Summary (Law 1 & 2)",
           "tab:law12_summary"),
    "t2": ("Table 2 — SSR Layer-Group Trend (Law 2)",
           "tab:ssr_layergroup"),
    "t3": ("Table 3 — Output Subspace Alignment cosU (Law 4)",
           "tab:law4_cosU"),
    "t4": ("Table 4 — Input Subspace Alignment cosV (Law 5)",
           "tab:law5_cosV"),
    "t5": ("Table 5 — Condition Number κ Summary (Law 3)",
           "tab:law3_cond"),
    "t6": ("Table 6 — Wang Score Leaderboard",
           "tab:wang_score"),
}


def format_all_latex(tables: dict[str, pd.DataFrame]) -> str:
    parts = []
    for key, df in tables.items():
        caption, label = TABLE_META[key]
        parts.append(df_to_latex(df, caption, label))
    return "\n\n".join(parts)


def format_all_markdown(tables: dict[str, pd.DataFrame]) -> str:
    parts = []
    for key, df in tables.items():
        caption, _ = TABLE_META[key]
        parts.append(df_to_markdown(df, caption))
    return "\n\n---\n\n".join(parts)