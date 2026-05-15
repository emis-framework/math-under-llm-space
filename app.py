# app.py
"""
Wang's Five Laws — LLM Spectral Analyzer
主入口，组装所有 Tab
"""

import gradio as gr
from db.schema import init_db
from ui.tab_inspect import build_tab_inspect
from ui.tab_analyze import build_tab_analyze
from ui.tab_leaderboard import build_tab_leaderboard
from ui.tab_database import build_tab_database
from ui.tab_plot import build_tab_plot
from ui.tab_tables import build_tab_tables

# ── 启动时初始化数据库 ────────────────────────
init_db()

with gr.Blocks(
    title="Wang's Five Laws — LLM Spectral Analyzer",
) as demo:

    # ── 双语标题 ──────────────────────────────
    gr.Markdown("""
    # 🔬 Wang's Five Laws — LLM Spectral Analyzer
    ### 王氏五定律 — 大模型谱分析工具
    **Mathematical Foundations of Large Language Models (MF-LLM)**

    Reads HF weights via **HTTP Range Request** — no full model download required.
    Auto-detects model structure (GQA / MHA / K=V shared / heterogeneous head_dim),
    computes all Five Laws metrics per attention head, persists results to SQLite.

    通过 **HTTP Range Request** 直接读取 HF 权重，无需下载整个模型。
    自动识别模型结构，逐头计算王氏五定律全部指标，结果持久化到 SQLite。

    [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19707844-blue)](https://doi.org/10.5281/zenodo.19707844)
    [![HAL](https://img.shields.io/badge/HAL-hal--05609398-red)](https://hal.science/hal-05609398)
    [![Wang's Law](https://img.shields.io/badge/Wang%27s%20Law-r%3D1-blue)](https://github.com/emis-framework/math-under-llm)
    """)

    # ── 双语表格并排 ──────────────────────────
    with gr.Row():
        gr.Markdown("""
| Law | Metric | Ideal |
|-----|--------|-------|
| Law 1 | Pearson r (Q–K spectral alignment) | → 1 |
| Law 2 | SSR (spectral shape residual) | → 0 |
| Law 3 | Condition number κ | smaller = more stable |
| Law 4 | cosU(Uq, Uv)  super-orthogonal | < 1/√d_head |
| Law 5 | cosV  input subspace random orthogonal | ≈ 1/√d_model |
        """)
        gr.Markdown("""
| 定律 | 指标 | 理论极值 |
|------|------|---------|
| 第一定律 | Pearson r（Q-K 谱线性对齐） | → 1 |
| 第二定律 | SSR（谱形状残差） | → 0 |
| 第三定律 | 条件数 κ | 越小越稳定 |
| 第四定律 | cosU(Uq, Uv)（超正交） | < 1/√d_head |
| 第五定律 | cosV（输入子空间随机正交） | ≈ 1/√d_model |
        """)

    # ── Tabs ──────────────────────────────────
    with gr.Tabs():
        inspect_model_id, inspect_token = build_tab_inspect()
        analyze_model_id, analyze_token = build_tab_analyze()
        build_tab_leaderboard()
        build_tab_database()
        build_tab_plot()
        build_tab_tables()

    # ── Tab1 → Tab2 联动 ──────────────────────
    inspect_model_id.change(
        fn=lambda x: x,
        inputs=inspect_model_id,
        outputs=analyze_model_id,
    )
    inspect_token.change(
        fn=lambda x: x,
        inputs=inspect_token,
        outputs=analyze_token,
    )

if __name__ == "__main__":
    demo.launch()