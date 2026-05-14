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

# ── 启动时初始化数据库 ────────────────────────
init_db()

# ─────────────────────────────────────────────
# 主界面
# ─────────────────────────────────────────────

with gr.Blocks(
    title="Wang's Five Laws — LLM Spectral Analyzer",
) as demo:

    gr.Markdown("""
    # 🔬 Wang's Five Laws — LLM Spectral Analyzer
    **Mathematical Foundations of Large Language Models (MF-LLM)**

    通过 **HTTP Range Request** 直接读取 HF 权重，**无需下载整个模型**。  
    自动识别模型结构（GQA / MHA / K=V共享 / 异构head_dim），  
    逐头计算王氏五定律全部指标，结果持久化到 SQLite。

    | 定律 | 指标 | 理论极值 |
    |------|------|---------|
    | 第一定律 | Pearson r（Q-K谱线性对齐） | → 1 |
    | 第二定律 | SSR（谱形状残差） | → 0 |
    | 第三定律 | 条件数 κ | 越小越稳定 |
    | 第四定律 | cosU(Uq,Uv)（超正交） | < 1/√d_head |
    | 第五定律 | cosV（输入子空间随机正交） | ≈ 1/√d_model |

    [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19707844-blue)](https://doi.org/10.5281/zenodo.19707844)
    [![HAL](https://img.shields.io/badge/HAL-hal--05609398-red)](https://hal.science/hal-05609398)
    [![Wang's Law](https://img.shields.io/badge/Wang%27s%20Law-r%3D1-blue)](https://github.com/emis-framework/math-under-llm)
    """)

    with gr.Tabs():
        # Tab1：结构探测
        inspect_model_id, inspect_token = build_tab_inspect()

        # Tab2：分析（含数据库写入 + 断点续传）
        analyze_model_id, analyze_token = build_tab_analyze()

        # Tab3：王氏评分排行榜
        build_tab_leaderboard()

        # Tab4：数据库浏览
        build_tab_database()

        # Tab5：作图（论文级别）
        build_tab_plot()

    # ── Tab1 → Tab2 同步模型 ID 和 token ─────────
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