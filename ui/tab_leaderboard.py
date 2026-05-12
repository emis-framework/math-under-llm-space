# ui/tab_leaderboard.py
"""
Tab3：王氏评分排行榜
- 从 model_summary 读取，按 wang_score 降序
- 支持按组件过滤（language_model / vision_tower / all）
- 支持按 layer_type 过滤（standard / global / all）
"""

import gradio as gr
import pandas as pd
import numpy as np

from db.schema import init_db
from db.reader import get_leaderboard


# ─────────────────────────────────────────────
# 排行榜列格式化
# ─────────────────────────────────────────────

def _format_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """格式化排行榜显示列"""
    if df.empty:
        return df

    # 提取可读的模型名（去掉 org 前缀）
    df = df.copy()
    df["model_name"] = df["model_id"].apply(
        lambda x: x.split("/")[-1] if "/" in x else x
    )

    # 王氏评分百分制（便于直觉理解）
    df["wang_score_pct"] = df["wang_score"].apply(
        lambda x: f"{x*100:.3f}" if pd.notna(x) else "N/A"
    )

    # 格式化关键指标
    for col in ["median_pearson_QK", "median_ssr_QK", "mean_ssr_QK"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: f"{x:.6f}" if pd.notna(x) else "N/A"
            )

    # 选择展示列
    display_cols = [
        "model_name",
        "prefix",
        "layer_type",
        "wang_score_pct",
        "median_pearson_QK",
        "median_ssr_QK",
        "mean_ssr_QK",
        "median_cosU_QK",
        "median_cosU_QV",
        "median_cosV_QK",
        "n_layers",
        "n_records",
        "model_id",       # 完整 ID 放最后
    ]
    existing = [c for c in display_cols if c in df.columns]
    return df[existing]


def load_leaderboard(
    prefix_filter: str,
    layer_type:    str,
) -> tuple[pd.DataFrame, str]:
    """
    加载排行榜数据
    返回 (DataFrame, 状态文本)
    """
    conn = init_db()

    # prefix_filter 空字符串 → None（不过滤）
    pfx = prefix_filter.strip() or None
    lt  = layer_type if layer_type != "all" else "standard"

    df = get_leaderboard(conn, prefix_filter=pfx, layer_type=lt, limit=100)

    if df.empty:
        return pd.DataFrame(), (
            "📭 排行榜暂无数据\n"
            "请先在「分析」Tab 分析至少一个模型的完整层。\n"
            f"（当前过滤：prefix='{pfx}', layer_type='{lt}'）"
        )

    formatted = _format_leaderboard(df)
    status = (
        f"✅ 共 {len(formatted)} 条记录  "
        f"| layer_type={lt}  "
        f"| prefix_filter='{pfx or '全部'}'"
    )
    return formatted, status


# ─────────────────────────────────────────────
# Tab3 UI
# ─────────────────────────────────────────────

def build_tab_leaderboard():
    with gr.Tab("🏆 排行榜"):
        gr.Markdown("""
        ## 王氏评分排行榜
        **Wang Score = 1 − median(SSR_QK)**，越高越好（理论极值 = 1）  
        基于 `standard` 层计算（排除 K=V 共享的全局层干扰）。
        """)

        with gr.Row():
            prefix_input = gr.Textbox(
                label="组件过滤（含关键词即匹配，留空=全部）",
                placeholder="language_model",
                value="",
                scale=3,
            )
            layer_type_input = gr.Dropdown(
                label="层类型",
                choices=["standard", "global", "all"],
                value="standard",
                scale=1,
            )
            refresh_btn = gr.Button("🔄 刷新排行榜", variant="primary", scale=1)

        status_text = gr.Textbox(
            label="状态",
            value="点击「刷新排行榜」加载数据",
            lines=1,
            interactive=False,
        )

        leaderboard_table = gr.Dataframe(
            label="王氏评分排行榜（按 Wang Score 降序）",
            headers=[
                "model_name", "prefix", "layer_type",
                "wang_score_pct",
                "median_pearson_QK", "median_ssr_QK", "mean_ssr_QK",
                "median_cosU_QK", "median_cosU_QV", "median_cosV_QK",
                "n_layers", "n_records", "model_id",
            ],
            interactive=False,
            wrap=True,
        )

        gr.Markdown("""
        ### 指标说明
        | 指标 | 含义 | 越好 |
        |------|------|------|
        | Wang Score | 1 − median(SSR_QK)，综合推理能力评分 | ↑ 高 |
        | median_pearson_QK | Q/K 奇异值谱 Pearson 相关中位数（第一定律） | ↑ 高 |
        | median_ssr_QK | Q/K 归一化谱失配中位数（第二定律） | ↓ 低 |
        | median_cosU_QK | Q/K 输出子空间对齐（第四定律，≈随机正交） | ≈ 1/√d |
        | median_cosU_QV | Q/V 输出子空间（第四定律，超正交） | ↓ 低 |
        | median_cosV_QK | Q/K 输入子空间（第五定律，≈随机正交） | ≈ 1/√D |
        """)

        # 事件绑定
        refresh_btn.click(
            fn=load_leaderboard,
            inputs=[prefix_input, layer_type_input],
            outputs=[leaderboard_table, status_text],
        )

        # 启动时自动加载
        leaderboard_table.change(fn=None)