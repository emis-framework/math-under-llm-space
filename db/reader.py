# db/reader.py
"""
数据库查询模块
- 排行榜查询
- 模型详情查询
- 断点续传状态查询
"""

import sqlite3
import pandas as pd
from db.schema import get_connection, init_db


# ─────────────────────────────────────────────
# 排行榜
# ─────────────────────────────────────────────

def get_leaderboard(
    conn:          sqlite3.Connection,
    prefix_filter: str  = None,   # 只看某个组件，None=全部
    layer_type:    str  = "standard",
    limit:         int  = 50,
) -> pd.DataFrame:
    """
    排行榜查询
    按 wang_score 降序排列
    """
    sql = """
        SELECT
            s.model_id,
            s.prefix,
            s.layer_type,
            s.wang_score,
            s.median_pearson_QK,
            s.median_ssr_QK,
            s.mean_ssr_QK,
            s.median_cosU_QK,
            s.median_cosU_QV,
            s.median_cosV_QK,
            s.median_cond_Q,
            s.n_layers,
            s.n_records,
            s.updated_at,
            -- 组件信息
            c.head_dim_min,
            c.head_dim_max,
            c.has_kv_shared,
            c.has_global,
            c.d_model
        FROM model_summary s
        LEFT JOIN components c
            ON s.model_id = c.model_id AND s.prefix = c.prefix
        WHERE s.layer_type = ?
    """
    params = [layer_type]

    if prefix_filter:
        sql += " AND s.prefix LIKE ?"
        params.append(f"%{prefix_filter}%")

    sql += " ORDER BY s.wang_score DESC LIMIT ?"
    params.append(limit)

    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()

    if not rows:
        return pd.DataFrame()

    cols = [d[0] for d in cur.description]
    return pd.DataFrame([dict(zip(cols, row)) for row in rows])


# ─────────────────────────────────────────────
# 模型详情
# ─────────────────────────────────────────────

def get_model_summary(
    conn:     sqlite3.Connection,
    model_id: str,
) -> pd.DataFrame:
    """获取某模型所有组件的汇总统计"""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM model_summary
        WHERE model_id = ?
        ORDER BY prefix, layer_type
        """,
        (model_id,)
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    cols = [d[0] for d in cur.description]
    return pd.DataFrame([dict(zip(cols, row)) for row in rows])


def get_layer_metrics(
    conn:       sqlite3.Connection,
    model_id:   str,
    prefix:     str = None,
    layer_type: str = None,
    start_layer:int = None,
    end_layer:  int = None,
) -> pd.DataFrame:
    """
    查询逐头原始数据
    支持按 prefix / layer_type / 层号范围过滤
    """
    sql    = "SELECT * FROM layer_head_metrics WHERE model_id = ?"
    params = [model_id]

    if prefix:
        sql += " AND prefix = ?"
        params.append(prefix)
    if layer_type:
        sql += " AND layer_type = ?"
        params.append(layer_type)
    if start_layer is not None:
        sql += " AND layer >= ?"
        params.append(start_layer)
    if end_layer is not None:
        sql += " AND layer <= ?"
        params.append(end_layer)

    sql += " ORDER BY prefix, layer, kv_head, q_head"

    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()

    if not rows:
        return pd.DataFrame()
    cols = [d[0] for d in cur.description]
    return pd.DataFrame([dict(zip(cols, row)) for row in rows])


def get_analyzed_models(conn: sqlite3.Connection) -> pd.DataFrame:
    """获取所有已分析模型列表"""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            m.model_id,
            m.model_type,
            m.analyzed_at,
            m.analyze_sec,
            COUNT(DISTINCT c.prefix) as n_components,
            SUM(c.n_layers) as total_layers
        FROM models m
        LEFT JOIN components c ON m.model_id = c.model_id
        GROUP BY m.model_id
        ORDER BY m.analyzed_at DESC
        """
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    cols = [d[0] for d in cur.description]
    return pd.DataFrame([dict(zip(cols, row)) for row in rows])


# ─────────────────────────────────────────────
# 断点续传状态
# ─────────────────────────────────────────────

def get_resume_status(
    conn:     sqlite3.Connection,
    model_id: str,
    prefix:   str,
) -> dict:
    """
    查询某 (model_id, prefix) 的断点续传状态
    返回已完成的层号集合和统计信息
    """
    cur = conn.cursor()

    # 已完成的层
    cur.execute(
        """
        SELECT DISTINCT layer, COUNT(*) as n_heads
        FROM layer_head_metrics
        WHERE model_id = ? AND prefix = ?
        GROUP BY layer
        ORDER BY layer
        """,
        (model_id, prefix)
    )
    rows = cur.fetchall()

    done_layers = {r[0]: r[1] for r in rows}

    return {
        "done_layers":  set(done_layers.keys()),
        "layer_detail": done_layers,   # layer → n_heads
        "total_done":   len(done_layers),
    }