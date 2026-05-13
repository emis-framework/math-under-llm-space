# db/reader.py
"""
数据库查询模块
- 排行榜查询
- 模型详情查询（方案A：按modality聚合 + 方案B：原始components行）
- 逐头原始数据查询
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
    modality:      str  = "language",   # language/vision/audio/all
    layer_type:    str  = "standard",
    limit:         int  = 100,
) -> pd.DataFrame:
    """
    排行榜查询，按 wang_score 降序。
    modality 过滤通过 components 表的 prefix 关联实现。
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
            c.modality,
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

    if modality != "all":
        sql += " AND c.modality = ?"
        params.append(modality)

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
# 模型列表（方案A：按modality聚合）
# ─────────────────────────────────────────────

def get_analyzed_models(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    模型列表，按 modality 聚合层数。
    language_layers = SUM(n_layers) WHERE modality='language'
    自动包含 standard + global 层（同一 prefix 下）。
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            m.model_id,
            m.model_type,
            m.analyzed_at,
            m.analyze_sec,
            COUNT(DISTINCT c.prefix)  AS n_components,
            SUM(CASE WHEN c.modality = 'language'
                THEN c.n_layers ELSE 0 END) AS language_layers,
            SUM(CASE WHEN c.modality = 'vision'
                THEN c.n_layers ELSE 0 END) AS vision_layers,
            SUM(CASE WHEN c.modality = 'audio'
                THEN c.n_layers ELSE 0 END) AS audio_layers
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
# 模型详情（方案B：原始components行）
# ─────────────────────────────────────────────

def get_model_components(
    conn:     sqlite3.Connection,
    model_id: str,
) -> pd.DataFrame:
    """
    返回某模型的原始 components 行（方案B详情展开用）。
    每行 = 一个 prefix，含 modality/n_layers/head_dim 等。
    """
    cur = conn.cursor()
    cur.execute(
        """SELECT
               prefix, modality, n_layers,
               head_dim_min, head_dim_max,
               has_kv_shared, has_global, d_model
           FROM components
           WHERE model_id = ?
           ORDER BY modality, prefix""",
        (model_id,)
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    cols = [d[0] for d in cur.description]
    return pd.DataFrame([dict(zip(cols, row)) for row in rows])


def get_model_summary(
    conn:     sqlite3.Connection,
    model_id: str,
) -> pd.DataFrame:
    """获取某模型所有组件的汇总统计"""
    cur = conn.cursor()
    cur.execute(
        """SELECT * FROM model_summary
           WHERE model_id = ?
           ORDER BY prefix, layer_type""",
        (model_id,)
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    cols = [d[0] for d in cur.description]
    return pd.DataFrame([dict(zip(cols, row)) for row in rows])


# ─────────────────────────────────────────────
# 逐头原始数据
# ─────────────────────────────────────────────

def get_layer_metrics(
    conn:        sqlite3.Connection,
    model_id:    str,
    prefix:      str  = None,
    modality:    str  = None,   # language/vision/audio
    layer_type:  str  = None,   # standard/global
    start_layer: int  = None,
    end_layer:   int  = None,
) -> pd.DataFrame:
    """
    逐头原始数据查询。
    modality 和 layer_type 是两个独立维度，可以组合过滤。
    """
    sql    = "SELECT * FROM layer_head_metrics WHERE model_id = ?"
    params = [model_id]

    if prefix:
        sql += " AND prefix = ?"
        params.append(prefix)
    if modality:
        sql += " AND modality = ?"
        params.append(modality)
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


# ─────────────────────────────────────────────
# 断点续传状态
# ─────────────────────────────────────────────

def get_resume_status(
    conn:     sqlite3.Connection,
    model_id: str,
    prefix:   str,
) -> dict:
    cur = conn.cursor()
    cur.execute(
        """SELECT DISTINCT layer, COUNT(*) as n_heads
           FROM layer_head_metrics
           WHERE model_id = ? AND prefix = ?
           GROUP BY layer ORDER BY layer""",
        (model_id, prefix)
    )
    rows = cur.fetchall()
    done_layers = {r[0]: r[1] for r in rows}
    return {
        "done_layers":  set(done_layers.keys()),
        "layer_detail": done_layers,
        "total_done":   len(done_layers),
    }