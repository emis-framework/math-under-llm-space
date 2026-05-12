# db/writer.py
"""
数据库写入模块
- 写入分析结果到 layer_head_metrics
- 计算并写入 model_summary
- 支持断点续传（以 prefix+layer 为粒度）
"""

import sqlite3
import numpy as np
from datetime import datetime
from db.schema import get_connection, init_db


# ─────────────────────────────────────────────
# layer_type 推断
# ─────────────────────────────────────────────

def infer_layer_type(kv_shared: bool) -> str:
    """
    从 kv_shared 推断层类型
    kv_shared=True  → 'global'  （K=V共享，如 Gemma-4-31B 全局层）
    kv_shared=False → 'standard'
    零 hard coding，纯从结构特征推断
    """
    return "global" if kv_shared else "standard"


# ─────────────────────────────────────────────
# 断点续传：检查已完成的层
# ─────────────────────────────────────────────

def get_analyzed_layers(
    conn:     sqlite3.Connection,
    model_id: str,
    prefix:   str,
) -> set[int]:
    """
    返回已完成分析的层号集合
    用于断点续传：跳过已有数据的层
    粒度：(model_id, prefix, layer)
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT layer
        FROM layer_head_metrics
        WHERE model_id = ? AND prefix = ?
        """,
        (model_id, prefix)
    )
    return {row[0] for row in cur.fetchall()}


def is_layer_complete(
    conn:     sqlite3.Connection,
    model_id: str,
    prefix:   str,
    layer:    int,
    expected_records: int,
) -> bool:
    """
    检查某层是否已完整写入
    expected_records = n_q_heads（该层应有的记录数）
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*)
        FROM layer_head_metrics
        WHERE model_id = ? AND prefix = ? AND layer = ?
        """,
        (model_id, prefix, layer)
    )
    actual = cur.fetchone()[0]
    return actual >= expected_records


# ─────────────────────────────────────────────
# 写入模型元数据
# ─────────────────────────────────────────────

def upsert_model(
    conn:       sqlite3.Connection,
    model_id:   str,
    model_type: str = None,
    notes:      str = None,
):
    """写入或更新模型基本信息"""
    conn.execute(
        """
        INSERT INTO models(model_id, model_type, analyzed_at, notes)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(model_id) DO UPDATE SET
            model_type  = excluded.model_type,
            analyzed_at = excluded.analyzed_at,
            notes       = excluded.notes
        """,
        (model_id, model_type, datetime.utcnow().isoformat(), notes)
    )
    conn.commit()


def upsert_component(
    conn:         sqlite3.Connection,
    model_id:     str,
    prefix:       str,
    n_layers:     int,
    head_dim_min: int,
    head_dim_max: int,
    has_kv_shared:bool,
    has_global:   bool,
    d_model:      int,
):
    """写入或更新组件信息"""
    conn.execute(
        """
        INSERT INTO components(
            model_id, prefix, n_layers,
            head_dim_min, head_dim_max,
            has_kv_shared, has_global, d_model
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model_id, prefix) DO UPDATE SET
            n_layers      = excluded.n_layers,
            head_dim_min  = excluded.head_dim_min,
            head_dim_max  = excluded.head_dim_max,
            has_kv_shared = excluded.has_kv_shared,
            has_global    = excluded.has_global,
            d_model       = excluded.d_model
        """,
        (
            model_id, prefix, n_layers,
            head_dim_min, head_dim_max,
            int(has_kv_shared), int(has_global), d_model
        )
    )
    conn.commit()


# ─────────────────────────────────────────────
# 写入逐头指标
# ─────────────────────────────────────────────

def write_layer_records(
    conn:     sqlite3.Connection,
    model_id: str,
    records:  list[dict],
):
    """
    批量写入一层的逐头指标
    使用 INSERT OR REPLACE 实现幂等写入
    """
    if not records:
        return

    rows = []
    for r in records:
        layer_type = infer_layer_type(bool(r.get("kv_shared", False)))
        rows.append((
            model_id,
            r["prefix"],
            r["layer"],
            layer_type,
            r["kv_head"],
            r["q_head"],
            int(r.get("kv_shared", False)),
            r.get("head_dim"),
            r.get("d_model"),
            r.get("n_q_heads"),
            r.get("n_kv_heads"),
            # 第一定律
            r.get("pearson_QK"),
            r.get("spearman_QK"),
            r.get("pearson_QV"),
            r.get("pearson_KV"),
            # 第二定律
            r.get("ssr_QK"),
            r.get("ssr_QV"),
            r.get("ssr_KV"),
            # 第三定律
            r.get("sigma_max_Q"),
            r.get("sigma_min_Q"),
            r.get("cond_Q"),
            r.get("sigma_max_K"),
            r.get("sigma_min_K"),
            r.get("cond_K"),
            r.get("sigma_max_V"),
            r.get("sigma_min_V"),
            r.get("cond_V"),
            # 第四定律
            r.get("cosU_QK"),
            r.get("cosU_QV"),
            r.get("cosU_KV"),
            # 第五定律
            r.get("cosV_QK"),
            r.get("cosV_QV"),
            r.get("cosV_KV"),
            # 尺度因子
            r.get("alpha_QK"),
            r.get("alpha_res_QK"),
            r.get("alpha_QV"),
            r.get("alpha_res_QV"),
            r.get("alpha_KV"),
            r.get("alpha_res_KV"),
        ))

    conn.executemany(
        """
        INSERT OR REPLACE INTO layer_head_metrics(
            model_id, prefix, layer, layer_type,
            kv_head, q_head, kv_shared,
            head_dim, d_model, n_q_heads, n_kv_heads,
            pearson_QK, spearman_QK, pearson_QV, pearson_KV,
            ssr_QK, ssr_QV, ssr_KV,
            sigma_max_Q, sigma_min_Q, cond_Q,
            sigma_max_K, sigma_min_K, cond_K,
            sigma_max_V, sigma_min_V, cond_V,
            cosU_QK, cosU_QV, cosU_KV,
            cosV_QK, cosV_QV, cosV_KV,
            alpha_QK, alpha_res_QK,
            alpha_QV, alpha_res_QV,
            alpha_KV, alpha_res_KV
        ) VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,
            ?,?,?,?,?,?,?,
            ?,?,?,?,?,?,?,?,?,
            ?,?,?,?,?,?,
            ?,?,?,?,?,?
        )
        """,
        rows
    )
    conn.commit()


# ─────────────────────────────────────────────
# 计算并写入 model_summary
# ─────────────────────────────────────────────

def _calc_summary_row(
    rows: list[sqlite3.Row],
    model_id: str,
    prefix: str,
    layer_type: str,
) -> dict | None:
    """
    从一组 layer_head_metrics 行计算汇总统计
    返回 model_summary 的一行
    """
    if not rows:
        return None

    def col(name):
        vals = [r[name] for r in rows if r[name] is not None]
        return np.array(vals) if vals else np.array([])

    def med(arr):
        return float(np.median(arr)) if len(arr) > 0 else None

    def avg(arr):
        return float(np.mean(arr)) if len(arr) > 0 else None

    ssr_qk = col("ssr_QK")
    wang_score = float(1 - np.median(ssr_qk)) if len(ssr_qk) > 0 else None

    # 统计层数（去重）
    n_layers  = len(set(r["layer"] for r in rows))
    n_records = len(rows)

    return {
        "model_id":          model_id,
        "prefix":            prefix,
        "layer_type":        layer_type,
        "median_pearson_QK": med(col("pearson_QK")),
        "mean_pearson_QK":   avg(col("pearson_QK")),
        "median_ssr_QK":     med(ssr_qk),
        "mean_ssr_QK":       avg(ssr_qk),
        "median_ssr_QV":     med(col("ssr_QV")),
        "mean_ssr_QV":       avg(col("ssr_QV")),
        "median_cond_Q":     med(col("cond_Q")),
        "mean_cond_Q":       avg(col("cond_Q")),
        "median_cosU_QK":    med(col("cosU_QK")),
        "median_cosU_QV":    med(col("cosU_QV")),
        "median_cosV_QK":    med(col("cosV_QK")),
        "median_cosV_QV":    med(col("cosV_QV")),
        "wang_score":        wang_score,
        "n_layers":          n_layers,
        "n_records":         n_records,
        "updated_at":        datetime.utcnow().isoformat(),
    }


def update_model_summary(
    conn:     sqlite3.Connection,
    model_id: str,
    prefix:   str,
):
    """
    重新计算并写入 model_summary
    对每个 (model_id, prefix) 生成三行：
      - layer_type='all'
      - layer_type='standard'
      - layer_type='global'
    王氏评分固定用 standard 层计算
    """
    cur = conn.cursor()

    for layer_type in ["all", "standard", "global"]:
        # 查询对应数据
        if layer_type == "all":
            cur.execute(
                """
                SELECT * FROM layer_head_metrics
                WHERE model_id = ? AND prefix = ?
                """,
                (model_id, prefix)
            )
        else:
            cur.execute(
                """
                SELECT * FROM layer_head_metrics
                WHERE model_id = ? AND prefix = ? AND layer_type = ?
                """,
                (model_id, prefix, layer_type)
            )

        rows = cur.fetchall()
        summary = _calc_summary_row(rows, model_id, prefix, layer_type)

        if summary is None:
            continue

        # 王氏评分统一用 standard 层（如果当前是 all/global，重新取 standard 的 ssr）
        if layer_type != "standard":
            cur.execute(
                """
                SELECT ssr_QK FROM layer_head_metrics
                WHERE model_id = ? AND prefix = ? AND layer_type = 'standard'
                """,
                (model_id, prefix)
            )
            std_rows = cur.fetchall()
            if std_rows:
                std_ssr = np.array([r[0] for r in std_rows if r[0] is not None])
                summary["wang_score"] = float(1 - np.median(std_ssr)) if len(std_ssr) > 0 else None

        conn.execute(
            """
            INSERT OR REPLACE INTO model_summary(
                model_id, prefix, layer_type,
                median_pearson_QK, mean_pearson_QK,
                median_ssr_QK, mean_ssr_QK,
                median_ssr_QV, mean_ssr_QV,
                median_cond_Q, mean_cond_Q,
                median_cosU_QK, median_cosU_QV,
                median_cosV_QK, median_cosV_QV,
                wang_score,
                n_layers, n_records, updated_at
            ) VALUES (
                :model_id, :prefix, :layer_type,
                :median_pearson_QK, :mean_pearson_QK,
                :median_ssr_QK, :mean_ssr_QK,
                :median_ssr_QV, :mean_ssr_QV,
                :median_cond_Q, :mean_cond_Q,
                :median_cosU_QK, :median_cosU_QV,
                :median_cosV_QK, :median_cosV_QV,
                :wang_score,
                :n_layers, :n_records, :updated_at
            )
            """,
            summary
        )

    conn.commit()