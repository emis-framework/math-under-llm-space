# db/writer.py
"""
数据库写入模块
- 写入分析结果到 layer_head_metrics
- 计算并写入 model_summary（pseudo-bulk 两步聚合，避免 GQA 伪重复）
- 支持断点续传（以 prefix+layer 为粒度）
- 写入权限验证
- 级联删除模型
"""

import os
import sqlite3
import numpy as np
from collections import defaultdict
from datetime import datetime
from db.schema import get_connection, init_db


# ─────────────────────────────────────────────
# 推断函数：layer_type 和 modality
# ─────────────────────────────────────────────

def infer_layer_type(kv_shared: bool) -> str:
    return "global" if kv_shared else "standard"


def infer_modality(prefix: str) -> str:
    p = prefix.lower()
    if "vision" in p or "visual" in p or "image" in p:
        return "vision"
    if "audio" in p or "speech" in p or "acoustic" in p:
        return "audio"
    return "language"


# ─────────────────────────────────────────────
# 写入权限验证
# ─────────────────────────────────────────────

def check_write_permission(admin_token: str) -> bool:
    server_token = os.environ.get("WRITE_TOKEN", "")
    if not server_token:
        return False
    return admin_token.strip() == server_token


# ─────────────────────────────────────────────
# 断点续传
# ─────────────────────────────────────────────

def get_analyzed_layers(
    conn:     sqlite3.Connection,
    model_id: str,
    prefix:   str,
) -> set:
    cur = conn.cursor()
    cur.execute(
        """SELECT DISTINCT layer FROM layer_head_metrics
           WHERE model_id = ? AND prefix = ?""",
        (model_id, prefix)
    )
    return {row[0] for row in cur.fetchall()}


def is_layer_complete(
    conn:             sqlite3.Connection,
    model_id:         str,
    prefix:           str,
    layer:            int,
    expected_records: int,
) -> bool:
    cur = conn.cursor()
    cur.execute(
        """SELECT COUNT(*) FROM layer_head_metrics
           WHERE model_id = ? AND prefix = ? AND layer = ?""",
        (model_id, prefix, layer)
    )
    return cur.fetchone()[0] >= expected_records


# ─────────────────────────────────────────────
# 写入模型元数据
# ─────────────────────────────────────────────

def upsert_model(
    conn:       sqlite3.Connection,
    model_id:   str,
    model_type: str = None,
    notes:      str = None,
):
    conn.execute(
        """INSERT INTO models(model_id, model_type, analyzed_at, notes)
           VALUES(?, ?, ?, ?)
           ON CONFLICT(model_id) DO UPDATE SET
               model_type  = excluded.model_type,
               analyzed_at = excluded.analyzed_at,
               notes       = excluded.notes""",
        (model_id, model_type, datetime.utcnow().isoformat(), notes)
    )
    conn.commit()


def upsert_component(
    conn:          sqlite3.Connection,
    model_id:      str,
    prefix:        str,
    n_layers:      int,
    head_dim_min:  int,
    head_dim_max:  int,
    has_kv_shared: bool,
    has_global:    bool,
    d_model:       int,
):
    modality = infer_modality(prefix)
    conn.execute(
        """INSERT INTO components(
               model_id, prefix, modality, n_layers,
               head_dim_min, head_dim_max,
               has_kv_shared, has_global, d_model
           )
           VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(model_id, prefix) DO UPDATE SET
               modality      = excluded.modality,
               n_layers      = excluded.n_layers,
               head_dim_min  = excluded.head_dim_min,
               head_dim_max  = excluded.head_dim_max,
               has_kv_shared = excluded.has_kv_shared,
               has_global    = excluded.has_global,
               d_model       = excluded.d_model""",
        (
            model_id, prefix, modality, n_layers,
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
    """批量写入一层的逐头指标，幂等"""
    if not records:
        return

    rows = []
    for r in records:
        layer_type = infer_layer_type(bool(r.get("kv_shared", False)))
        modality   = infer_modality(r["prefix"])
        rows.append((
            model_id,
            r["prefix"],
            r["layer"],
            layer_type,
            modality,
            r["kv_head"],
            r["q_head"],
            int(r.get("kv_shared", False)),
            r.get("head_dim"),
            r.get("d_model"),
            r.get("n_q_heads"),
            r.get("n_kv_heads"),
            r.get("pearson_QK"),  r.get("spearman_QK"),
            r.get("pearson_QV"),  r.get("pearson_KV"),
            r.get("ssr_QK"),      r.get("ssr_QV"),      r.get("ssr_KV"),
            r.get("sigma_max_Q"), r.get("sigma_min_Q"), r.get("cond_Q"),
            r.get("sigma_max_K"), r.get("sigma_min_K"), r.get("cond_K"),
            r.get("sigma_max_V"), r.get("sigma_min_V"), r.get("cond_V"),
            r.get("cosU_QK"),     r.get("cosU_QV"),     r.get("cosU_KV"),
            r.get("cosV_QK"),     r.get("cosV_QV"),     r.get("cosV_KV"),
            r.get("alpha_QK"),    r.get("alpha_res_QK"),
            r.get("alpha_QV"),    r.get("alpha_res_QV"),
            r.get("alpha_KV"),    r.get("alpha_res_KV"),
        ))

    conn.executemany(
        """INSERT OR REPLACE INTO layer_head_metrics(
               model_id, prefix, layer, layer_type, modality,
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
               ?,?,?,?,?,?,?,?,?,?,?,?,
               ?,?,?,?,?,?,?,
               ?,?,?,?,?,?,?,?,?,
               ?,?,?,?,?,?,
               ?,?,?,?,?,?
           )""",
        rows
    )
    conn.commit()


# ─────────────────────────────────────────────
# Pseudo-bulk 聚合核心函数
# ─────────────────────────────────────────────

def _pseudobulk(rows, col_name: str) -> np.ndarray:
    """
    Pseudo-bulk two-step aggregation (Nature Comms 2021).
    Avoids GQA pseudoreplication (e.g. 4Q:1K → 4 correlated records per KV head).

    Step 1: median within each (layer, kv_head) group
            → one value per KV-head per layer
    Step 2: return flat array of Step-1 values
            → caller computes final median / mean / quantile

    Works with both sqlite3.Row objects and plain dicts.
    """
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        try:
            v       = r["ssr_QK"] if col_name == "ssr_QK" else r[col_name]
            layer   = int(r["layer"])
            kv_head = int(r["kv_head"]) if r["kv_head"] is not None else 0
        except (KeyError, TypeError, IndexError):
            continue
        if v is None:
            continue
        groups[(layer, kv_head)].append(float(v))

    if not groups:
        return np.array([])

    # Step 1: median within each (layer, kv_head) group
    return np.array([float(np.median(vals)) for vals in groups.values()])


def _pseudobulk_col(rows, col_name: str) -> np.ndarray:
    """Generic version of _pseudobulk for any column name."""
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        try:
            v       = r[col_name]
            layer   = int(r["layer"])
            kv_head = int(r["kv_head"]) if r["kv_head"] is not None else 0
        except (KeyError, TypeError, IndexError):
            continue
        if v is None:
            continue
        groups[(layer, kv_head)].append(float(v))

    if not groups:
        return np.array([])

    return np.array([float(np.median(vals)) for vals in groups.values()])


# ─────────────────────────────────────────────
# 计算并写入 model_summary
# ─────────────────────────────────────────────

def _calc_summary_row(
    rows,
    model_id:   str,
    prefix:     str,
    layer_type: str,
) -> dict | None:
    if not rows:
        return None

    def pb(col):
        return _pseudobulk_col(rows, col)

    def med(arr): return float(np.median(arr)) if len(arr) > 0 else None
    def avg(arr): return float(np.mean(arr))   if len(arr) > 0 else None

    ssr_qk     = pb("ssr_QK")
    wang_score = float(1 - np.median(ssr_qk)) if len(ssr_qk) > 0 else None
    n_layers   = len(set(r["layer"] for r in rows))
    n_records  = len(rows)

    return {
        "model_id":          model_id,
        "prefix":            prefix,
        "layer_type":        layer_type,
        "median_pearson_QK": med(pb("pearson_QK")),
        "mean_pearson_QK":   avg(pb("pearson_QK")),
        "median_ssr_QK":     med(ssr_qk),
        "mean_ssr_QK":       avg(ssr_qk),
        "median_ssr_QV":     med(pb("ssr_QV")),
        "mean_ssr_QV":       avg(pb("ssr_QV")),
        "median_cond_Q":     med(pb("cond_Q")),
        "mean_cond_Q":       avg(pb("cond_Q")),
        "median_cosU_QK":    med(pb("cosU_QK")),
        "median_cosU_QV":    med(pb("cosU_QV")),
        "median_cosV_QK":    med(pb("cosV_QK")),
        "median_cosV_QV":    med(pb("cosV_QV")),
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
    重新计算并写入 model_summary（all / standard / global 三行）。
    wang_score 统一用 standard 层 pseudo-bulk median(SSR_QK) 计算。
    """
    cur = conn.cursor()
    cur.row_factory = sqlite3.Row

    # ── Wang Score: standard 层 pseudo-bulk ──────────────────────────────
    cur.execute(
        """SELECT layer, kv_head, ssr_QK FROM layer_head_metrics
           WHERE model_id = ? AND prefix = ? AND layer_type = 'standard'
             AND kv_shared = 0""",
        (model_id, prefix)
    )
    std_ssr       = _pseudobulk_col(cur.fetchall(), "ssr_QK")
    std_wang_score = float(1 - np.median(std_ssr)) if len(std_ssr) > 0 else None

    for layer_type in ["all", "standard", "global"]:
        if layer_type == "all":
            cur.execute(
                "SELECT * FROM layer_head_metrics WHERE model_id=? AND prefix=?",
                (model_id, prefix)
            )
        else:
            cur.execute(
                """SELECT * FROM layer_head_metrics
                   WHERE model_id=? AND prefix=? AND layer_type=?""",
                (model_id, prefix, layer_type)
            )

        rows    = cur.fetchall()
        summary = _calc_summary_row(rows, model_id, prefix, layer_type)
        if summary is None:
            continue

        summary["wang_score"] = std_wang_score  # always from standard pseudo-bulk

        conn.execute(
            """INSERT OR REPLACE INTO model_summary(
                   model_id, prefix, layer_type,
                   median_pearson_QK, mean_pearson_QK,
                   median_ssr_QK, mean_ssr_QK,
                   median_ssr_QV, mean_ssr_QV,
                   median_cond_Q, mean_cond_Q,
                   median_cosU_QK, median_cosU_QV,
                   median_cosV_QK, median_cosV_QV,
                   wang_score, n_layers, n_records, updated_at
               ) VALUES (
                   :model_id, :prefix, :layer_type,
                   :median_pearson_QK, :mean_pearson_QK,
                   :median_ssr_QK, :mean_ssr_QK,
                   :median_ssr_QV, :mean_ssr_QV,
                   :median_cond_Q, :mean_cond_Q,
                   :median_cosU_QK, :median_cosU_QV,
                   :median_cosV_QK, :median_cosV_QV,
                   :wang_score, :n_layers, :n_records, :updated_at
               )""",
            summary
        )

    conn.commit()


# ─────────────────────────────────────────────
# 批量刷新所有模型的 model_summary
# ─────────────────────────────────────────────

def refresh_all_summaries(conn: sqlite3.Connection) -> int:
    """
    Re-compute model_summary for every (model_id, prefix) in the DB.
    Called by Tab 3 Refresh button to migrate historical data to pseudo-bulk.
    Returns number of (model_id, prefix) pairs refreshed.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT model_id, prefix FROM layer_head_metrics"
    )
    pairs = cur.fetchall()
    for model_id, prefix in pairs:
        update_model_summary(conn, model_id, prefix)
    return len(pairs)


# ─────────────────────────────────────────────
# 删除模型（级联清除所有相关数据）
# ─────────────────────────────────────────────

def delete_model(
    conn:        sqlite3.Connection,
    model_id:    str,
    admin_token: str,
) -> tuple[bool, str]:
    """
    级联删除一个模型的所有数据。
    删除顺序：layer_head_metrics → model_summary → components → models
    需要 WRITE_TOKEN 验证。
    返回 (success, message)
    """
    if not check_write_permission(admin_token):
        return False, "❌ Permission denied: invalid or missing Admin Write Token."

    model_id = model_id.strip()
    if not model_id:
        return False, "❌ Model ID is empty."

    cur = conn.cursor()

    # 先确认模型是否存在
    cur.execute("SELECT model_id FROM models WHERE model_id = ?", (model_id,))
    if cur.fetchone() is None:
        return False, f"❌ Model not found in DB: '{model_id}'"

    # 统计各表将被删除的行数（用于日志）
    stats = {}
    for table in ["layer_head_metrics", "model_summary", "components"]:
        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE model_id = ?", (model_id,))
        stats[table] = cur.fetchone()[0]

    # 级联删除（子表先删，最后删 models）
    conn.execute("DELETE FROM layer_head_metrics WHERE model_id = ?", (model_id,))
    conn.execute("DELETE FROM model_summary      WHERE model_id = ?", (model_id,))
    conn.execute("DELETE FROM components         WHERE model_id = ?", (model_id,))
    conn.execute("DELETE FROM models             WHERE model_id = ?", (model_id,))
    conn.commit()

    msg = (
        f"✅ Deleted: '{model_id}'\n"
        f"   layer_head_metrics : {stats['layer_head_metrics']} rows\n"
        f"   model_summary      : {stats['model_summary']} rows\n"
        f"   components         : {stats['components']} rows\n"
        f"   models             : 1 row"
    )
    return True, msg