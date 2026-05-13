# db/writer.py
"""
数据库写入模块
- 写入分析结果到 layer_head_metrics
- 计算并写入 model_summary
- 支持断点续传（以 prefix+layer 为粒度）
- 写入权限验证
"""

import os
import sqlite3
import numpy as np
from datetime import datetime
from db.schema import get_connection, init_db


# ─────────────────────────────────────────────
# 推断函数：layer_type 和 modality
# ─────────────────────────────────────────────

def infer_layer_type(kv_shared: bool) -> str:
    """
    从结构特征推断层类型
    kv_shared=True  → 'global'  （K=V共享，如 Gemma 全局层）
    kv_shared=False → 'standard'
    """
    return "global" if kv_shared else "standard"


def infer_modality(prefix: str) -> str:
    """
    从组件前缀推断模态
    纯关键词匹配，不 hard coding 模型名
    未匹配到任何关键词 → 默认 'language'
    （覆盖纯语言模型，如 "model." 前缀的 LLaMA/Qwen）
    """
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
    """
    验证管理员写入权限。
    WRITE_TOKEN 存储在 HF Space Secrets（加密，不进入 git repo）。
    运行时由 HF 注入为环境变量，只在服务端比对，不返回给前端。

    返回：
      True  = 有写入权限
      False = 只读模式（分析可以跑，结果不写库）
    """
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
    """返回已完成分析的层号集合"""
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
    """检查某层是否已完整写入"""
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
# 计算并写入 model_summary
# ─────────────────────────────────────────────

def _calc_summary_row(
    rows:       list,
    model_id:   str,
    prefix:     str,
    layer_type: str,
) -> dict | None:
    if not rows:
        return None

    def col(name):
        vals = [r[name] for r in rows if r[name] is not None]
        return np.array(vals, dtype=float) if vals else np.array([])

    def med(arr): return float(np.median(arr)) if len(arr) > 0 else None
    def avg(arr): return float(np.mean(arr))   if len(arr) > 0 else None

    ssr_qk     = col("ssr_QK")
    wang_score = float(1 - np.median(ssr_qk)) if len(ssr_qk) > 0 else None
    n_layers   = len(set(r["layer"] for r in rows))
    n_records  = len(rows)

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
    重新计算并写入 model_summary（all / standard / global 三行）
    wang_score 统一用 standard 层计算
    """
    cur = conn.cursor()

    # 预取 standard 层的 ssr_QK（wang_score 统一用这个）
    cur.execute(
        """SELECT ssr_QK FROM layer_head_metrics
           WHERE model_id = ? AND prefix = ? AND layer_type = 'standard'""",
        (model_id, prefix)
    )
    std_ssr_rows = cur.fetchall()
    std_ssr = np.array(
        [r[0] for r in std_ssr_rows if r[0] is not None], dtype=float
    )
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

        # wang_score 统一用 standard 层
        summary["wang_score"] = std_wang_score

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