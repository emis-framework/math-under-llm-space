# db/schema.py
"""
数据库表结构定义与初始化
SQLite 存储在 /data/wang_laws.db（HF Space bucket 持久化）
"""

import sqlite3
import os
from datetime import datetime


def get_db_path() -> str:
    if os.path.exists("/data"):
        return "/data/wang_laws.db"
    return "wang_laws.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ─────────────────────────────────────────────
# 建表 SQL
# ─────────────────────────────────────────────

SQL_CREATE_MODELS = """
CREATE TABLE IF NOT EXISTS models (
    model_id      TEXT PRIMARY KEY,
    model_type    TEXT,
    analyzed_at   TIMESTAMP,
    analyze_sec   REAL,
    notes         TEXT
);
"""

SQL_CREATE_COMPONENTS = """
CREATE TABLE IF NOT EXISTS components (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id      TEXT NOT NULL,
    prefix        TEXT NOT NULL,
    modality      TEXT DEFAULT 'language',  -- language/vision/audio
    n_layers      INTEGER,
    head_dim_min  INTEGER,
    head_dim_max  INTEGER,
    has_kv_shared INTEGER DEFAULT 0,
    has_global    INTEGER DEFAULT 0,
    d_model       INTEGER,
    UNIQUE(model_id, prefix),
    FOREIGN KEY(model_id) REFERENCES models(model_id)
);
"""

SQL_CREATE_LAYER_HEAD_METRICS = """
CREATE TABLE IF NOT EXISTS layer_head_metrics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id      TEXT NOT NULL,
    prefix        TEXT NOT NULL,
    layer         INTEGER NOT NULL,
    layer_type    TEXT DEFAULT 'standard',  -- standard/global
    modality      TEXT DEFAULT 'language',  -- language/vision/audio
    kv_head       INTEGER NOT NULL,
    q_head        INTEGER NOT NULL,
    kv_shared     INTEGER DEFAULT 0,
    head_dim      INTEGER,
    d_model       INTEGER,
    n_q_heads     INTEGER,
    n_kv_heads    INTEGER,
    -- 第一定律
    pearson_QK    REAL, spearman_QK  REAL,
    pearson_QV    REAL, pearson_KV   REAL,
    -- 第二定律
    ssr_QK        REAL, ssr_QV       REAL, ssr_KV      REAL,
    -- 第三定律
    sigma_max_Q   REAL, sigma_min_Q  REAL, cond_Q      REAL,
    sigma_max_K   REAL, sigma_min_K  REAL, cond_K      REAL,
    sigma_max_V   REAL, sigma_min_V  REAL, cond_V      REAL,
    -- 第四定律
    cosU_QK       REAL, cosU_QV      REAL, cosU_KV     REAL,
    -- 第五定律
    cosV_QK       REAL, cosV_QV      REAL, cosV_KV     REAL,
    -- 尺度因子
    alpha_QK      REAL, alpha_res_QK REAL,
    alpha_QV      REAL, alpha_res_QV REAL,
    alpha_KV      REAL, alpha_res_KV REAL,

    UNIQUE(model_id, prefix, layer, kv_head, q_head),
    FOREIGN KEY(model_id) REFERENCES models(model_id)
);
"""

SQL_CREATE_MODEL_SUMMARY = """
CREATE TABLE IF NOT EXISTS model_summary (
    model_id          TEXT NOT NULL,
    prefix            TEXT NOT NULL,
    layer_type        TEXT NOT NULL DEFAULT 'all',
    -- 第一定律
    median_pearson_QK REAL, mean_pearson_QK REAL,
    -- 第二定律
    median_ssr_QK     REAL, mean_ssr_QK     REAL,
    median_ssr_QV     REAL, mean_ssr_QV     REAL,
    -- 第三定律
    median_cond_Q     REAL, mean_cond_Q     REAL,
    -- 第四定律
    median_cosU_QK    REAL, median_cosU_QV  REAL,
    -- 第五定律
    median_cosV_QK    REAL, median_cosV_QV  REAL,
    -- 王氏评分
    wang_score        REAL,
    -- 统计范围
    n_layers          INTEGER,
    n_records         INTEGER,
    updated_at        TIMESTAMP,

    PRIMARY KEY(model_id, prefix, layer_type),
    FOREIGN KEY(model_id) REFERENCES models(model_id)
);
"""

SQL_CREATE_INDEXES = [
    """CREATE INDEX IF NOT EXISTS idx_metrics_model_prefix
       ON layer_head_metrics(model_id, prefix)""",
    """CREATE INDEX IF NOT EXISTS idx_metrics_layer
       ON layer_head_metrics(model_id, prefix, layer)""",
    """CREATE INDEX IF NOT EXISTS idx_metrics_modality
       ON layer_head_metrics(model_id, modality)""",
    """CREATE INDEX IF NOT EXISTS idx_summary_wang_score
       ON model_summary(wang_score DESC)""",
    """CREATE INDEX IF NOT EXISTS idx_metrics_resume
       ON layer_head_metrics(model_id, prefix, layer, kv_head, q_head)""",
    """CREATE INDEX IF NOT EXISTS idx_components_modality
       ON components(model_id, modality)""",
]


# ─────────────────────────────────────────────
# 迁移：为旧数据库加 modality 列
# ─────────────────────────────────────────────

def _migrate_add_modality(conn: sqlite3.Connection):
    """
    幂等迁移：给旧表加 modality 列并回填数据。
    新建数据库时这些列已在建表SQL中，PRAGMA会检测到直接跳过。
    """
    cur = conn.cursor()

    # ── layer_head_metrics ────────────────────
    cur.execute("PRAGMA table_info(layer_head_metrics)")
    lhm_cols = [row[1] for row in cur.fetchall()]

    if "modality" not in lhm_cols:
        cur.execute(
            "ALTER TABLE layer_head_metrics "
            "ADD COLUMN modality TEXT DEFAULT 'language'"
        )
        # 回填 vision
        cur.execute(
            """UPDATE layer_head_metrics SET modality = 'vision'
               WHERE prefix LIKE '%vision%'
                  OR prefix LIKE '%visual%'
                  OR prefix LIKE '%image%'"""
        )
        # 回填 audio
        cur.execute(
            """UPDATE layer_head_metrics SET modality = 'audio'
               WHERE prefix LIKE '%audio%'
                  OR prefix LIKE '%speech%'
                  OR prefix LIKE '%acoustic%'"""
        )
        # language 已由 DEFAULT 'language' 覆盖，无需额外更新

    # ── components ────────────────────────────
    cur.execute("PRAGMA table_info(components)")
    comp_cols = [row[1] for row in cur.fetchall()]

    if "modality" not in comp_cols:
        cur.execute(
            "ALTER TABLE components "
            "ADD COLUMN modality TEXT DEFAULT 'language'"
        )
        cur.execute(
            """UPDATE components SET modality = 'vision'
               WHERE prefix LIKE '%vision%'
                  OR prefix LIKE '%visual%'
                  OR prefix LIKE '%image%'"""
        )
        cur.execute(
            """UPDATE components SET modality = 'audio'
               WHERE prefix LIKE '%audio%'
                  OR prefix LIKE '%speech%'
                  OR prefix LIKE '%acoustic%'"""
        )

    conn.commit()


# ─────────────────────────────────────────────
# 初始化
# ─────────────────────────────────────────────

def init_db() -> sqlite3.Connection:
    conn = get_connection()
    cur  = conn.cursor()

    # 第一步：建表
    cur.execute(SQL_CREATE_MODELS)
    cur.execute(SQL_CREATE_COMPONENTS)
    cur.execute(SQL_CREATE_LAYER_HEAD_METRICS)
    cur.execute(SQL_CREATE_MODEL_SUMMARY)
    conn.commit()

    # 第二步：迁移旧数据（加 modality 列）← 必须在建索引之前
    _migrate_add_modality(conn)

    # 第三步：建索引（此时 modality 列已确保存在）
    cur = conn.cursor()
    for sql in SQL_CREATE_INDEXES:
        cur.execute(sql)
    conn.commit()

    return conn


def get_db_stats(conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()
    stats = {}
    for table in ["models", "components", "layer_head_metrics", "model_summary"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        stats[table] = cur.fetchone()[0]
    db_path = get_db_path()
    if os.path.exists(db_path):
        stats["db_size_mb"] = round(os.path.getsize(db_path) / 1024 / 1024, 2)
    else:
        stats["db_size_mb"] = 0
    return stats