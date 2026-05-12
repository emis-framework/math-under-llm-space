# db/schema.py
"""
数据库表结构定义与初始化
SQLite 存储在 /data/wang_laws.db（HF Space bucket 持久化）
"""

import sqlite3
import os
from datetime import datetime

# ─────────────────────────────────────────────
# 数据库路径
# /data 是 HF Space bucket 挂载点，重启后数据不丢失
# 本地开发时自动回退到当前目录
# ─────────────────────────────────────────────

def get_db_path() -> str:
    if os.path.exists("/data"):
        return "/data/wang_laws.db"
    return "wang_laws.db"


def get_connection() -> sqlite3.Connection:
    """获取数据库连接，启用 WAL 模式提升并发性能"""
    conn = sqlite3.connect(get_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row      # 支持按列名访问
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ─────────────────────────────────────────────
# 建表 SQL
# ─────────────────────────────────────────────

SQL_CREATE_MODELS = """
CREATE TABLE IF NOT EXISTS models (
    model_id      TEXT PRIMARY KEY,
    model_type    TEXT,              -- gemma4 / llama / qwen2 等
    analyzed_at   TIMESTAMP,
    analyze_sec   REAL,              -- 分析耗时（秒）
    notes         TEXT               -- 备注
);
"""

SQL_CREATE_COMPONENTS = """
CREATE TABLE IF NOT EXISTS components (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id      TEXT NOT NULL,
    prefix        TEXT NOT NULL,     -- 如 model.language_model.
    n_layers      INTEGER,           -- 该组件完整层数
    head_dim_min  INTEGER,           -- 最小 head_dim（异构层用）
    head_dim_max  INTEGER,           -- 最大 head_dim
    has_kv_shared INTEGER DEFAULT 0, -- 是否有 K=V 共享层（全局层）
    has_global    INTEGER DEFAULT 0, -- 是否有 global 层
    d_model       INTEGER,           -- 输入维度
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
    layer_type    TEXT DEFAULT 'standard', -- standard / global
    kv_head       INTEGER NOT NULL,
    q_head        INTEGER NOT NULL,
    kv_shared     INTEGER DEFAULT 0,  -- 1=K=V共享（理论值），0=独立V
    head_dim      INTEGER,
    d_model       INTEGER,
    n_q_heads     INTEGER,
    n_kv_heads    INTEGER,
    -- 第一定律：谱线性对齐
    pearson_QK    REAL,
    spearman_QK   REAL,
    pearson_QV    REAL,
    pearson_KV    REAL,
    -- 第二定律：谱形状残差
    ssr_QK        REAL,
    ssr_QV        REAL,
    ssr_KV        REAL,
    -- 第三定律：条件数
    sigma_max_Q   REAL,
    sigma_min_Q   REAL,
    cond_Q        REAL,
    sigma_max_K   REAL,
    sigma_min_K   REAL,
    cond_K        REAL,
    sigma_max_V   REAL,
    sigma_min_V   REAL,
    cond_V        REAL,
    -- 第四定律：左奇异向量对齐（输出子空间）
    cosU_QK       REAL,
    cosU_QV       REAL,
    cosU_KV       REAL,
    -- 第五定律：右奇异向量对齐（输入子空间）
    cosV_QK       REAL,
    cosV_QV       REAL,
    cosV_KV       REAL,
    -- 尺度因子 + 最小二乘残差
    alpha_QK      REAL,
    alpha_res_QK  REAL,
    alpha_QV      REAL,
    alpha_res_QV  REAL,
    alpha_KV      REAL,
    alpha_res_KV  REAL,

    UNIQUE(model_id, prefix, layer, kv_head, q_head),
    FOREIGN KEY(model_id) REFERENCES models(model_id)
);
"""

SQL_CREATE_MODEL_SUMMARY = """
CREATE TABLE IF NOT EXISTS model_summary (
    model_id          TEXT NOT NULL,
    prefix            TEXT NOT NULL,
    layer_type        TEXT NOT NULL DEFAULT 'all', -- all / standard / global
    -- 第一定律
    median_pearson_QK REAL,
    mean_pearson_QK   REAL,
    -- 第二定律（王氏评分核心）
    median_ssr_QK     REAL,
    mean_ssr_QK       REAL,
    median_ssr_QV     REAL,
    mean_ssr_QV       REAL,
    -- 第三定律
    median_cond_Q     REAL,
    mean_cond_Q       REAL,
    -- 第四定律
    median_cosU_QK    REAL,
    median_cosU_QV    REAL,
    -- 第五定律
    median_cosV_QK    REAL,
    median_cosV_QV    REAL,
    -- 王氏评分（暂时 = 1 - median_ssr_QK，基于 standard 层）
    wang_score        REAL,
    -- 统计范围
    n_layers          INTEGER,  -- 参与统计的层数
    n_records         INTEGER,  -- 参与统计的记录数
    updated_at        TIMESTAMP,

    PRIMARY KEY(model_id, prefix, layer_type),
    FOREIGN KEY(model_id) REFERENCES models(model_id)
);
"""

# 索引：加速常用查询
SQL_CREATE_INDEXES = [
    # 按模型+组件查询层数据
    """CREATE INDEX IF NOT EXISTS idx_metrics_model_prefix
       ON layer_head_metrics(model_id, prefix)""",
    # 按层号范围查询
    """CREATE INDEX IF NOT EXISTS idx_metrics_layer
       ON layer_head_metrics(model_id, prefix, layer)""",
    # 排行榜查询
    """CREATE INDEX IF NOT EXISTS idx_summary_wang_score
       ON model_summary(wang_score DESC)""",
    # 断点续传：快速判断某层是否已分析
    """CREATE INDEX IF NOT EXISTS idx_metrics_resume
       ON layer_head_metrics(model_id, prefix, layer, kv_head, q_head)""",
]


# ─────────────────────────────────────────────
# 初始化函数
# ─────────────────────────────────────────────

def init_db() -> sqlite3.Connection:
    """
    初始化数据库：建表 + 建索引
    幂等操作，重复调用安全
    返回数据库连接
    """
    conn = get_connection()
    cur  = conn.cursor()

    cur.execute(SQL_CREATE_MODELS)
    cur.execute(SQL_CREATE_COMPONENTS)
    cur.execute(SQL_CREATE_LAYER_HEAD_METRICS)
    cur.execute(SQL_CREATE_MODEL_SUMMARY)

    for sql in SQL_CREATE_INDEXES:
        cur.execute(sql)

    conn.commit()
    return conn


def get_db_stats(conn: sqlite3.Connection) -> dict:
    """获取数据库统计信息"""
    cur = conn.cursor()
    stats = {}

    for table in ["models", "components", "layer_head_metrics", "model_summary"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        stats[table] = cur.fetchone()[0]

    # 数据库文件大小
    db_path = get_db_path()
    if os.path.exists(db_path):
        stats["db_size_mb"] = round(os.path.getsize(db_path) / 1024 / 1024, 2)
    else:
        stats["db_size_mb"] = 0

    return stats