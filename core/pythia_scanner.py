# core/pythia_scanner.py
"""
Pythia checkpoint SSR/UniIso 动态扫描
======================================
扫描 EleutherAI/pythia-{size} 的多个训练 checkpoint，
计算每层每头的 SSR、UniIso、eff_rank 等指标，
与 p4-lab3-llama_scan.py 的指标定义完全对齐。

架构（MHA，1Q对1K）：
  160m: 12层 12头 d_model=768  d_head=64
  410m: 24层 16头 d_model=1024 d_head=64
  1B:   16层  8头 d_model=2048 d_head=256
  1.4B: 24层 16头 d_model=2048 d_head=128
  2.8B: 32层 32头 d_model=2560 d_head=80

输出：
  /data/pythia_{size}_ssr_{run_ts}.csv   完整 per-head 记录
  每行 = 一个 (step, layer, head) 的 QK 指标对

CSV 字段：
  run_ts, model, step, layer, head,
  Q_eff_rank, Q_uni_iso, Q_theory_uniiso, Q_gap,
  Q_ssr, Q_sv_entropy, Q_sv_max_ratio, Q_sv1, Q_sv2, Q_sv3,
  K_eff_rank, K_uni_iso, K_sv_entropy
"""

import math
import time
import os
import csv
import numpy as np
import torch
from datetime import datetime
from core.fetcher import read_safetensors_header, load_tensors_batch, get_file_url
from core.debug import dprint

# ── Pythia 架构配置 ────────────────────────────────────────────────────────────
PYTHIA_CONFIGS = {
    "160m": {
        "model_id": "EleutherAI/pythia-160m",
        "n_layers":  12,
        "n_heads":   12,
        "d_model":   768,
        "d_head":    64,
    },
    "410m": {
        "model_id": "EleutherAI/pythia-410m",
        "n_layers":  24,
        "n_heads":   16,
        "d_model":   1024,
        "d_head":    64,
    },
    "1b": {
        "model_id": "EleutherAI/pythia-1b",
        "n_layers": 16,
        "n_heads": 8,
        "d_model": 2048,
        "d_head": 256,
    },
    "1.4b": {
        "model_id": "EleutherAI/pythia-1.4b",
        "n_layers": 24,
        "n_heads": 16,
        "d_model": 2048,
        "d_head": 128,
    },
    "2.8b": {
        "model_id": "EleutherAI/pythia-2.8b",
        "n_layers": 32,
        "n_heads": 32,
        "d_model": 2560,
        "d_head": 80,
    },
}

# log-spaced 早期 + 等间距中后期，共 20 个 checkpoint
DEFAULT_STEPS = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 3000, 5000, 10000, 20000,
    30000, 50000, 70000, 100000, 143000,
]

DATA_DIR = "/data"
ENERGY_THRESHOLD = 0.90   # eff_rank 定义：解释 90% 能量所需的奇异值数

# ── 指标函数（与 p4-lab3-llama_scan.py 完全对齐）─────────────────────────────

def effective_rank(sv_numpy: np.ndarray, threshold: float = ENERGY_THRESHOLD) -> int:
    """90% 能量阈值定义的有效秩（整数）"""
    sv2   = sv_numpy ** 2
    total = sv2.sum()
    if total < 1e-12:
        return 1
    cumvar = np.cumsum(sv2) / total
    return int(np.searchsorted(cumvar, threshold) + 1)


def uni_iso(W_numpy: np.ndarray) -> float:
    """
    UniIso(W) = ||G/||G||_F - I/||I||_F||_F
    G = W @ W.T,  scale-invariant isometry error
    与 Lab3 uni_iso_numpy() 完全相同。
    """
    G   = W_numpy @ W_numpy.T          # [d_h, d_h]
    G_f = np.linalg.norm(G, 'fro')
    d_h = W_numpy.shape[0]
    I   = np.eye(d_h, dtype=np.float32)
    I_f = math.sqrt(d_h)
    if G_f < 1e-12:
        return float(np.linalg.norm(I / I_f, 'fro'))
    return float(np.linalg.norm(G / G_f - I / I_f, 'fro'))


def theory_uniiso(d_h: int, k: int) -> float:
    """理论预测值：sqrt(2 - 2*sqrt(k)/sqrt(d_h))"""
    if k <= 0 or k >= d_h:
        return 0.0
    return math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(k) / math.sqrt(d_h)))


def sv_entropy(sv_numpy: np.ndarray) -> float:
    """奇异值熵（归一化后的 Shannon 熵，代理 rank）"""
    sv_norm = sv_numpy / (sv_numpy.sum() + 1e-12)
    return float(-np.sum(sv_norm * np.log(sv_norm + 1e-12)))


def compute_ssr(sq: np.ndarray, sk: np.ndarray) -> float:
    """SSR(W_Q, W_K)：奇异值归一化后的 L1 距离"""
    sq = sq / (np.linalg.norm(sq) + 1e-12)
    sk = sk / (np.linalg.norm(sk) + 1e-12)
    n  = min(len(sq), len(sk))
    return float(np.mean(np.abs(sq[:n] - sk[:n])))


# ── checkpoint URL 构造 ────────────────────────────────────────────────────────

def checkpoint_url(model_id: str, step: int) -> str:
    """
    Pythia checkpoint 存在 HF 的 step{N} branch。
    URL 格式：https://huggingface.co/{model_id}/resolve/step{N}/model.safetensors
    """
    return f"https://huggingface.co/{model_id}/resolve/step{step}/model.safetensors"


# ── 单个 checkpoint 扫描 ──────────────────────────────────────────────────────

def scan_checkpoint(model_id: str, step: int, cfg: dict, token: str = None) -> list:
    """
    扫描一个 checkpoint 的所有层所有头。
    返回 list of dict，每个 dict 对应一个 (layer, head)。
    """
    n_layers = cfg["n_layers"]
    n_heads  = cfg["n_heads"]
    d_head   = cfg["d_head"]
    d_model  = cfg["d_model"]

    url = checkpoint_url(model_id, step)
    dprint(f"[SCAN] step={step}  url={url}")

    # 读 header
    header, header_size = read_safetensors_header(url, token=token)

    # 构造所有层的 QKV weight key
    # Pythia weight key: gpt_neox.layers.{l}.attention.query_key_value.weight
    # shape: [3*d_model, d_model] = [2304, 768] for 160m
    qkv_keys = [
        f"gpt_neox.layers.{l}.attention.query_key_value.weight"
        for l in range(n_layers)
    ]

    # 一次 Range Request 读取所有层
    tensors = load_tensors_batch(url, qkv_keys, header, header_size, token=token)

    records = []
    for layer in range(n_layers):
        key = f"gpt_neox.layers.{layer}.attention.query_key_value.weight"
        if key not in tensors:
            dprint(f"[SCAN]   layer {layer} 缺失，跳过")
            continue

        W_qkv = tensors[key].numpy()   # [3*d_model, d_model]

        # 切出 Q 和 K（按行，各占 d_model 行）
        # Pythia QKV 合并：[Q(d_model), K(d_model), V(d_model), d_model]
        W_Q_full = W_qkv[0          : d_model,   :]   # [d_model, d_model]
        W_K_full = W_qkv[d_model    : 2*d_model, :]   # [d_model, d_model]

        for head in range(n_heads):
            # 每个 head 占 d_head 行
            Wq = W_Q_full[head*d_head : (head+1)*d_head, :]   # [d_head, d_model]
            Wk = W_K_full[head*d_head : (head+1)*d_head, :]   # [d_head, d_model]

            # SVD
            sq = np.linalg.svd(Wq, compute_uv=False)   # [d_head]
            sk = np.linalg.svd(Wk, compute_uv=False)

            # Q 指标
            q_eff_rank  = effective_rank(sq)
            q_uni_iso   = uni_iso(Wq)
            q_theory    = theory_uniiso(d_head, q_eff_rank)
            q_gap       = q_uni_iso - q_theory
            q_ssr       = compute_ssr(sq, sk)
            q_sv_ent    = sv_entropy(sq)
            q_sv_maxr   = float(sq[0] / (sq[1] + 1e-12))

            # K 指标（控制量）
            k_eff_rank  = effective_rank(sk)
            k_uni_iso   = uni_iso(Wk)
            k_sv_ent    = sv_entropy(sk)

            records.append({
                "layer":            layer,
                "head":             head,
                "Q_eff_rank":       q_eff_rank,
                "Q_uni_iso":        round(q_uni_iso, 6),
                "Q_theory_uniiso":  round(q_theory,  6),
                "Q_gap":            round(q_gap,     6),
                "Q_ssr":            round(q_ssr,     8),
                "Q_sv_entropy":     round(q_sv_ent,  4),
                "Q_sv_max_ratio":   round(q_sv_maxr, 4),
                "Q_sv1":            round(float(sq[0]), 4),
                "Q_sv2":            round(float(sq[1]), 4),
                "Q_sv3":            round(float(sq[2]), 4),
                "K_eff_rank":       k_eff_rank,
                "K_uni_iso":        round(k_uni_iso, 6),
                "K_sv_entropy":     round(k_sv_ent,  4),
            })

        del tensors[key]   # 及时释放内存

    return records


# ── 断点续跑：检查已完成的 steps ─────────────────────────────────────────────

def load_done_steps(csv_path: str) -> set:
    """从已有 CSV 中读出已完成的 step 集合"""
    if not os.path.exists(csv_path):
        return set()
    done = set()
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(int(row["step"]))
    except Exception:
        pass
    return done


# ── 主扫描函数（供 tab_pythia.py 调用）────────────────────────────────────────

CSV_FIELDS = [
    "run_ts", "model", "step", "layer", "head",
    "Q_eff_rank", "Q_uni_iso", "Q_theory_uniiso", "Q_gap",
    "Q_ssr", "Q_sv_entropy", "Q_sv_max_ratio",
    "Q_sv1", "Q_sv2", "Q_sv3",
    "K_eff_rank", "K_uni_iso", "K_sv_entropy",
]


def scan_pythia(
    model_size: str = "160m",
    steps: list = None,
    token: str = None,
    progress_fn=None,       # Gradio gr.Progress 回调，接受 (current, total, desc)
) -> str:
    """
    扫描 Pythia 所有指定 checkpoint。

    参数：
        model_size  : "160m" 或 "410m"
        steps       : checkpoint step 列表，默认 DEFAULT_STEPS
        token       : HF token（Pythia 是公开模型，通常不需要）
        progress_fn : 进度回调，每完成一个 step 调用一次

    返回：
        csv_path（写入 /data 目录）
    """
    if steps is None:
        steps = DEFAULT_STEPS

    cfg      = PYTHIA_CONFIGS[model_size]
    model_id = cfg["model_id"]
    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, f"pythia_{model_size}_ssr_{run_ts}.csv")

    # 断点续跑：检查是否已有进行中的 CSV
    # 策略：同一 model_size 的最新文件若存在则续跑
    existing = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.startswith(f"pythia_{model_size}_ssr_") and f.endswith(".csv")
    ])
    if existing:
        csv_path  = os.path.join(DATA_DIR, existing[-1])
        done_steps = load_done_steps(csv_path)
        steps_todo = [s for s in steps if s not in done_steps]
        print(f"[SCAN] 续跑模式：已完成 {len(done_steps)} steps，"
              f"待跑 {len(steps_todo)} steps")
        file_mode = "a"   # 追加
    else:
        done_steps = set()
        steps_todo = steps
        file_mode  = "w"  # 新建

    if not steps_todo:
        print("[SCAN] 所有 step 已完成，直接返回")
        return csv_path

    with open(csv_path, file_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if file_mode == "w":
            writer.writeheader()

        total = len(steps_todo)
        for idx, step in enumerate(steps_todo):
            print(f"\n[SCAN] === step {step} ({idx+1}/{total}) ===", flush=True)
            t0 = time.time()

            try:
                records = scan_checkpoint(model_id, step, cfg, token=token)
            except Exception as e:
                print(f"[SCAN]   step {step} 失败: {e}", flush=True)
                if progress_fn:
                    progress_fn(idx+1, total, f"step {step} 失败: {e}")
                continue

            # 写入 CSV
            for rec in records:
                row = {"run_ts": run_ts, "model": model_id, "step": step}
                row.update(rec)
                writer.writerow(row)
            f.flush()

            elapsed = time.time() - t0
            n_heads_total = len(records)
            # pseudo-bulk 摘要（per-layer median）用于实时 log
            if records:
                import pandas as pd
                df_step = pd.DataFrame(records)
                pb = df_step.groupby("layer")["Q_uni_iso"].median()
                ui_med  = float(pb.median())
                rk_med  = float(df_step.groupby("layer")["Q_eff_rank"].median().median())
                ssr_med = float(df_step.groupby("layer")["Q_ssr"].median().median())
                print(f"[SCAN]   {n_heads_total} heads  "
                      f"Q_uni_iso(pb_med)={ui_med:.4f}  "
                      f"Q_eff_rank(pb_med)={rk_med:.1f}  "
                      f"Q_ssr(pb_med)={ssr_med:.6f}  "
                      f"耗时={elapsed:.1f}s", flush=True)

            if progress_fn:
                progress_fn(idx+1, total,
                            f"step {step} 完成  uni_iso={ui_med:.4f}  "
                            f"eff_rank={rk_med:.1f}  耗时={elapsed:.1f}s")

    print(f"\n[SCAN] 完成，输出: {csv_path}")
    return csv_path
