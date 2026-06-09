# core/olmoe_scanner.py
"""
OLMoE Checkpoint SSR/UniIso 动态扫描
======================================
扫描 allenai/OLMoE-1B-7B-0924 的多个训练 checkpoint，
计算每层每头的 SSR、UniIso、eff_rank 等指标。

架构（标准MHA，Q/K分开存储，BF16）：
  OLMoE-1B-7B-0924: 16层 16头 d_model=2048 d_head=128

和 pythia_scanner.py 的关键区别：
  Pythia：query_key_value合并 [3*d_model, d_model]
  OLMoE： q_proj/k_proj分开  [d_model, d_model] × 2
  Pythia：branch = step{N}
  OLMoE： branch = step{N}-tokens{M}B

输出：
  /data/olmoe_{model_name}_ssr_{run_ts}.csv
  指标定义与 pythia_scanner.py 完全对齐
"""

import math
import time
import os
import csv
import re
import numpy as np
import torch
from datetime import datetime
from core.fetcher import read_safetensors_header, load_tensors_batch
from core.debug import dprint

# ── OLMoE 模型配置 ─────────────────────────────────────────────────────────────
OLMOE_CONFIGS = {
    "OLMoE-1B-7B-0924": {
        "model_id": "allenai/OLMoE-1B-7B-0924",
        "n_layers":  16,
        "n_heads":   16,
        "d_model":   2048,
        "d_head":    128,
        "n_kv_heads": 16,    # MHA，KV头数=Q头数
        "dtype":     "BF16",
        "q_key_fmt": "model.layers.{l}.self_attn.q_proj.weight",
        "k_key_fmt": "model.layers.{l}.self_attn.k_proj.weight",
        "n_shards":  3,      # model-0000{1,2,3}-of-00003.safetensors
    },
}

# 删除原来的GROKKING_REGIONS（如果有）
# 新增：
GROKKING_THRESHOLD_STEP   = 980000        # step980k，3/4 domain acc>0.9
GROKKING_THRESHOLD_TOKENS = 4110          # 对应4110B tokens
GROKKING_START_TOKENS     = 2580          # step615k，benchmark acc开始回升

DATA_DIR = "/data"
ENERGY_THRESHOLD = 0.90

# 默认扫描步骤：均匀采样20个，覆盖完整训练
# 实际branch名在运行时动态解析（step→branch映射）
DEFAULT_STEPS_OLMOE = [
    5000, 25000, 50000, 75000, 100000,
    150000, 200000, 300000, 400000, 500000,
    600000, 700000, 800000, 900000, 1000000,
    1050000, 1100000, 1150000, 1200000, 1220000,
]

# ── 指标函数（与 pythia_scanner.py 完全对齐）──────────────────────────────────

def effective_rank(sv: np.ndarray, threshold: float = ENERGY_THRESHOLD) -> int:
    sv2   = sv ** 2
    total = sv2.sum()
    if total < 1e-12:
        return 1
    cumvar = np.cumsum(sv2) / total
    return int(np.searchsorted(cumvar, threshold) + 1)


def uni_iso(W: np.ndarray) -> float:
    G   = W @ W.T
    G_f = np.linalg.norm(G, 'fro') + 1e-12
    d_h = W.shape[0]
    I_f = math.sqrt(d_h)
    return float(np.linalg.norm(G / G_f - np.eye(d_h) / I_f, 'fro'))


def theory_uniiso(d_h: int, k: int) -> float:
    return math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(k) / math.sqrt(d_h)))


def sv_entropy(sv: np.ndarray) -> float:
    p = sv / (sv.sum() + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))


def compute_ssr(sq: np.ndarray, sk: np.ndarray) -> float:
    n  = min(len(sq), len(sk))
    sq = sq[:n] / (np.linalg.norm(sq[:n]) + 1e-12)
    sk = sk[:n] / (np.linalg.norm(sk[:n]) + 1e-12)
    return float(np.mean(np.abs(sq - sk)))


# ── Branch 映射（step → branch名）────────────────────────────────────────────

def build_step_branch_map(model_id: str) -> dict:
    """
    从HF拉取所有branch，建立 {step: branch_name} 映射。
    branch格式：step{N}-tokens{M}B
    """
    from huggingface_hub import list_repo_refs
    refs = list_repo_refs(model_id)
    step_map = {}
    for b in refs.branches:
        m = re.match(r'step(\d+)-tokens(\d+)B', b.name)
        if m:
            step = int(m.group(1))
            step_map[step] = b.name
    dprint(f"[OLMOE] {len(step_map)} checkpoints found")
    return step_map


# ── 分片处理 ───────────────────────────────────────────────────────────────────

def get_shard_urls(model_id: str, branch: str, cfg: dict) -> dict:
    """
    读取 model.safetensors.index.json，返回
    {shard_filename: (url, header, header_size)}
    只加载包含QK权重的分片。
    """
    import requests
    n_layers = cfg["n_layers"]

    # 收集所有需要的key
    needed_keys = set()
    for l in range(n_layers):
        needed_keys.add(cfg["q_key_fmt"].format(l=l))
        needed_keys.add(cfg["k_key_fmt"].format(l=l))

    # 读index.json
    idx_url = (f"https://huggingface.co/{model_id}"
               f"/resolve/{branch}/model.safetensors.index.json")
    r = requests.get(idx_url, timeout=30)
    r.raise_for_status()
    wmap = r.json()["weight_map"]

    # 按分片分组
    shard_to_keys = {}
    for k in needed_keys:
        shard = wmap.get(k)
        if shard:
            shard_to_keys.setdefault(shard, []).append(k)

    # 读各分片header
    shard_info = {}
    for shard_name, keys in shard_to_keys.items():
        url = (f"https://huggingface.co/{model_id}"
               f"/resolve/{branch}/{shard_name}")
        header, header_size = read_safetensors_header(url)
        shard_info[shard_name] = (url, header, header_size, keys)

    return shard_info


# ── 单个 checkpoint 扫描 ──────────────────────────────────────────────────────

def scan_checkpoint_olmoe(
    model_id: str,
    branch: str,
    step: int,
    cfg: dict,
    token: str = None,
) -> list:
    """
    扫描一个OLMoE checkpoint的所有层所有头。
    Q和K分开读取，各自计算指标后合并。
    返回 list of dict，每个dict对应一个(layer, head)。
    """
    n_layers  = cfg["n_layers"]
    n_heads   = cfg["n_heads"]
    d_head    = cfg["d_head"]
    d_model   = cfg["d_model"]

    dprint(f"[SCAN] step={step}  branch={branch}")

    # 读分片信息
    shard_info = get_shard_urls(model_id, branch, cfg)

    # 从各分片批量读取QK权重
    all_tensors = {}
    for shard_name, (url, header, header_size, keys) in shard_info.items():
        t = load_tensors_batch(url, keys, header, header_size, token=token)
        all_tensors.update(t)

    records = []
    for layer in range(n_layers):
        q_key = cfg["q_key_fmt"].format(l=layer)
        k_key = cfg["k_key_fmt"].format(l=layer)

        if q_key not in all_tensors or k_key not in all_tensors:
            dprint(f"[SCAN] layer {layer} 缺失QK，跳过")
            continue

        # shape: [d_model, d_model] → 按head切片
        # OLMoE的q_proj: [n_heads*d_head, d_model]
        W_Q = all_tensors[q_key].numpy()  # [2048, 2048]
        W_K = all_tensors[k_key].numpy()

        for head in range(n_heads):
            # 切出单个head
            Wq = W_Q[head*d_head:(head+1)*d_head, :]  # [128, 2048]
            Wk = W_K[head*d_head:(head+1)*d_head, :]

            sq = np.linalg.svd(Wq, compute_uv=False)
            sk = np.linalg.svd(Wk, compute_uv=False)

            q_eff_rank = effective_rank(sq)
            q_uni_iso  = uni_iso(Wq)
            q_theory   = theory_uniiso(d_head, q_eff_rank)
            q_gap      = q_uni_iso - q_theory
            q_ssr      = compute_ssr(sq, sk)
            q_sv_ent   = sv_entropy(sq)
            q_sv_maxr  = float(sq[0] / (sq[1] + 1e-12))

            k_eff_rank = effective_rank(sk)
            k_uni_iso  = uni_iso(Wk)
            k_sv_ent   = sv_entropy(sk)

            records.append({
                "layer":           layer,
                "head":            head,
                "Q_eff_rank":      q_eff_rank,
                "Q_uni_iso":       round(q_uni_iso,  6),
                "Q_theory_uniiso": round(q_theory,   6),
                "Q_gap":           round(q_gap,      6),
                "ssr":             round(q_ssr,      8),
                "Q_sv_entropy":    round(q_sv_ent,   4),
                "Q_sv_max_ratio":  round(q_sv_maxr,  4),
                "Q_sv1":           round(float(sq[0]), 4),
                "Q_sv2":           round(float(sq[1]), 4),
                "Q_sv3":           round(float(sq[2]), 4),
                "K_eff_rank":      k_eff_rank,
                "K_uni_iso":       round(k_uni_iso,  6),
                "K_sv_entropy":    round(k_sv_ent,   4),
            })

        del all_tensors[q_key], all_tensors[k_key]

    return records


# ── CSV字段 ────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "run_ts", "model", "step", "branch", "tokens_B", "layer", "head",
    "Q_eff_rank", "Q_uni_iso", "Q_theory_uniiso", "Q_gap",
    "ssr", "Q_sv_entropy", "Q_sv_max_ratio",
    "Q_sv1", "Q_sv2", "Q_sv3",
    "K_eff_rank", "K_uni_iso", "K_sv_entropy",
]


# ── 断点续跑 ───────────────────────────────────────────────────────────────────

def load_done_steps(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    done = set()
    try:
        with open(csv_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                done.add(int(row["step"]))
    except Exception:
        pass
    return done


# ── 主扫描函数 ─────────────────────────────────────────────────────────────────

def scan_olmoe(
    model_name: str = "OLMoE-1B-7B-0924",
    steps: list = None,
    token: str = None,
    progress_fn=None,
) -> str:
    """
    扫描OLMoE所有指定checkpoint。

    参数：
        model_name  : OLMOE_CONFIGS的key
        steps       : 要扫描的step列表，默认DEFAULT_STEPS_OLMOE
        token       : HF token（OLMoE是公开模型，通常不需要）
        progress_fn : Gradio进度回调

    返回：csv_path
    """
    if steps is None:
        steps = DEFAULT_STEPS_OLMOE

    cfg      = OLMOE_CONFIGS[model_name]
    model_id = cfg["model_id"]
    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(DATA_DIR, exist_ok=True)

    # 建立step→branch映射
    print(f"[OLMOE] 获取checkpoint列表...", flush=True)
    step_map = build_step_branch_map(model_id)
    available = set(step_map.keys())
    steps_valid = [s for s in steps if s in available]
    steps_missing = [s for s in steps if s not in available]
    if steps_missing:
        print(f"[OLMOE] 以下step不存在，跳过：{steps_missing}", flush=True)

    # 断点续跑
    existing = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.startswith(f"olmoe_{model_name}_ssr_") and f.endswith(".csv")
    ])
    if existing:
        csv_path   = os.path.join(DATA_DIR, existing[-1])
        done_steps = load_done_steps(csv_path)
        steps_todo = [s for s in steps_valid if s not in done_steps]
        print(f"[OLMOE] 续跑：已完成{len(done_steps)}步，待跑{len(steps_todo)}步",
              flush=True)
        file_mode = "a"
    else:
        csv_path   = os.path.join(DATA_DIR,
                                  f"olmoe_{model_name}_ssr_{run_ts}.csv")
        done_steps = set()
        steps_todo = steps_valid
        file_mode  = "w"

    if not steps_todo:
        print("[OLMOE] 所有step已完成", flush=True)
        return csv_path

    with open(csv_path, file_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if file_mode == "w":
            writer.writeheader()

        total = len(steps_todo)
        for idx, step in enumerate(steps_todo):
            branch   = step_map[step]
            tokens_B = int(re.search(r'tokens(\d+)B', branch).group(1))

            print(f"\n[OLMOE] === step {step} ({idx+1}/{total}) "
                  f"branch={branch} ===", flush=True)
            t0 = time.time()

            try:
                records = scan_checkpoint_olmoe(
                    model_id, branch, step, cfg, token=token)
            except Exception as e:
                print(f"[OLMOE] step {step} 失败: {e}", flush=True)
                if progress_fn:
                    progress_fn(idx+1, total, f"step {step} 失败: {e}")
                continue

            for rec in records:
                row = {
                    "run_ts":   run_ts,
                    "model":    model_id,
                    "step":     step,
                    "branch":   branch,
                    "tokens_B": tokens_B,
                }
                row.update(rec)
                writer.writerow(row)
            f.flush()

            elapsed = time.time() - t0
            if records:
                import pandas as pd
                df_s  = pd.DataFrame(records)
                pb    = df_s.groupby("layer")["Q_uni_iso"].median()
                ui_m  = float(pb.median())
                rk_m  = float(df_s.groupby("layer")["Q_eff_rank"]
                              .median().median())
                ssr_m = float(df_s.groupby("layer")["ssr"].median().median())
                print(f"[OLMOE]   {len(records)} heads  "
                      f"Q_uni_iso(pb)={ui_m:.4f}  "
                      f"Q_eff_rank(pb)={rk_m:.1f}  "
                      f"ssr(pb)={ssr_m:.6f}  "
                      f"耗时={elapsed:.1f}s", flush=True)

            if progress_fn:
                progress_fn(idx+1, total,
                            f"step {step} ({tokens_B}B tokens) 完成  "
                            f"耗时={elapsed:.1f}s")

    print(f"\n[OLMOE] 完成，输出: {csv_path}")
    return csv_path
