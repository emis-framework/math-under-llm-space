# core/metrics.py
import torch
import numpy as np
from scipy.stats import spearmanr
from core.layer_profile import LayerProfile


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    am, bm = a - a.mean(), b - b.mean()
    den = torch.norm(am) * torch.norm(bm)
    return float(torch.dot(am, bm) / den) if den > 1e-10 else 0.0


def spearman_r(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(spearmanr(a.numpy(), b.numpy())[0])


def ssr(a: torch.Tensor, b: torch.Tensor) -> float:
    n  = min(a.shape[0], b.shape[0])
    an = a[:n] / (torch.norm(a[:n]) + 1e-10)
    bn = b[:n] / (torch.norm(b[:n]) + 1e-10)
    return float(torch.mean(torch.abs(an - bn)))


def svr(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """
    最小二乘法拟合：alpha = argmin ||s_a - alpha * s_b||^2
    返回 (alpha, residual)
    residual = mean((s_a - alpha * s_b)^2)
    """
    n       = min(a.shape[0], b.shape[0])
    sa, sb  = a[:n], b[:n]
    den     = torch.dot(sb, sb)
    if den < 1e-10:
        return 1.0, 0.0
    alpha   = torch.dot(sa, sb) / den
    residual= float(torch.mean((sa - alpha * sb) ** 2))
    return float(alpha), residual


def cos_U(U_a: torch.Tensor, U_b: torch.Tensor) -> float:
    r  = min(U_a.shape[0], U_b.shape[0])
    c  = min(U_a.shape[1], U_b.shape[1])
    Ua = U_a[:r, :c] / (torch.norm(U_a[:r, :c], dim=0, keepdim=True) + 1e-10)
    Ub = U_b[:r, :c] / (torch.norm(U_b[:r, :c], dim=0, keepdim=True) + 1e-10)
    return float(torch.diag(torch.abs(Ua.T @ Ub)).mean())


def cos_V(Vt_a: torch.Tensor, Vt_b: torch.Tensor) -> float:
    r  = min(Vt_a.shape[0], Vt_b.shape[0])
    c  = min(Vt_a.shape[1], Vt_b.shape[1])
    Va = Vt_a[:r, :c] / (torch.norm(Vt_a[:r, :c], dim=1, keepdim=True) + 1e-10)
    Vb = Vt_b[:r, :c] / (torch.norm(Vt_b[:r, :c], dim=1, keepdim=True) + 1e-10)
    return float(torch.abs((Va * Vb).sum(dim=1)).mean())


def sigma_stats(s: torch.Tensor) -> tuple[float, float, float]:
    """
    返回 (sigma_max, sigma_min, cond)
    sigma_min 过滤接近零的奇异值，避免条件数虚高
    """
    s_max  = float(s.max())
    valid  = s[s > 1e-10]
    s_min  = float(valid.min()) if valid.numel() > 0 else 0.0
    cond   = s_max / (s_min + 1e-10)
    return s_max, s_min, cond


def analyze_layer(
    W_q:     torch.Tensor,
    W_k:     torch.Tensor,
    W_v:     torch.Tensor,
    profile: LayerProfile,
) -> tuple[list[dict], str]:

    n_q       = profile.n_q_heads
    n_kv      = profile.n_kv_heads
    d_head    = profile.head_dim
    kv_shared = profile.kv_shared
    group     = n_q // n_kv

    records: list[dict] = []
    lines:   list[str]  = []

    # ── 调试：打印整体 shape ──────────────────────
    lines.append(
        f"\n{'─'*80}\n"
        f"[DEBUG] W_q={list(W_q.shape)} W_k={list(W_k.shape)} "
        f"W_v={list(W_v.shape)}\n"
        f"[DEBUG] n_q={n_q} n_kv={n_kv} group={group} "
        f"d_head={d_head} source={profile.head_dim_source}\n"
    )

    kv_tag = " [K=V共享]" if kv_shared else ""
    lines.append(
        f"[{profile.prefix}] Layer {profile.layer_idx:3d}{kv_tag}  "
        f"n_q={n_q} n_kv={n_kv} group={group} "
        f"d_head={d_head}({profile.head_dim_source})\n"
        f"{'─'*80}\n"
        f"  {'KV':>3} {'Q':>3} │"
        f" {'P_QK':>7} {'Sp_QK':>7} {'SSR_QK':>8} │"
        f" {'SSR_QV':>8} {'SSR_KV':>8} │"
        f" {'cosU_QK':>8} {'cosU_QV':>8} {'cosU_KV':>8} │"
        f" {'cosV_QK':>8} {'cosV_QV':>8} {'cosV_KV':>8} │"
        f" {'α_QK':>7} {'α_QV':>7} {'α_KV':>7}\n"
    )

    for kv_h in range(n_kv):
        k_t = W_k[kv_h * d_head:(kv_h + 1) * d_head, :]
        v_t = W_v[kv_h * d_head:(kv_h + 1) * d_head, :]

        U_k, s_k, Vt_k = torch.linalg.svd(k_t, full_matrices=False)
        U_v, s_v, Vt_v = torch.linalg.svd(v_t, full_matrices=False)

        smxk, smnk, cond_k = sigma_stats(s_k)
        smxv, smnv, cond_v = sigma_stats(s_v)

        # ── 调试：打印每个 KV 头的切片和奇异值 ──────
        lines.append(
            f"[DEBUG] KV头{kv_h}: "
            f"k_t={list(k_t.shape)} "
            f"s_k前5={s_k[:5].tolist()}\n"
            f"[DEBUG] KV头{kv_h}: "
            f"v_t={list(v_t.shape)} "
            f"s_v前5={s_v[:5].tolist()}\n"
        )

        # KV 指标
        if kv_shared:
            ssr_kv   = 0.0
            pkv      = 1.0
            cosU_KV  = 1.0
            cosV_KV  = 1.0
            alpha_kv = 1.0
            res_kv   = 0.0
        else:
            n_kv_sv       = min(len(s_k), len(s_v))
            ssr_kv        = ssr(s_k, s_v)
            pkv           = pearson(s_k[:n_kv_sv], s_v[:n_kv_sv])
            cosU_KV       = cos_U(U_k, U_v)
            cosV_KV       = cos_V(Vt_k, Vt_v)
            alpha_kv, res_kv = svr(s_k, s_v)

        for q_off in range(group):
            h   = kv_h * group + q_off
            q_t = W_q[h * d_head:(h + 1) * d_head, :]
            U_q, s_q, Vt_q = torch.linalg.svd(q_t, full_matrices=False)

            smxq, smnq, cond_q = sigma_stats(s_q)

            # ── 调试：打印每个 Q 头的切片和奇异值 ────
            lines.append(
                f"[DEBUG]   Q头{h}: "
                f"q_t={list(q_t.shape)} "
                f"s_q前5={s_q[:5].tolist()}\n"
            )

            nqk = min(len(s_q), len(s_k))
            nqv = min(len(s_q), len(s_v))

            # QK
            pqk        = pearson(s_q[:nqk], s_k[:nqk])
            spqk       = spearman_r(s_q[:nqk], s_k[:nqk])
            ssr_qk     = ssr(s_q, s_k)
            a_qk, r_qk = svr(s_q, s_k)
            cU_QK      = cos_U(U_q, U_k)
            cV_QK      = cos_V(Vt_q, Vt_k)

            # QV
            pqv        = pearson(s_q[:nqv], s_v[:nqv])
            ssr_qv     = ssr(s_q, s_v)
            a_qv, r_qv = svr(s_q, s_v)
            cU_QV      = cos_U(U_q, U_v)
            cV_QV      = cos_V(Vt_q, Vt_v)

            # ── 调试：打印关键指标 ────────────────────
            lines.append(
                f"[DEBUG]   Q头{h}: "
                f"pearson={pqk:+.4f} "
                f"alpha_QK={a_qk:.4f} "
                f"s_q[0]={s_q[0]:.4f} "
                f"s_k[0]={s_k[0]:.4f}\n"
            )

            records.append({
                "prefix":        profile.prefix,
                "layer":         profile.layer_idx,
                "kv_head":       kv_h,
                "q_head":        h,
                "kv_shared":     kv_shared,
                "pearson_QK":    round(pqk,    6),
                "spearman_QK":   round(spqk,   6),
                "pearson_QV":    round(pqv,    6),
                "pearson_KV":    round(pkv,    6),
                "ssr_QK":        round(ssr_qk,  8),
                "ssr_QV":        round(ssr_qv,  8),
                "ssr_KV":        round(ssr_kv,  8),
                "cosU_QK":       round(cU_QK,   6),
                "cosU_QV":       round(cU_QV,   6),
                "cosU_KV":       round(cosU_KV, 6),
                "cosV_QK":       round(cV_QK,   6),
                "cosV_QV":       round(cV_QV,   6),
                "cosV_KV":       round(cosV_KV, 6),
                "alpha_QK":      round(a_qk,    4),
                "alpha_QV":      round(a_qv,    4),
                "alpha_KV":      round(alpha_kv,4),
                "alpha_res_QK":  round(r_qk,    6),
                "alpha_res_QV":  round(r_qv,    6),
                "alpha_res_KV":  round(res_kv,  6),
                "sigma_max_Q":   round(smxq, 4),
                "sigma_min_Q":   round(smnq, 4),
                "sigma_max_K":   round(smxk, 4),
                "sigma_min_K":   round(smnk, 4),
                "sigma_max_V":   round(smxv, 4),
                "sigma_min_V":   round(smnv, 4),
                "cond_Q":        round(cond_q, 2),
                "cond_K":        round(cond_k, 2),
                "cond_V":        round(cond_v, 2),
                "head_dim":      d_head,
                "d_model":       profile.d_model,
                "n_q_heads":     n_q,
                "n_kv_heads":    n_kv,
            })

            lines.append(
                f"  {kv_h:>3d} {h:>3d} │"
                f" {pqk:>+7.4f} {spqk:>+7.4f} {ssr_qk:>8.6f} │"
                f" {ssr_qv:>8.6f} {ssr_kv:>8.6f} │"
                f" {cU_QK:>8.4f} {cU_QV:>8.4f} {cosU_KV:>8.4f} │"
                f" {cV_QK:>8.4f} {cV_QV:>8.4f} {cosV_KV:>8.4f} │"
                f" {a_qk:>7.4f} {a_qv:>7.4f} {alpha_kv:>7.4f}\n"
            )

    return records, "".join(lines)


def summarize_records(records: list[dict], model_id: str) -> str:
    if not records:
        return "❌ 无记录\n"

    import pandas as pd
    df = pd.DataFrame(records)

    def stat(arr, name):
        if len(arr) == 0:
            return f"  {name:<14} 无数据\n"
        return (
            f"  {name:<14}"
            f" Median={np.median(arr):.6f}"
            f" Mean={np.mean(arr):.6f}"
            f" Min={np.min(arr):.6f}"
            f" Max={np.max(arr):.6f}\n"
        )

    lines = [
        f"\n{'═'*80}\n",
        f"📊 王氏五定律汇总 — {model_id}\n",
        f"{'═'*80}\n",
    ]

    for pfx in sorted(df["prefix"].unique()):
        pdf      = df[df["prefix"] == pfx]
        real_kv  = pdf[~pdf["kv_shared"]]
        kv_df    = real_kv if len(real_kv) > 0 else pdf

        lines.append(
            f"\n▶ {pfx}\n"
            f"  记录：{len(pdf)} 条，"
            f"层：{sorted(pdf['layer'].unique())}\n"
        )
        if pdf["kv_shared"].any():
            n_shared = pdf[pdf["kv_shared"]]["layer"].nunique()
            lines.append(f"  ⚠️  含 {n_shared} 个 K=V共享层，KV指标为理论值\n")

        lines += [
            "  【第一定律 Pearson r → 1】\n",
            stat(pdf["pearson_QK"].values, "Q-K:"),
            stat(pdf["pearson_QV"].values, "Q-V:"),
            stat(kv_df["pearson_KV"].values, "K-V(实):"),
            "  【第二定律 SSR → 0】\n",
            stat(pdf["ssr_QK"].values, "Q-K:"),
            stat(pdf["ssr_QV"].values, "Q-V:"),
            stat(kv_df["ssr_KV"].values, "K-V(实):"),
            "  【第四定律 cosU 输出子空间】\n",
            stat(pdf["cosU_QK"].values, "cosU Q-K:"),
            stat(pdf["cosU_QV"].values, "cosU Q-V:"),
            stat(kv_df["cosU_KV"].values, "cosU K-V:"),
            "  【第五定律 cosV 输入子空间】\n",
            stat(pdf["cosV_QK"].values, "cosV Q-K:"),
            stat(pdf["cosV_QV"].values, "cosV Q-V:"),
            stat(kv_df["cosV_KV"].values, "cosV K-V:"),
            "  【第三定律 条件数（sigma_min 已过滤零值）】\n",
            stat(pdf["cond_Q"].values, "cond Q:"),
            stat(pdf["cond_K"].values, "cond K:"),
            stat(pdf["cond_V"].values, "cond V:"),
        ]

    lines.append(
        f"\n⚡ 理论极值：Pearson→1, SSR→0, cosU(QV)<1/√d_head\n"
        f"{'═'*80}\n"
    )
    return "".join(lines)