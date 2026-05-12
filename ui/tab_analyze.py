# ui/tab_analyze.py
"""
Tab2: Analyze a single model
- Auto-infer structure via LayerProfile
- Filter layers by start_layer / end_layer (raw index)
- Compute all Wang's Five Laws metrics per head
- Write results to SQLite if admin token is valid (read-only for reviewers)
"""

import gradio as gr
import requests
import pandas as pd
import numpy as np
from datetime import datetime

from core.debug import dlog
from core.fetcher import (
    load_all_shard_headers,
    load_tensor_remote,
    get_file_url,
    check_quantization,
    http_error_msg,
)
from core.layer_profile import (
    scan_model_structure,
    extract_config_params,
)
from core.metrics import analyze_layer, summarize_records

from db.schema import init_db
from db.writer import (
    upsert_model,
    upsert_component,
    write_layer_records,
    update_model_summary,
    get_analyzed_layers,
    infer_layer_type,
    check_write_permission,
)

SIDEBAR_MD = """
### Recommended Models
`google/gemma-4-e2b`  
`google/gemma-4-e4b-it`  
`google/gemma-4-31b-it`  
`Qwen/Qwen2.5-14B-Instruct`  
`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`  
`meta-llama/Meta-Llama-3-8B`  

---

### Layer Index
- Layer index = **N** in `layers.{N}` of safetensors keys
- Raw index, **not re-numbered per component**
- Multi-modal models (e.g. Gemma-4):
  - `layers.0~11` may contain audio / vision / text layers
  - All components output separately, distinguished by prefix

### Example: Gemma-4-E2B
| Component | Layer Range |
|-----------|-------------|
| audio_tower | 0 ~ 11 |
| language_model | 0 ~ 34 |
| vision_tower | 0 ~ 15 |

### Example: Gemma-4-31B
| Component | Layer Range |
|-----------|-------------|
| language (local) | 0 ~ 59 |
| language (global) | 5, 11, 17 … 59 |
| vision_tower | 0 ~ 26 |

---

### Reviewer Note
Leave **Admin Write Token** empty to run the full analysis  
without writing to the database.  
All metrics are computed and displayed normally.
"""


def run_analysis(
    model_id:    str,
    hf_token:    str,
    start_layer: int,
    end_layer:   int,
    admin_token: str,
    progress=gr.Progress()
) -> tuple[str, pd.DataFrame]:

    if not model_id.strip():
        return "❌ Please enter a model ID.", None

    token      = hf_token.strip() or None
    start_l    = int(start_layer)
    end_l      = int(end_layer)
    t_start    = datetime.utcnow()
    can_write  = check_write_permission(admin_token)

    log = [
        f"🔍 Analyzing: {model_id}  layers {start_l}~{end_l}\n"
        f"{'═'*80}\n"
        f"💾 Database write: {'✅ ENABLED (admin)' if can_write else '🔒 DISABLED (read-only mode)'}\n"
        f"{'═'*80}\n"
    ]

    if not can_write:
        log.append(
            "ℹ️  Running in read-only mode.\n"
            "   Analysis will run normally. Results displayed below but NOT saved to DB.\n"
            "   Reviewers: this is intentional — full reproducibility without DB access.\n"
            f"{'─'*80}\n"
        )

    all_records: list[dict] = []

    # ── DB connection (needed for resume check even in read-only) ──
    conn = init_db()

    # ── Quantization check ────────────────────────────────────────
    progress(0.02, desc="Checking quantization...")
    blocked, qmsg = check_quantization(model_id, token)
    log.append(f"[Quantization Check]\n{qmsg}\n{'─'*80}\n")
    if blocked:
        return "".join(log), None

    # ── config.json ───────────────────────────────────────────────
    progress(0.05, desc="Reading config...")
    config_params = {}
    try:
        r = requests.get(
            f"https://huggingface.co/{model_id}/resolve/main/config.json",
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=15
        )
        if r.status_code == 200:
            config_params = extract_config_params(r.json())
            log.append(
                f"📋 Config: model_type={config_params.get('model_type')}  "
                f"head_dim={config_params.get('head_dim')}\n"
                f"{'─'*80}\n"
            )
    except Exception:
        log.append("⚠️  Could not read config.json\n")

    # ── Write model metadata (admin only) ────────────────────────
    if can_write:
        model_type = config_params.get("model_type", "unknown")
        upsert_model(conn, model_id, model_type=model_type)

    # ── Load all shard headers ────────────────────────────────────
    progress(0.08, desc="Loading shard headers...")
    try:
        all_headers = load_all_shard_headers(model_id, token)
    except requests.exceptions.HTTPError as e:
        return http_error_msg(e, model_id), None
    except Exception as e:
        return "".join(log) + f"❌ Failed to load headers: {e}\n", None

    log.append(
        f"📦 Shards: {len(all_headers)}  "
        f"Total keys: {sum(len(h) for h,_ in all_headers.values())}\n"
    )

    # ── Scan layer structure ──────────────────────────────────────
    progress(0.12, desc="Scanning layer structure...")
    profiles = scan_model_structure(all_headers, config_params)

    if not profiles:
        return "".join(log) + "⚠️ No Q/K/V layers found.\n", None

    # ── Write component metadata (admin only) ────────────────────
    if can_write:
        by_prefix: dict[str, list] = {}
        for (pfx, idx), prof in profiles.items():
            by_prefix.setdefault(pfx, []).append(prof)

        for pfx, profs in by_prefix.items():
            complete_profs = [p for p in profs if p.complete]
            if not complete_profs:
                continue
            head_dims  = [p.head_dim for p in complete_profs]
            has_shared = any(p.kv_shared for p in complete_profs)
            d_models   = [p.d_model for p in complete_profs if p.d_model > 0]
            upsert_component(
                conn          = conn,
                model_id      = model_id,
                prefix        = pfx,
                n_layers      = len(complete_profs),
                head_dim_min  = min(head_dims),
                head_dim_max  = max(head_dims),
                has_kv_shared = has_shared,
                has_global    = has_shared,
                d_model       = d_models[0] if d_models else 0,
            )

    # ── Filter by layer range ─────────────────────────────────────
    filtered = {
        (pfx, idx): prof
        for (pfx, idx), prof in profiles.items()
        if start_l <= idx <= end_l and prof.complete
    }

    if not filtered:
        by_pfx_all: dict[str, list] = {}
        for (pfx, idx) in profiles:
            by_pfx_all.setdefault(pfx, []).append(idx)
        info = "\n".join(
            f"  '{p}': {sorted(v)}"
            for p, v in sorted(by_pfx_all.items())
        )
        return (
            "".join(log) +
            f"⚠️ No complete layers found in range {start_l}~{end_l}.\n"
            f"Available layer indices:\n{info}\n", None
        )

    # ── Resume check (always query DB, write only if can_write) ──
    done_layers: dict[str, set] = {}
    for pfx in set(pfx for pfx, _ in filtered):
        done_layers[pfx] = get_analyzed_layers(conn, model_id, pfx)

    # ── Print analysis plan ───────────────────────────────────────
    by_pfx2: dict[str, list] = {}
    for (pfx, idx) in filtered:
        by_pfx2.setdefault(pfx, []).append(idx)

    log.append("📐 Analysis plan:\n")
    skipped_total = 0
    for pfx, idxs in sorted(by_pfx2.items()):
        done = done_layers.get(pfx, set())
        todo = [i for i in sorted(idxs) if i not in done]
        skip = [i for i in sorted(idxs) if i in done]
        skipped_total += len(skip)
        log.append(f"  [{pfx}]\n")
        log.append(f"    To analyze : {todo}\n")
        if skip:
            log.append(
                f"    Skipped (resume): {skip}\n"
                if can_write else
                f"    Already in DB   : {skip}  "
                f"(read-only: will re-compute but not save)\n"
            )
    log.append(f"{'═'*80}\n")

    if can_write and skipped_total > 0:
        log.append(
            f"⚡ Resume: skipping {skipped_total} already-analyzed layers.\n"
        )

    # ── Layer-by-layer analysis ───────────────────────────────────
    sorted_items = sorted(filtered.items(), key=lambda x: (x[0][0], x[0][1]))
    total = len(sorted_items)

    for i, ((pfx, idx), prof) in enumerate(sorted_items):

        # Resume skip (only in write mode — reviewers always re-compute)
        if can_write and idx in done_layers.get(pfx, set()):
            continue

        progress(
            0.15 + 0.80 * i / max(total, 1),
            desc=f"{pfx.split('.')[-2] if '.' in pfx else pfx} L{idx}..."
        )

        # ── Load Q / K / V ────────────────────────────────────────
        try:
            q_url = get_file_url(model_id, prof.q.shard)
            k_url = get_file_url(model_id, prof.k.shard)
            q_hdr, q_hs = all_headers[prof.q.shard]
            k_hdr, k_hs = all_headers[prof.k.shard]

            dlog(log,
                f"Layer {idx}:\n"
                f"  q: {prof.q.shard} → {prof.q.key}\n"
                f"  k: {prof.k.shard} → {prof.k.key}\n"
                f"  v: {prof.v.shard + ' → ' + prof.v.key if prof.v else 'K=V shared'}\n"
            )

            W_q = load_tensor_remote(q_url, prof.q.key, q_hdr, q_hs, token)
            W_k = load_tensor_remote(k_url, prof.k.key, k_hdr, k_hs, token)

            if prof.kv_shared:
                W_v = W_k.clone()
            else:
                v_url = get_file_url(model_id, prof.v.shard)
                v_hdr, v_hs = all_headers[prof.v.shard]
                W_v = load_tensor_remote(v_url, prof.v.key, v_hdr, v_hs, token)

        except Exception as e:
            log.append(f"[{pfx}] Layer {idx}: ❌ Load failed: {e}\n")
            continue

        if W_q is None or W_k is None or W_v is None:
            log.append(f"[{pfx}] Layer {idx}: ⚠️ Tensor is None\n")
            continue

        # ── Compute Five Laws ─────────────────────────────────────
        try:
            records, layer_log = analyze_layer(W_q, W_k, W_v, prof)
            all_records.extend(records)
            log.append(layer_log)

            # ── Write to DB (admin only) ──────────────────────────
            if can_write and records:
                write_layer_records(conn, model_id, records)
                update_model_summary(conn, model_id, pfx)
                log.append(
                    f"  ✅ Saved to DB: {len(records)} records "
                    f"[{pfx}] Layer {idx}\n"
                )
            elif not can_write and records:
                log.append(
                    f"  📊 Computed: {len(records)} records "
                    f"[{pfx}] Layer {idx}  (read-only, not saved)\n"
                )

        except Exception as e:
            log.append(f"[{pfx}] Layer {idx}: ❌ Compute failed: {e}\n")
        finally:
            del W_q, W_k, W_v

    # ── Update elapsed time (admin only) ─────────────────────────
    if can_write:
        elapsed = (datetime.utcnow() - t_start).total_seconds()
        conn.execute(
            "UPDATE models SET analyze_sec = ? WHERE model_id = ?",
            (elapsed, model_id)
        )
        conn.commit()

    # ── Summary ───────────────────────────────────────────────────
    elapsed = (datetime.utcnow() - t_start).total_seconds()

    if not all_records:
        msg = (
            "\n⚡ All layers already in DB (resume mode). "
            "See Leaderboard or Database tab.\n"
            if can_write else
            "\n⚠️ No records computed.\n"
        )
        return "".join(log) + msg, None

    summary = summarize_records(all_records, model_id)
    log.append(summary)
    log.append(
        f"\n⏱️  Elapsed: {elapsed:.1f}s\n"
        f"{'═'*80}\n"
    )

    if not can_write:
        log.append(
            "🔒 Read-only mode: results above are NOT saved to the database.\n"
            "   To save, provide a valid Admin Write Token.\n"
        )

    df = pd.DataFrame(all_records)
    return "".join(log), df


# ─────────────────────────────────────────────
# Tab2 UI
# ─────────────────────────────────────────────

def build_tab_analyze():
    with gr.Tab("📊 Analyze"):
        gr.Markdown("""
        **Step 2: Select layer range and compute Wang's Five Laws metrics.**  
        Layer index = raw **N** in `layers.{N}` of safetensors keys.  
        K=V shared layers (e.g. Gemma-4 global layers) are handled automatically.  
        ⚡ **Resume supported**: already-analyzed layers are skipped automatically.

        > 第二步：选择层范围，计算王氏五定律全指标。支持断点续传，已分析层自动跳过。
        """)

        with gr.Row():
            with gr.Column(scale=3):
                model_id_input = gr.Textbox(
                    label="HuggingFace Model ID",
                    placeholder="google/gemma-4-e2b",
                    value="google/gemma-4-e2b"
                )
                token_input = gr.Textbox(
                    label="HF Access Token (leave empty for public models)",
                    type="password"
                )
                with gr.Row():
                    start_input = gr.Number(
                        label="Start Layer (inclusive)",
                        value=0, minimum=0, maximum=9999, precision=0
                    )
                    end_input = gr.Number(
                        label="End Layer (inclusive)",
                        value=5, minimum=0, maximum=9999, precision=0
                    )
                admin_token_input = gr.Textbox(
                    label="Admin Write Token",
                    placeholder="Leave empty to run analysis without saving to database",
                    type="password",
                    info=(
                        "Reviewers: leave empty. "
                        "Analysis runs fully — results shown below but not saved to DB. "
                        "| 审稿人请留空，分析正常运行，结果不写入数据库。"
                    )
                )
                analyze_btn = gr.Button("🚀 Start Analysis", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown(SIDEBAR_MD)

        analyze_log = gr.Textbox(
            label="Analysis Log (per-head details)",
            lines=35, max_lines=300
        )
        analyze_table = gr.Dataframe(
            label="Per-head metrics (all Five Laws)",
            headers=[
                "prefix", "layer", "kv_head", "q_head", "kv_shared",
                "pearson_QK", "spearman_QK", "pearson_QV", "pearson_KV",
                "ssr_QK", "ssr_QV", "ssr_KV",
                "cosU_QK", "cosU_QV", "cosU_KV",
                "cosV_QK", "cosV_QV", "cosV_KV",
                "alpha_QK", "alpha_QV", "alpha_KV",
                "alpha_res_QK", "alpha_res_QV", "alpha_res_KV",
                "sigma_max_Q", "sigma_min_Q",
                "sigma_max_K", "sigma_min_K",
                "sigma_max_V", "sigma_min_V",
                "cond_Q", "cond_K", "cond_V",
                "head_dim", "d_model", "n_q_heads", "n_kv_heads",
            ]
        )

        analyze_btn.click(
            fn=run_analysis,
            inputs=[
                model_id_input,
                token_input,
                start_input,
                end_input,
                admin_token_input,   # ← 新增
            ],
            outputs=[analyze_log, analyze_table]
        )

    return model_id_input, token_input