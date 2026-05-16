# ui/tab_inspect.py
"""
Tab1: Model Structure Inspection
- Read all shard headers
- Display raw key structure
- Auto-build LayerProfile and display inferred results
"""

import gradio as gr
import requests
import pandas as pd

from core.fetcher import (
    load_all_shard_headers,
    get_file_url,
    check_quantization,
    http_error_msg,
)
from core.layer_profile import (
    scan_model_structure,
    summarize_structure,
    extract_config_params,
)

SIDEBAR_MD = """
### ✅ Recommended Models
google/gemma-4-e2b  
google/gemma-4-e4b-it  
google/gemma-4-31b-it  
Qwen/Qwen2.5-14B-Instruct  
deepseek-ai/DeepSeek-R1-Distill-Qwen-14B  
meta-llama/Meta-Llama-3-8B （Need access right）   

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
"""


def inspect_model(
    model_id: str,
    hf_token: str,
    progress=gr.Progress()
) -> tuple[str, pd.DataFrame]:
    """
    Returns (inspection log text, layer structure DataFrame)
    """
    if not model_id.strip():
        return "❌ Please enter a model ID.", None

    token = hf_token.strip() or None
    log   = [f"🔬 Structure Inspection: {model_id}\n{'═'*80}\n"]

    # ── Quantization check ────────────────────────────────────────────────────
    progress(0.05, desc="Checking quantization...")
    blocked, qmsg = check_quantization(model_id, token)
    log.append(f"[Quantization Check]\n{qmsg}\n{'─'*80}\n")
    if blocked:
        return "".join(log), None

    # ── config.json ───────────────────────────────────────────────────────────
    progress(0.10, desc="Reading config...")
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
                f"📋 Config:\n"
                f"   model_type = {config_params.get('model_type')}\n"
                f"   hidden     = {config_params.get('hidden_size')}\n"
                f"   n_heads    = {config_params.get('num_attention_heads')}\n"
                f"   n_kv       = {config_params.get('num_key_value_heads')}\n"
                f"   head_dim   = {config_params.get('head_dim')}\n"
                f"{'─'*80}\n"
            )
    except Exception as e:
        log.append(f"⚠️  Could not read config.json: {e}\n")

    # ── Load all shard headers ─────────────────────────────────────────────────
    progress(0.20, desc="Loading shard headers...")
    try:
        all_headers = load_all_shard_headers(model_id, token)
    except requests.exceptions.HTTPError as e:
        return http_error_msg(e, model_id), None
    except Exception as e:
        return "".join(log) + f"❌ Failed to load headers: {e}\n", None

    total_keys = sum(len(h) for h, _ in all_headers.values())
    log.append(
        f"📦 Shards: {len(all_headers)}  "
        f"Total keys: {total_keys}\n"
        f"{'─'*80}\n"
    )

    # ── Scan layer structure ───────────────────────────────────────────────────
    progress(0.50, desc="Scanning layer structure...")
    profiles = scan_model_structure(all_headers, config_params)

    if not profiles:
        sample = []
        for h, _ in list(all_headers.values())[:1]:
            sample = list(h.keys())[:30]
        return (
            "".join(log) +
            "⚠️  No Q/K/V layers found. First 30 keys:\n" +
            "\n".join(sample), None
        )

    # ── Generate structure text ────────────────────────────────────────────────
    progress(0.80, desc="Generating report...")
    struct_text = summarize_structure(profiles)
    log.append(struct_text)

    # ── Build overview DataFrame ───────────────────────────────────────────────
    rows = []
    for (prefix, layer_idx), p in sorted(profiles.items()):
        rows.append({
            "prefix":     prefix,
            "layer":      layer_idx,
            "d_model":    p.d_model,
            "head_dim":   p.head_dim,
            "dim_source": p.head_dim_source,
            "n_q":        p.n_q_heads,
            "n_kv":       p.n_kv_heads,
            "kv_shared":  p.kv_shared,
            "complete":   p.complete,
            "q_shape":    str(p.q.shape) if p.q else "",
            "k_shape":    str(p.k.shape) if p.k else "",
            "v_shape":    str(p.v.shape) if p.v else "K=V",
        })

    df = pd.DataFrame(rows)

    progress(1.0, desc="Done")
    return "".join(log), df


# ─────────────────────────────────────────────
# Tab1 UI
# ─────────────────────────────────────────────

def build_tab_inspect():
    with gr.Tab("🔬 Inspect"):
        gr.Markdown("""
        **Step 1: Inspect model structure** — auto-detect components, head_dim, and K=V shared layers.
        Results are used by the **Analyze** tab.

        > No weights are downloaded — structure is inferred from safetensors headers only.
        """)

        with gr.Row():
            with gr.Column(scale=3):
                inspect_model_id = gr.Textbox(
                    label="HuggingFace Model ID",
                    placeholder="google/gemma-4-e2b",
                    value="google/gemma-4-e2b"
                )
                inspect_token = gr.Textbox(
                    label="HF Access Token (leave empty for public models)",
                    type="password"
                )
                inspect_btn = gr.Button("🔍 Inspect Structure", variant="secondary")

            with gr.Column(scale=1):
                gr.Markdown(SIDEBAR_MD)

        inspect_log = gr.Textbox(
            label="Inspection Log",
            lines=30, max_lines=200
        )
        inspect_table = gr.Dataframe(
            label="Layer Structure Overview",
            headers=[
                "prefix", "layer", "d_model", "head_dim", "dim_source",
                "n_q", "n_kv", "kv_shared", "complete",
                "q_shape", "k_shape", "v_shape"
            ]
        )

        inspect_btn.click(
            fn=inspect_model,
            inputs=[inspect_model_id, inspect_token],
            outputs=[inspect_log, inspect_table]
        )

    return inspect_model_id, inspect_token