# Port of MiaAI-Lab's NVFP4 KV cache stack onto our fleet image
# (sglang-qwen38fn:sm121-qsa already carries the QSA guard + NCCL 2.30.7 pin).
FROM sglang-qwen38fn:sm121-qsa

COPY qsa_fa_fallback.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa_fa_fallback.py
COPY qsa_nvfp4_kv.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa_nvfp4_kv.py
COPY apply_nvfp4_patches.py /tmp/apply_nvfp4_patches.py

RUN python3 - <<'PYEOF'
p = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
s = open(p).read()
anchor = "    try:\n        from flash_attn import flash_attn_varlen_func"
patch = (
    "    from sglang.srt.utils import is_sm100_supported\n"
    "    if not is_sm100_supported():\n"
    "        from sglang.srt.layers.attention.qsa_fa_fallback import triton_varlen_attn_func\n"
    "        return triton_varlen_attn_func\n"
) + anchor
assert anchor in s, "anchor not found — image layout changed"
assert "qsa_fa_fallback" not in s, "already patched"
open(p, "w").write(s.replace(anchor, patch, 1))
print("qwen_sparse_attn_backend.py patched: Triton varlen fallback for SM121")
PYEOF

RUN python3 /tmp/apply_nvfp4_patches.py && rm -f /tmp/apply_nvfp4_patches.py \
    && find /sgl-workspace/sglang/python/sglang/srt -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
