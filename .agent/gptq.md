# Block-GPTQ with Joint Codebook+Scale Optimization

## Context

We now have `fakequant_blocks_with_scale()` which jointly optimizes codebook entry (k ∈ {1..364}) and per-block FP8 scale factor to minimize real-value MSE. This gives **17-18% MSE reduction** over fixed-scale codebook selection on random blocks.

The next step: integrate this into GPTQ so that quantization errors in block k get compensated in blocks k+1, k+2, ... via the Hessian inverse.

## Key Design Decision: Scale Computed On-The-Fly

Unlike FP-Quant (which pre-computes all scales from original weights, then locks them), our approach computes the optimal (codebook, scale) pair per block on the already-compensated weights:

```
FP-Quant:  pre-compute all scales → lock → GPTQ with stale scales
Ours:      GPTQ compensates weights → compute (codebook, scale) on current weights → propagate error
```

This is both simpler (no scale pre-computation, no locking hacks) and better (scale adapts to error-compensated distribution). Matters especially for codebook because the scale determines which 4 FP4 values survive — a small scale change can completely flip the codebook selection.

## Algorithm

```
Input: W [out_features, in_features] — BF16 reference weights
       H [in_features, in_features] — Hessian from calibration data

1. H_inv_cho = cholesky(inverse(H + λI))
2. for c in range(0, in_features, 16):             # block_size = 16 = codebook group size
     w_block = W[:, c:c+16]                        # already error-compensated from prior blocks

     # Joint optimization on compensated weights
     q_fp4, s = fakequant_blocks_with_scale(w_block, H_diag=diag(H_inv_cho)[c:c+16])
     w_q = q_fp4 * s                               # dequantized

     # Column-by-column error propagation to all subsequent columns
     for j in range(16):
       err_j = (w_block[:, j] - w_q[:, j]) / H_inv_cho[c+j, c+j]
       W[:, c+j+1:] -= err_j.outer(H_inv_cho[c+j, c+j+1:])

Output: q_fp4 per block, s per block (FP8), codebook index per block
```

All 16 columns in a block are quantized atomically (they share one codebook entry + scale).
Column-by-column error propagation still applies — each of the 16 error columns propagates independently to all future columns.

## Reference Implementation: FP-Quant

`/home/huanchen/FP-Quant/src/quantization/gptq.py` — reuse infrastructure, replace quantization step:

| Component | Source | Action |
|---|---|---|
| `GPTQ.update()` — Hessian accumulation | `gptq.py:68-102` | Copy verbatim |
| `_get_hessian_inverse()` — damping + Cholesky | `gptq.py:199-218` | Copy verbatim |
| `accumulate_hessian.py` — Triton kernel | entire file | Copy (fallback: `H += X.T @ X`) |
| Calibration loop — hooks, activation updates | `gptq.py:225-470` | Adapt for MoE |
| `GPTQ.step()` inner loop | `gptq.py:159-187` | Replace: block-16 with `fakequant_blocks_with_scale` |
| `Quantizer` class | `quantizer.py` | Not needed (we have our own) |

## Implementation Phases

### Phase 1: Calibration Infrastructure

Collect per-layer Hessian matrices from calibration data.

**Calibration data**: 128 sequences × 2048 tokens from C4 or RedPajama.

**MoE expert handling**: hook each expert's linear layers separately. Only tokens routed to expert e contribute to its Hessian. Low-sample experts get stronger damping.

**Files**:
```
NEW  calibrate.py     — layer-by-layer Hessian collection, save to disk
```

### Phase 2: Block-GPTQ Loop

The main implementation. Process each layer's weight matrix with block_size=16.

**Modifications to `fakequant_blocks_with_scale`**:
- Accept optional `hessian_diag` [16] per block
- Weight the MSE: `(H_diag * (w - q*s)²).sum() / H_diag.sum()` instead of `mean()`
- Makes within-block codebook selection importance-aware

**GPTQ inner loop** (replaces FP-Quant's per-column loop):
```python
for c in range(0, d_col, 16):
    w_block = W[:, c:c+16]                             # compensated weights
    h_diag = H_inv_cho.diag()[c:c+16]

    q_fp4, s_opt = quantizer.fakequant_blocks_with_scale(
        w_block.reshape(-1, 16),
        hessian_diag=h_diag.expand(out_features, -1).reshape(-1, 16),
    )
    w_q = (q_fp4 * s_opt).reshape_as(w_block)

    for j in range(16):
        err = (w_block[:, j] - w_q[:, j]) / H_inv_cho[c+j, c+j]
        W[:, c+j+1:].addr_(err, H_inv_cho[c+j, c+j+1:], alpha=-1)
```

**Files**:
```
NEW   gptq_codebook.py      — CodebookGPTQ class with block-16 inner loop
EDIT  fakequant.py           — add hessian_diag param to fakequant_blocks_with_scale
EDIT  fakequant_model.py     — --mode=gptq flag, load Hessians, run CodebookGPTQ per layer
```

### Phase 3: Iterative Refinement

After the initial GPTQ pass, re-visit each block and re-optimize given the final state of all other blocks. Wrap the Phase 2 block loop in an outer iteration (T=3-10 rounds).

**Expected gain**: Small (5-15% of remaining gap). Only pursue if Phase 2 leaves a noticeable quality gap.

```
EDIT  gptq_codebook.py      — outer iteration loop
```

## Reference Weights: BF16 vs NVFP4

With joint scale optimization, `--ref=bf16` is the natural choice:

| Mode | Reference | Hessian from | Scale | Best for |
|---|---|---|---|---|
| `--ref=bf16` | Original BF16 model | BF16 forward pass | Optimized per block | Maximum quality recovery |
| `--ref=fp4` | NVFP4-dequantized | NVFP4 forward pass | Fixed (NVFP4's) | Quick baseline, no BF16 model needed |

`--ref=bf16` is strictly more powerful — optimizes (codebook, scale) to approximate full-precision weights directly. Implement `--ref=fp4` first (simpler), then add `--ref=bf16`.

## Memory Budget (Qwen3-30B-A3B)

- Full Hessian per layer: up to 3584² × 4 bytes ≈ 49 MB — fits in GPU memory
- Expert Hessians: smaller in_features → cheaper
- All Hessians on disk: ~200 layers × 50 MB ≈ 10 GB total
- Peak GPU: one layer's weights + Hessian + Cholesky workspace

## Execution Order

1. **Phase 1**: `calibrate.py` — Hessian collection
2. **Phase 2**: `gptq_codebook.py` + hessian_diag in `fakequant.py` — main GPTQ loop
3. **Benchmark** on eval suite (GSM8K, GPQA Diamond, MMLU, MMMU)
4. **Phase 3**: only if Phase 2 gap > 0.5pt on key benchmarks
5. **`--ref=bf16`**: add after `--ref=fp4` is validated

## File Changes Summary

```
Phase 1:
  NEW   calibrate.py              — Hessian collection per layer, save to disk

Phase 2:
  NEW   gptq_codebook.py          — CodebookGPTQ with block-16 + joint scale opt
  EDIT  fakequant.py              — hessian_diag param in fakequant_blocks_with_scale
  EDIT  fakequant_model.py        — --mode=gptq flag

Phase 3:
  EDIT  gptq_codebook.py          — outer iteration loop
```

## Open Questions

- [ ] Calibration dataset: C4 vs RedPajama
- [ ] How to load NVFP4 model for calibration — `nvidia-modelopt` or manual dequant?
- [ ] Per-expert Hessians: sufficient samples for all experts?
- [ ] FP32 vs FP64 Hessian accumulation precision
- [ ] Interaction between GPTQ permutation order and codebook block boundaries
- [ ] For `--ref=bf16`: access to `Qwen/Qwen3-30B-A3B` base model on HF
