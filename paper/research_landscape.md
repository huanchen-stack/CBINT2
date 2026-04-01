# CBINT2 Research Landscape — Literature Survey (2026-03-31)

## TL;DR

**The gap is confirmed: no prior work combines sub-4-bit storage with NVFP4 dequantization.**

Two research threads exist in complete isolation:
- NVFP4 improvement papers (RaZeR, MR-GPTQ, FourOverSix, ARCQuant, BlockDialect, FAAR) — all 4-bit
- Sub-4-bit VQ/codebook papers (AQLM, QuIP#, QTIP, GPTVQ, VPTQ) — all dequant to FP16

CBINT2 is the bridge.

---

## 1. NVFP4-Specific Papers (All 4-bit, None Sub-4-bit)

### MR-GPTQ (ICLR 2026) — Egiazarian et al. (Yandex/ISTA/Red Hat/ETH)
- **arXiv:** 2509.23202 | **OpenReview:** zCBGe9AqJZ
- **What:** GPTQ + block-wise Hadamard rotation tailored for FP4 E2M1
- **Key findings:**
  - NVFP4's small group size (16) *provably neutralizes* SmoothQuant-style outlier mitigation
  - Group-level Hadamard rotation spreads MSE evenly (Lemma 1)
  - MSE-optimized scale search (not codebook redesign — grid is fixed)
  - **No sub-4-bit experiments.** Never asks whether machinery applies to W3/W2.
- **Speedup:** 3.6x layer-wise, 2.2x end-to-end on B200; 6x/4x on RTX5090
- **Models:** Llama-3.1-8B/70B, Qwen-3
- **CBINT2 differentiation:** MR-GPTQ treats the FP4 grid as fixed. We select subsets of it.

### BOF4 (ICLR 2026) — Blumenberg et al.
- **arXiv:** 2505.06653
- **What:** EM/Lloyd's algorithm for optimal 4-bit block-wise codebook
- **Key findings:**
  - Correctly optimizes centroid updates accounting for scale-dependent distortion
  - BOF4-S (signed absmax) frees one codebook slot → 15 free levels
  - Ablated constrained subsets — found {-1, 0, +1} must be preserved
  - **Framework is bit-width agnostic but never applied to sub-4-bit or NVFP4 constraint**
- **CBINT2 differentiation:** BOF4's EM could be extended to our constrained problem (select k from 16 NVFP4 values). They never do this.

### RaZeR (ICML 2026) — arXiv:2501.04052
- **What:** Remaps NVFP4's redundant negative zero to a per-block special value (±5)
- **Key quote motivating CBINT2:**
  > "Lookup-based formats such as Normal-Float and Student-Float store the lookup table in FP16, necessitating high-precision MAC operations with significant energy overhead. In contrast, the FP4-E2M1 format offers a favorable trade-off between model accuracy and hardware efficiency."
- **CBINT2 response:** We get both — sub-4-bit compression via codebook AND FP4-native tensor core compute.

### Other NVFP4 Papers
| Paper | arXiv | Key idea |
|---|---|---|
| ARCQuant | 2601.07475 | Augmented residual channels for NVFP4 error compensation |
| FAAR | 2603.22370 | Format-aware adaptive rounding for NVFP4 non-uniform grid |
| FourOverSix | 2512.02010 | Adaptive block scaling (MIT Han Lab) |
| BlockDialect | 2501.01144 | Per-block FP4 variant selection from a "DialectFP4" formatbook |
| Quartet/II | 2505.14669, 2601.22813 | Native FP4 pre-training (IST-DASLab) |

---

## 2. Sub-4-bit VQ/Codebook Papers (All Dequant to FP16)

| Paper | Venue | Method | bpw | Dequant | FP4 target? |
|---|---|---|---|---|---|
| AQLM | ICML 2024 | Multi-codebook additive VQ | 2-3 | FP16 | No |
| QuIP# | ICML 2024 | Hadamard + E8 lattice codebooks | 2-4 | FP16 | No |
| QTIP | NeurIPS 2024 | Trellis coded quantization | 1-4 | FP16 | No |
| GPTVQ | arXiv 2024 | Hessian-guided VQ | 2-4 | FP16 | No |
| VPTQ | arXiv 2024 | Second-order VQ optimization | 2-4 | FP16 | No |
| Leech Lattice VQ | arXiv 2026 | 24D Leech lattice (SOTA) | 2-4 | FP16 | No |
| SBVR | arXiv 2025 | Summation of BitVector Repr. | 2-3 | FP16 | No |
| PyramidVQ | arXiv 2024 | Pyramid VQ on sphere | 2-4 | FP16 | No |

**Confirmed: NONE target FP4/NVFP4 as dequantization format.**

---

## 3. Theoretical Foundations

### Rate-Distortion for Quantized MatMul
- **Ordentlich & Polyanskiy (2024-2026):** arXiv:2410.13780, 2601.17187
  - IT-optimal lower bounds for quantized matrix multiplication
  - Phase transition at R≈0.906 bits/entry
  - **Our codebook achieves some point on their curve; cite as theoretical floor**

- **WaterSIC (Mar 2026):** arXiv:2603.04956
  - Shows GPTQ can be arbitrarily far from IT optimal
  - Waterfilling-based rate allocation within 0.255 bits of IT limit
  - **Motivates codebook-based approach over pure GPTQ**

### Optimal Format Design
- **"Optimal Formats for Weight Quantisation" (Orr et al., Graphcore 2025):** arXiv:2505.12988
  - Frames format design as KL minimization under model-size constraint
  - **Most directly related** — optimizes the format itself, but unconstrained
  - We add the NVFP4 hardware constraint

- **NF4 (Dettmers 2023):**
  - Selects 16 optimal values from continuous normal distribution
  - **Direct predecessor** — our problem selects ≤16 values from NVFP4's 16 codewords

- **HIGGS (Malinovskii 2025):** arXiv:2411.17525
  - MSE-optimal grid on Hadamard-rotated weights beats NF4
  - **Key result:** grid selection matters more than bit-width

### Constrained Codebook Theory
- **any4 (Meta, ICML 2025):** arXiv:2507.04610
  - Learns arbitrary 16-value codebook per-tensor
  - **Closest prior work:** they learn 16 codewords freely; we constrain to NVFP4's 16
- **LO-BCQ (NVIDIA, TMLR 2025):** arXiv:2502.05376
  - Locally optimal grid per block for W4A4
  - Integer grid, not constrained to FP4 codewords

### Classical
- **Lloyd (1982) / Max (1960):** Optimal scalar quantization with free reconstruction points
- **ECVQ (Chou et al. 1989):** Entropy-constrained VQ — optimizes codebook jointly with variable-length codes
- **Optimal Quantization Using Scaled Codebook (NVIDIA/UC Merced, CVPR 2021):** Scaled codebook optimization for neural nets

---

## 4. LUT-GEMM / Kernel Prior Art

### FLUTE (EMNLP 2024) — **Most relevant kernel template**
- arXiv:2407.10960 | GitHub: HanGuo97/flute
- Does exactly `W_hat[i,j] = T[Q[i,j]]` — index → codebook lookup in SMEM
- Offline weight restructuring + vectorized LUT in shared memory
- **But outputs FP16, not NVFP4.** Our kernel would output E2M1 packed bytes instead.

### LUT-GEMM (ICLR 2024) — Foundational GPU LUT paper
- arXiv:2206.09557 | GitHub: naver-aics/lut-gemm
- Eliminates dequant by precomputing partial sums into LUTs
- 2.1x speedup on OPT-175B at 3-bit vs GPTQ

### LUT Tensor Core (ISCA 2025) — Hardware co-design
- arXiv:2408.06003
- Proposes hardware modification for native LUT tensor core support
- **Key finding:** Software LUT overhead is real on current hardware. Without hardware LUT support, table precompute storage dominates.
- **Implication for CBINT2:** Our 182-entry codebook (273 bytes) is small enough for registers, avoiding the overhead LUT Tensor Core identifies for larger codebooks.

### T-MAC (EuroSys 2025)
- arXiv:2407.00088 | GitHub: microsoft/T-MAC
- CPU/NPU-focused LUT-GEMM. Not applicable to Blackwell tensor cores.

### Blackwell FP4 Kernel Ecosystem
- **Colfax CUTLASS sub-byte GEMM tutorial:**
  - Blackwell's `f8f6f4` UMMA supports E2M1 natively
  - TMA unpacks packed 4-bit GMEM → padded SMEM
  - **Dynamic data types** via instruction descriptor (runtime)
  - Our dequant path: sub-4-bit GMEM → codebook lookup → E2M1 SMEM → UMMA

- **"Twelve Attempts at an FP4 Kernel" (Amandeep Singh, Feb 2026):**
  - `cvt.rn.f16x2.e2m1x2` = 1 PTX instruction for FP4→FP16 decode
  - Top hackathon solutions: raw PTX, cache policy differentiation, 256-bit vectorized loads
  - Best hand-written kernel ~2x off speed-of-light on B200

- **vLLM NVFP4 support:**
  - PR #18312: CUTLASS for Blackwell, Marlin emulation for older HW
  - Issue #31782: NVFP4 for MoE models
  - Issue #31085: SM120 (RTX 5000/6000) native NVFP4

---

## 5. The Genuine Novelty — Gap Map

| Prior Work | What They Do | What's Missing |
|---|---|---|
| NF4 (Dettmers 2023) | Selects 16 optimal values from R for normal weights | Doesn't constrain to NVFP4's 16 codewords |
| any4 (Elhoushi 2025) | Learns arbitrary 16-value codebook | No hardware constraint; dequant target is free |
| HIGGS (Malinovskii 2024) | MSE-optimal grid on rotated weights | Grid is free, not subset of NVFP4 |
| BOF4 (Blumenberg 2026) | Optimizes 4-bit float format | Designs format, not sub-format codebook |
| FourOverSix (Cook 2025) | Adaptive block scaling for FP4 | Heuristic, not principled subset selection |
| LO-BCQ (Elangovan 2025) | Locally optimal grid per block | Integer grid, not constrained to FP4 |
| QuIP#/QTIP/LLVQ | Lattice codebooks | High-dim VQ, not scalar sub-format selection |
| MR-GPTQ (Egiazarian 2026) | GPTQ+rotation for FP4 | Fixed grid, no sub-4-bit, no subset selection |
| FLUTE (Guo 2024) | LUT-GEMM for codebook quant | Outputs FP16, not NVFP4 |
| Ordentlich & Polyanskiy | IT lower bounds for MatMul quant | Bounds for unconstrained quantizers |

**The genuine novelty:** No paper addresses selecting an optimal k-element subset (k=2^N, N=1,2,3) from a fixed 16-element hardware format grid (NVFP4) to minimize rate-distortion, jointly optimizing (N, subset, per-block scale granularity). This is a **constrained subset selection problem on a non-uniform fixed grid** — a variant of Lloyd-Max where reconstruction points are constrained to a hardware-defined superset.

---

## 6. Key Paper Differentiation Statement (for Related Work section)

> All existing sub-4-bit methods (AQLM, VPTQ, QTIP, QuIP#, GPTVQ) dequantize to FP16, requiring CUDA-core dequantization and forfeiting Blackwell tensor core throughput. All existing NVFP4 methods (RaZeR, MR-GPTQ, FourOverSix, ARCQuant) operate at exactly 4 bits/weight. No prior work constrains codebook entries to FP4-representable values to enable sub-4-bit storage with hardware-native FP4 tensor core compute.

---

## 7. Recommended Experimental Design Space

The "optimal codebook" study that nobody has done:

| Dimension | Range | What to sweep |
|---|---|---|
| Index bits | 1, 2, 3 | How many bits per weight index |
| Codebook size | 2, 4, 8 values from 16 NVFP4 codewords | Which subset of E2M1 values |
| Codebook granularity | Per-tensor, per-column, per-group (g=128,64,32,16) | How often to switch codebook |
| Scale interaction | Shared with NVFP4 block scale vs. independent | Whether per-16-block E4M3 scale serves double duty |
| Subset selection | Uniform, MSE-optimal (Lloyd-Max on NVFP4 grid), learned | How to pick which FP4 values to keep |
| Rotation | None, Hadamard (MR-GPTQ found it *hurts* NVFP4 with RTN) | Whether to rotate before codebook quantization |
| Calibration | RTN, GPTQ (Hessian-weighted), WaterSIC-style | Rounding strategy |

### Baselines needed:
1. **NVFP4 native (4.5 bpw)** — upper bound on quality
2. **INT3 uniform + naive dequant to NVFP4** — the "obvious" baseline
3. **INT3 + per-group NVFP4 codebook** — strong baseline (C(14,7)=3432 possibilities, ~12 bits overhead)
4. **CBINT2 (INT2 + per-group codebook, 3.0 bpw)** — your current scheme
5. **any4 (learned 16-value codebook)** — unconstrained upper bound at 4-bit
6. **AQLM/QuIP# at 2-3 bpw** — sub-4-bit quality reference (different compute path)
