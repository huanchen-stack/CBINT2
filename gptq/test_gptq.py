import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from fakequant import CodebookQuantizer
from gptq import CodebookGPTQ

torch.manual_seed(42)

OUT, IN = 256, 512
NUM_SAMPLES = 1024

W = torch.randn(OUT, IN) * 0.3
X = torch.randn(NUM_SAMPLES, IN) * 0.5

q = CodebookQuantizer()

print(f"=== GPTQ test: linear layer ({OUT}x{IN}), {NUM_SAMPLES} calibration samples ===")
print(f"  W range: [{W.min().item():.4f}, {W.max().item():.4f}]")
print(f"  X range: [{X.min().item():.4f}, {X.max().item():.4f}]")

Y_ref = X @ W.T

print("\n--- Plain codebook (no GPTQ) ---")
blocks_plain = W.reshape(-1, 16)
fp4_plain, scale_plain = q.fakequant_blocks_with_scale(blocks_plain)
W_q_plain = (fp4_plain * scale_plain).reshape(OUT, IN)
Y_plain = X @ W_q_plain.T

raw_mse_plain = ((W - W_q_plain) ** 2).mean().item()
output_mse_plain = ((Y_ref - Y_plain) ** 2).mean().item()
print(f"  weight MSE:  {raw_mse_plain:.6f}")
print(f"  output MSE:  {output_mse_plain:.6f}")

print("\n--- GPTQ codebook ---")
gptq = CodebookGPTQ(in_features=IN, quantizer=q)
gptq.update(X)
fp4_gptq, scale_gptq, W_q_gptq, fp4_phase1 = gptq.quantize(W)
Y_gptq = X @ W_q_gptq.T

raw_mse_gptq = ((W - W_q_gptq) ** 2).mean().item()
output_mse_gptq = ((Y_ref - Y_gptq) ** 2).mean().item()
print(f"  weight MSE:  {raw_mse_gptq:.6f}")
print(f"  output MSE:  {output_mse_gptq:.6f}")

print("\n--- Comparison ---")
weight_ratio = raw_mse_gptq / raw_mse_plain
output_ratio = output_mse_gptq / output_mse_plain
print(f"  weight MSE ratio (GPTQ/plain): {weight_ratio:.4f}")
print(f"  output MSE ratio (GPTQ/plain): {output_ratio:.4f}")
print(f"  output MSE reduction:          {(1 - output_ratio) * 100:.1f}%")

blocks_gptq = fp4_gptq.reshape(-1, 16)
max_distinct = max(len(set(blocks_gptq[i].tolist())) for i in range(blocks_gptq.shape[0]))
print(f"  max distinct per block:        {max_distinct}")

assert max_distinct <= 4, f"GPTQ produced block with {max_distinct} distinct values"
assert output_mse_gptq < output_mse_plain, \
    f"GPTQ output MSE ({output_mse_gptq:.6f}) should be < plain ({output_mse_plain:.6f})"
assert output_mse_gptq < output_mse_plain * 0.80, \
    f"GPTQ output MSE reduction ({(1 - output_ratio) * 100:.1f}%) should be >= 20%"
assert not W_q_gptq.isnan().any(), "NaN in GPTQ output"

print("\n--- Intra-group column-by-column changes ---")
blocks_phase1 = fp4_plain.reshape(-1, 16)
blocks_phase2 = fp4_gptq.reshape(-1, 16)
total_elements = blocks_phase2.numel()
changed_elements = (blocks_phase1 != blocks_phase2).sum().item()
changed_blocks = ((blocks_phase1 != blocks_phase2).any(dim=1)).sum().item()
total_blocks = blocks_phase2.shape[0]
print(f"  all columns:")
print(f"    elements changed:  {changed_elements}/{total_elements} ({100 * changed_elements / total_elements:.1f}%)")
print(f"    blocks changed:    {changed_blocks}/{total_blocks} ({100 * changed_blocks / total_blocks:.1f}%)")

first16_plain = fp4_plain.reshape(OUT, IN)[:, :16]
first16_gptq = fp4_gptq[:, :16]
first16_total = first16_gptq.numel()
first16_changed = (first16_plain != first16_gptq).sum().item()
first16_per_col = [(first16_plain[:, j] != first16_gptq[:, j]).sum().item() for j in range(16)]
print(f"  first 16 columns (no inter-block compensation):")
print(f"    elements changed:  {first16_changed}/{first16_total} ({100 * first16_changed / first16_total:.1f}%)")
print(f"    per column:        {first16_per_col}")

print("\n--- Phase 1 vs Phase 2 (intra-group column-wise effect) ---")
total_el = fp4_gptq.numel()
changed_el = (fp4_phase1 != fp4_gptq).sum().item()
changed_per_group: list[int] = []
for g_start in range(0, IN, 16):
    g_end = min(g_start + 16, IN)
    diff = (fp4_phase1[:, g_start:g_end] != fp4_gptq[:, g_start:g_end]).sum().item()
    changed_per_group.append(diff)
group_size = OUT * 16
print(f"  total elements flipped:  {changed_el}/{total_el} ({100 * changed_el / total_el:.1f}%)")
print(f"  per 16-col group (first 8):  {[f'{c}/{group_size} ({100*c/group_size:.1f}%)' for c in changed_per_group[:8]]}")
print(f"  per 16-col group (last 8):   {[f'{c}/{group_size} ({100*c/group_size:.1f}%)' for c in changed_per_group[-8:]]}")
avg_changed = sum(changed_per_group) / len(changed_per_group)
print(f"  avg per group:               {avg_changed:.1f}/{group_size} ({100 * avg_changed / group_size:.1f}%)")

print("\n--- Codebook consistency ---")
cb = q.codebook
codebook_ok = True
for i in range(blocks_gptq.shape[0]):
    block_vals = set(blocks_gptq[i].tolist())
    found = False
    for j in range(cb.shape[0]):
        if block_vals <= set(cb[j].tolist()):
            found = True
            break
    if not found:
        print(f"  FAIL: block {i} values {block_vals} not in any codebook entry")
        codebook_ok = False
        break
if codebook_ok:
    print(f"  All {blocks_gptq.shape[0]} blocks use valid codebook entries")
assert codebook_ok, "GPTQ output contains blocks not matching any codebook entry"

print("\n--- Hessian-weighted MSE ---")
H = X.T @ X / NUM_SAMPLES
delta_plain = W - W_q_plain
delta_gptq = W - W_q_gptq
h_mse_plain = (delta_plain @ H @ delta_plain.T).trace().item() / OUT
h_mse_gptq = (delta_gptq @ H @ delta_gptq.T).trace().item() / OUT
print(f"  plain:  {h_mse_plain:.6f}")
print(f"  GPTQ:   {h_mse_gptq:.6f}")
print(f"  ratio:  {h_mse_gptq / h_mse_plain:.4f}")

assert h_mse_gptq < h_mse_plain, \
    f"GPTQ Hessian MSE ({h_mse_gptq:.6f}) should be < plain ({h_mse_plain:.6f})"

print("\nAll GPTQ tests passed.")
