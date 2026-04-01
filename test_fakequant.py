import argparse

from fakequant import CodebookQuantizer
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", action="store_true", help="Run speed benchmark")
args = parser.parse_args()

torch.manual_seed(42)
q = CodebookQuantizer()

print("=== Nibble-to-FP4 lookup table ===")
for i in range(16):
    print(f"  nibble {i:2d} (0b{i:04b}) -> {q.nibble_to_fp4[i].item():+.1f}")

print(f"\n=== Codebook: {q.codebook.shape[0]} entries (first 5) ===")
for i in range(5):
    vals = q.codebook[i].tolist()
    print(f"  entry {i}: {[f'{v:+.1f}' for v in vals]}")
print(f"  ...")

print("\n=== Single block fakequant ===")
block = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0]])
print(f"  original:   {block[0].tolist()}")
quantized, all_mse = q.fakequant_blocks(block, return_mse=True)
print(f"  quantized:  {quantized[0].tolist()}")
unique = sorted(set(quantized[0].tolist()))
print(f"  distinct values: {unique} (count: {len(unique)})")
print(f"  best MSE: {all_mse[0].min().item():.6f}")
top5_idx = all_mse[0].argsort()[:5]
print(f"  top-5 codebook entries (by MSE):")
for rank, idx in enumerate(top5_idx):
    entry = q.codebook[idx].tolist()
    print(f"    #{rank}: entry {idx.item()} {[f'{v:+.1f}' for v in entry]}  MSE={all_mse[0, idx].item():.6f}")

print("\n=== Random blocks (8 blocks of 16) ===")
random_blocks = q.fp4_representable[torch.randint(0, 15, (8, 16))]
quantized_random, random_mse = q.fakequant_blocks(random_blocks, return_mse=True)
for i in range(8):
    orig = [f"{v:+.1f}" for v in random_blocks[i].tolist()]
    qval = [f"{v:+.1f}" for v in quantized_random[i].tolist()]
    unique_q = sorted(set(quantized_random[i].tolist()))
    best = random_mse[i].min().item()
    print(f"  block {i}:")
    print(f"    orig: {orig}")
    print(f"    qnt:  {qval}")
    print(f"    unique: {[f'{v:+.1f}' for v in unique_q]} | MSE: {best:.4f}")

print("\n=== Pack/unpack roundtrip ===")
packed = q.pack_fp4_to_uint8(quantized_random)
unpacked = q.unpack_uint8_to_fp4(packed)
match = torch.equal(unpacked, quantized_random)
print(f"  roundtrip exact match: {match}")

print("\n=== NVFP4 from BF16 (single block) ===")
FP4_MAX = 6.0
bf16_block = torch.randn(1, 16, dtype=torch.bfloat16)

amax = bf16_block.float().abs().amax(dim=-1, keepdim=True)
scale_nvfp4 = (amax / FP4_MAX).to(torch.float8_e4m3fn).to(torch.float32).clamp(min=1e-10)

scaled = bf16_block.float() / scale_nvfp4
diffs = (scaled.unsqueeze(-1) - q.fp4_representable).abs()
fp4_block = q.fp4_representable[diffs.argmin(dim=-1)]

dequant_nvfp4 = fp4_block * scale_nvfp4
nvfp4_mse = ((bf16_block.float() - dequant_nvfp4) ** 2).mean().item()

print(f"  BF16 original:    {[f'{v:.4f}' for v in bf16_block[0].tolist()]}")
print(f"  scale:            {scale_nvfp4[0].item():.6f}")
print(f"  FP4 values:       {[f'{v:+.1f}' for v in fp4_block[0].tolist()]}")
print(f"  dequantized:      {[f'{v:.4f}' for v in dequant_nvfp4[0].tolist()]}")
print(f"  NVFP4 MSE (bf16): {nvfp4_mse:.6f}")

codebook_fp4 = q.fakequant_blocks(fp4_block)
codebook_dequant = codebook_fp4 * scale_nvfp4
codebook_mse = ((bf16_block.float() - codebook_dequant) ** 2).mean().item()

print(f"  codebook FP4:     {[f'{v:+.1f}' for v in codebook_fp4[0].tolist()]}")
print(f"  codebook dequant: {[f'{v:.4f}' for v in codebook_dequant[0].tolist()]}")
print(f"  codebook MSE:     {codebook_mse:.6f}")
print(f"  unique codebook:  {sorted(set(codebook_fp4[0].tolist()))}")
print(f"  MSE increase:     {codebook_mse / nvfp4_mse:.2f}x")

max_iters = 5
print(f"\n=== Scale optimization: single block (max_iters={max_iters}) ===")
opt_fp4, opt_scale = q.fakequant_blocks_with_scale(bf16_block.float(), max_iters=max_iters)
opt_dequant = opt_fp4 * opt_scale
opt_mse = ((bf16_block.float() - opt_dequant) ** 2).mean().item()

print(f"  joint-opt scale:  {opt_scale[0].item():.6f}  (was {scale_nvfp4[0].item():.6f})")
print(f"  joint-opt FP4:    {[f'{v:+.1f}' for v in opt_fp4[0].tolist()]}")
print(f"  joint-opt dequant:{[f'{v:.4f}' for v in opt_dequant[0].tolist()]}")
print(f"  joint-opt MSE:    {opt_mse:.6f}")
print(f"  vs fixed codebook: {(1 - opt_mse / codebook_mse) * 100:.1f}% MSE reduction")

print(f"\n=== Scale optimization: 1024 random blocks (max_iters={max_iters}) ===")
torch.manual_seed(123)
num_test_blocks = 1024
test_weights = torch.randn(num_test_blocks, 16)

amax_test = test_weights.abs().amax(dim=-1, keepdim=True)
scale_test = (amax_test / FP4_MAX).to(torch.float8_e4m3fn).to(torch.float32).clamp(min=1e-10)
fp4_indices_test = q._round_to_fp4_indices(test_weights / scale_test)
fp4_test = q.fp4_representable[fp4_indices_test]
dequant_test = fp4_test * scale_test
nvfp4_mse_per = ((test_weights - dequant_test) ** 2).mean(dim=-1)

codebook_fp4_test = q.fakequant_blocks(fp4_test)
fixed_dequant_test = codebook_fp4_test * scale_test
fixed_mse_per = ((test_weights - fixed_dequant_test) ** 2).mean(dim=-1)

opt_fp4_test, opt_scale_test = q.fakequant_blocks_with_scale(test_weights, max_iters=max_iters)
opt_dequant_test = opt_fp4_test * opt_scale_test
opt_mse_per = ((test_weights - opt_dequant_test) ** 2).mean(dim=-1)

avg_nvfp4 = nvfp4_mse_per.mean().item()
avg_fixed = fixed_mse_per.mean().item()
avg_opt = opt_mse_per.mean().item()
blocks_improved = (opt_mse_per < fixed_mse_per).sum().item()

print(f"  avg NVFP4 MSE:            {avg_nvfp4:.6f}")
print(f"  avg fixed-scale codebook: {avg_fixed:.6f}  ({avg_fixed / avg_nvfp4:.2f}x vs NVFP4)")
print(f"  avg joint-opt codebook:   {avg_opt:.6f}  ({avg_opt / avg_nvfp4:.2f}x vs NVFP4)")
print(f"  MSE reduction:            {(1 - avg_opt / avg_fixed) * 100:.1f}%")
print(f"  blocks improved:          {blocks_improved}/{num_test_blocks}")

print("\n=== Scale roundtrip: fakequant_layer dequant invariant ===")
torch.manual_seed(99)
out_f, in_f = 64, 768
bf16_ref = torch.randn(out_f, in_f)
FP8_MAX, FP4_MAX_VAL = 448.0, 6.0
gscale_rt = torch.tensor(FP4_MAX_VAL * FP8_MAX / bf16_ref.abs().max().item(), dtype=torch.float32)
weight_scale_2_rt = (1.0 / gscale_rt).reshape(())

raw_scale = bf16_ref.reshape(out_f, in_f // 16, 16).abs().amax(dim=-1) / FP4_MAX_VAL
weight_scale_rt = (raw_scale * gscale_rt).to(torch.float8_e4m3fn)

eff_scale = weight_scale_rt.to(torch.float32) * weight_scale_2_rt.to(torch.float32)
eff_expanded = eff_scale.repeat_interleave(16, dim=1)
fp4_idx = (bf16_ref / eff_expanded).unsqueeze(-1).sub(q.fp4_representable).abs().argmin(dim=-1)
fp4_orig = q.fp4_representable[fp4_idx]
packed_rt = q.pack_fp4_to_uint8(fp4_orig)

vanilla_packed = q._fakequant_layer_vanilla(packed_rt, weight_scale_rt, weight_scale_2_rt)
vanilla_fp4 = q.unpack_uint8_to_fp4(vanilla_packed)
vanilla_dequant = vanilla_fp4 * eff_expanded
vanilla_mse = ((bf16_ref - vanilla_dequant) ** 2).mean().item()

new_packed, new_ws = q.fakequant_layer(packed_rt, weight_scale_rt, weight_scale_2_rt)
new_fp4 = q.unpack_uint8_to_fp4(new_packed)
new_eff = new_ws.to(torch.float32) * weight_scale_2_rt.to(torch.float32)
new_eff_expanded = new_eff.repeat_interleave(16, dim=1)
dequant_new = new_fp4 * new_eff_expanded

nvfp4_mse_rt = ((bf16_ref - fp4_orig * eff_expanded) ** 2).mean().item()
new_mse = ((bf16_ref - dequant_new) ** 2).mean().item()

print(f"  weight_scale_2:      {weight_scale_2_rt.item():.6f}  (reciprocal convention)")
print(f"  weight_scale dtype:  {weight_scale_rt.dtype}")
print(f"  new_scale dtype:     {new_ws.dtype}")
assert new_ws.dtype == weight_scale_rt.dtype, f"Scale dtype changed: {weight_scale_rt.dtype} -> {new_ws.dtype}"
print(f"  NVFP4 MSE:           {nvfp4_mse_rt:.6f}")
print(f"  vanilla codebook:    {vanilla_mse:.6f}  ({vanilla_mse / nvfp4_mse_rt:.2f}x vs NVFP4)")
print(f"  scale-opt codebook:  {new_mse:.6f}  ({new_mse / nvfp4_mse_rt:.2f}x vs NVFP4)")
assert new_mse <= vanilla_mse * 1.01, f"Scale-opt worse than vanilla: {new_mse:.6f} > {vanilla_mse:.6f}"
print(f"  vs vanilla:          {(1 - new_mse / vanilla_mse) * 100:.1f}% MSE reduction")
print(f"  PASS: scale roundtrip correct, dequant = fp4 * weight_scale * weight_scale_2")

print("\n=== Full layer fakequant vanilla (32x512) ===")
layer_fp4 = q.fp4_representable[torch.randint(0, 15, (32, 512))]
layer_packed = q.pack_fp4_to_uint8(layer_fp4)
scale = torch.ones((32, 32), dtype=torch.float32)
gscale = torch.ones((1,), dtype=torch.float32)

quantized_packed = q._fakequant_layer_vanilla(layer_packed, scale, gscale)
quantized_layer = q.unpack_uint8_to_fp4(quantized_packed)

layer_mse = ((layer_fp4 - quantized_layer) ** 2).mean().item()
blocks_q = quantized_layer.reshape(-1, 16)
max_distinct = max(len(set(blocks_q[i].tolist())) for i in range(blocks_q.shape[0]))

print(f"  shape: {layer_fp4.shape} -> {quantized_layer.shape}")
print(f"  overall MSE: {layer_mse:.6f}")
print(f"  max distinct values per block: {max_distinct}")
print(f"  first block orig:  {[f'{v:+.1f}' for v in layer_fp4[0, :16].tolist()]}")
print(f"  first block quant: {[f'{v:+.1f}' for v in quantized_layer[0, :16].tolist()]}")

if args.benchmark:
    import time
    bench_out, bench_in = 2048, 768
    num_tensors = 128
    blocks_per_tensor = (bench_out * bench_in) // 16
    total_blocks = blocks_per_tensor * num_tensors

    layers = []
    for _ in range(num_tensors):
        fp4 = q.fp4_representable[torch.randint(0, 15, (bench_out, bench_in))]
        packed = q.pack_fp4_to_uint8(fp4)
        s = torch.rand(bench_out, bench_in // 16).to(torch.float8_e4m3fn).to(torch.float32).clamp(min=0.01)
        g = torch.tensor([2.0], dtype=torch.float32)
        layers.append((packed, s, g))

    device = "cuda"
    layers_d = [(p.to(device), s.to(device), g.to(device)) for p, s, g in layers]

    print(f"\n=== Benchmark vanilla: {num_tensors} x ({bench_out}x{bench_in}) ===")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for p, s, g in layers_d:
        _ = q._fakequant_layer_vanilla(p, s, g)
    torch.cuda.synchronize()
    elapsed_vanilla = time.perf_counter() - t0
    print(f"  {elapsed_vanilla:.2f}s  ({total_blocks} total blocks, {total_blocks / elapsed_vanilla:.0f} blocks/s)")

    print(f"\n=== Benchmark scale-opt (max_iters={max_iters}): {num_tensors} x ({bench_out}x{bench_in}) ===")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for p, s, g in layers_d:
        _, _ = q.fakequant_layer(p, s, g)
    torch.cuda.synchronize()
    elapsed_opt = time.perf_counter() - t0
    print(f"  {elapsed_opt:.2f}s  ({total_blocks} total blocks, {total_blocks / elapsed_opt:.0f} blocks/s)")
    print(f"  slowdown vs vanilla: {elapsed_opt / elapsed_vanilla:.1f}x")
