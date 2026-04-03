from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


def _reshard_by_block(model_dir: Path) -> None:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        single = model_dir / "model.safetensors"
        if single.exists():
            index = {"metadata": {}, "weight_map": {}}
            tensors = load_file(str(single))
            for key in tensors:
                index["weight_map"][key] = "model.safetensors"
        else:
            print("No safetensors index found, skipping re-shard.")
            return
    else:
        with index_path.open() as f:
            index = json.load(f)

    weight_map: dict[str, str] = index["weight_map"]

    block_tensors: dict[int, dict[str, str]] = {}
    non_block_tensors: dict[str, str] = {}
    for tensor_name, shard_file in weight_map.items():
        block_idx = _extract_block_index(tensor_name)
        if block_idx is not None:
            block_tensors.setdefault(block_idx, {})[tensor_name] = shard_file
        else:
            non_block_tensors[tensor_name] = shard_file

    num_blocks = len(block_tensors)
    if num_blocks == 0:
        print("No transformer blocks found in weight map, skipping re-shard.")
        return

    already_sharded = all(
        shard_file == f"block_{block_idx:04d}.safetensors"
        for block_idx, tensors in block_tensors.items()
        for shard_file in tensors.values()
    )
    if already_sharded:
        print(f"Model already sharded by block ({num_blocks} blocks). Skipping.")
        return

    print(f"Re-sharding model into {num_blocks} block files + 1 non-block file...")

    original_shards: set[str] = set(weight_map.values())
    loaded_shards: dict[str, dict] = {}
    new_weight_map: dict[str, str] = {}

    for block_idx in sorted(block_tensors):
        block_file = f"block_{block_idx:04d}.safetensors"
        block_data: dict = {}
        for tensor_name, shard_file in block_tensors[block_idx].items():
            if shard_file not in loaded_shards:
                loaded_shards[shard_file] = load_file(str(model_dir / shard_file))
            block_data[tensor_name] = loaded_shards[shard_file][tensor_name]
            new_weight_map[tensor_name] = block_file
        save_file(block_data, str(model_dir / block_file))
        print(f"  {block_file}: {len(block_data)} tensors")
        del block_data

    if non_block_tensors:
        non_block_file = "non_block.safetensors"
        non_block_data: dict = {}
        for tensor_name, shard_file in non_block_tensors.items():
            if shard_file not in loaded_shards:
                loaded_shards[shard_file] = load_file(str(model_dir / shard_file))
            non_block_data[tensor_name] = loaded_shards[shard_file][tensor_name]
            new_weight_map[tensor_name] = non_block_file
        save_file(non_block_data, str(model_dir / non_block_file))
        print(f"  {non_block_file}: {len(non_block_data)} tensors")
        del non_block_data

    del loaded_shards

    new_index = {"metadata": index.get("metadata", {}), "weight_map": new_weight_map}
    with index_path.open("w") as f:
        json.dump(new_index, f, indent=2)
    print(f"  Updated {index_path.name}")

    new_files = {f"block_{i:04d}.safetensors" for i in block_tensors}
    if non_block_tensors:
        new_files.add("non_block.safetensors")
    for old_shard in original_shards:
        if old_shard not in new_files:
            old_path = model_dir / old_shard
            if old_path.exists():
                old_path.unlink()
                print(f"  Removed old shard: {old_shard}")

    print(f"Re-shard complete: {num_blocks} block files"
          + (f" + 1 non-block file" if non_block_tensors else ""))


def _extract_block_index(tensor_name: str) -> int | None:
    parts = tensor_name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="nvidia/Qwen3-30B-A3B-NVFP4")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-reshard", action="store_true",
                        help="Download only, do not re-shard by transformer block")
    args = parser.parse_args()

    output_dir = args.output_dir or f"./models/{args.model_id.split('/')[-1]}"
    output_path = Path(output_dir)

    print(f"Downloading {args.model_id} to {output_dir}...")
    snapshot_download(repo_id=args.model_id, local_dir=output_dir)
    print(f"Download complete.")

    if not args.skip_reshard:
        _reshard_by_block(output_path)

    print(f"Done. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
