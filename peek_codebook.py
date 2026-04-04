from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help=".pt file or directory of .pt files")
    parser.add_argument("--top", type=int, default=32, help="Show top-N codebooks")
    parser.add_argument("--all", action="store_true", help="Print all codebooks")
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_dir():
        files = sorted(path.glob("*.pt"))
        if not files:
            raise SystemExit(f"No .pt files in {path}")
        for f in files:
            print(f"\n{'=' * 60}")
            _peek(f, args.top, args.all)
    elif path.is_file():
        _peek(path, args.top, args.all)
    else:
        raise SystemExit(f"Not found: {path}")


def _peek(path: Path, top: int, show_all: bool) -> None:
    t = torch.load(path, map_location="cpu", weights_only=True)
    print(f"{path.name}  shape={list(t.shape)}  dtype={t.dtype}")

    if t.dim() != 2:
        print(f"  (not a 2D codebook tensor, skipping)")
        return

    n, k = t.shape
    limit = n if show_all else min(top, n)

    print(f"  {n} codebooks, {k} values each")
    for i in range(limit):
        vals = t[i].tolist()
        formatted = ", ".join(f"{v:>6.2f}" for v in vals)
        print(f"  [{i:>3}] {formatted}")

    if not show_all and n > limit:
        print(f"  ... ({n - limit} more, use --all to show)")


if __name__ == "__main__":
    main()
