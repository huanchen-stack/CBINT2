from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tensorrt_llm import LLM, SamplingParams  # pyright: ignore[reportMissingImports]
from tensorrt_llm.evaluate import (  # pyright: ignore[reportMissingImports]
    GSM8K,
    GPQADiamond,
    GPQAExtended,
    GPQAMain,
    LongBenchV1,
    MMMU,
)

TASK_REGISTRY: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "evaluator_cls": GSM8K,
        "display_name": "GSM8K",
        "default_max_tokens": 256,
    },
    "gpqa_diamond": {
        "evaluator_cls": GPQADiamond,
        "display_name": "GPQA Diamond",
        "default_max_tokens": 32768,
    },
    "gpqa_main": {
        "evaluator_cls": GPQAMain,
        "display_name": "GPQA Main",
        "default_max_tokens": 32768,
    },
    "gpqa_extended": {
        "evaluator_cls": GPQAExtended,
        "display_name": "GPQA Extended",
        "default_max_tokens": 32768,
    },
    "mmmu": {
        "evaluator_cls": MMMU,
        "display_name": "MMMU",
        "default_max_tokens": 512,
        "evaluator_kwargs": {"is_multimodal": True, "apply_chat_template": True},
    },
    "longbench_v1": {
        "evaluator_cls": LongBenchV1,
        "display_name": "LongBench V1",
        "default_max_tokens": 512,
    },
}

DEFAULT_TASKS = ["gsm8k"]


def run_eval(
    model_path: str,
    tasks: list[str],
    tokenizer: str | None = None,
    tp_size: int = 1,
    pp_size: int = 1,
    limit: int | None = None,
    apply_chat_template: bool = False,
    max_tokens: int | None = None,
) -> dict:
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
    )

    results = {}

    for task_key in tasks:
        task_info = TASK_REGISTRY[task_key]

        evaluator_kwargs: dict[str, Any] = {
            "num_samples": limit,
            "apply_chat_template": apply_chat_template,
        }
        # Task-specific overrides (e.g. MMMU forces is_multimodal=True)
        evaluator_kwargs.update(task_info.get("evaluator_kwargs", {}))

        evaluator = task_info["evaluator_cls"](**evaluator_kwargs)

        tokens = max_tokens or task_info["default_max_tokens"]
        sampling_params = SamplingParams(max_tokens=tokens, temperature=0)

        score = evaluator.evaluate(llm, sampling_params)
        results[task_key] = {
            "score": score,
            "display_name": task_info["display_name"],
        }

    llm.shutdown()
    return results


def print_results(results: dict, tasks: list[str]) -> None:
    header = f"{'Task':<25} {'Score':>10}"
    print(header)
    print("-" * len(header))
    for task_key in tasks:
        r = results.get(task_key, {})
        display = r.get("display_name", task_key)
        score = r.get("score")
        score_str = f"{score:.2f}" if score is not None else "N/A"
        print(f"{display:<25} {score_str:>10}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate TensorRT-LLM models on standard benchmarks"
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="HuggingFace model name, checkpoint path, or TRT engine directory")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer path (required when --model-path is an engine directory)")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        choices=list(TASK_REGISTRY.keys()))
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor parallelism size")
    parser.add_argument("--pp-size", type=int, default=1,
                        help="Pipeline parallelism size")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per task (default: full dataset)")
    parser.add_argument("--apply-chat-template", action="store_true",
                        help="Apply chat template to prompts")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override default max output tokens per task")
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()

    tasks = args.tasks or DEFAULT_TASKS

    results = run_eval(
        model_path=args.model_path,
        tasks=tasks,
        tokenizer=args.tokenizer,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        limit=args.limit,
        apply_chat_template=args.apply_chat_template,
        max_tokens=args.max_tokens,
    )

    print_results(results, tasks)

    if args.output_file is not None:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
