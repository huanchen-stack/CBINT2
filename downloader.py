import argparse
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="nvidia/Qwen3-30B-A3B-NVFP4")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or f"./models/{args.model_id.split('/')[-1]}"

    print(f"Downloading {args.model_id} to {output_dir}...")
    snapshot_download(repo_id=args.model_id, local_dir=output_dir)
    print(f"Done. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
