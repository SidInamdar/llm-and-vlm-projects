"""CLI entrypoint for downloading datasets used by the llama-sft project."""

import argparse
import sys
from pathlib import Path

from shared.clm_dataset_download import download_dataset


def find_repo_root() -> Path:
    """
    Find the repository root by searching upward for pyproject.toml.

    Returns:
        Path to the repository root directory.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def main() -> None:
    """Parse arguments and download the requested datasets."""
    parser = argparse.ArgumentParser(
        description="Download datasets for SFT training.",
    )
    parser.add_argument(
        "--dataset",
        choices=["bitext", "multiwoz", "all"],
        default="all",
        help="Name of the dataset to download (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where datasets are saved (default: datasets/raw/ in repo root)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional HuggingFace cache directory",
    )

    args = parser.parse_args()

    repo_root = find_repo_root()
    output_dir = args.output_dir or str(repo_root / "datasets" / "raw")

    print(f"Repository root : {repo_root}")
    print(f"Output directory: {output_dir}")

    names = ["bitext", "multiwoz"] if args.dataset == "all" else [args.dataset]

    for name in names:
        print(f"\n--- Downloading: {name} ---")
        try:
            download_dataset(name, output_dir, cache_dir=args.cache_dir)
        except Exception as e:
            print(f"Error downloading {name}: {e}", file=sys.stderr)
            sys.exit(1)

    print("\nAll downloads finished successfully!")


if __name__ == "__main__":
    main()
