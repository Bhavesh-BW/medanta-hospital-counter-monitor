"""CLI entrypoint for labling_helper.
Usage:
    python -m labling_helper
    python -m labling_helper --input backend/training_samples --output labbled_images
"""

import argparse
import logging
import sys

from .auto_labeler import auto_label_dataset
from .config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Auto-labeling tool for AnyLabeling (Medanta Hospital Counter Monitor)."
    )
    parser.add_argument(
        "--input",
        "-i",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Path to input sample images folder (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Path to output labeled dataset folder (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--overwrite-all",
        action="store_true",
        help="Re-label all images even if JSON already exists (default: False)",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Only write JSON files, do not copy images into output directory",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print(" Medanta Counter Monitor — Auto-Labeling Helper")
    print("=" * 60)
    print(f" Input folder:  {args.input}")
    print(f" Output folder: {args.output}")
    print(f" Skip existing: {not args.overwrite_all}")
    print("=" * 60)

    summary = auto_label_dataset(
        input_dir=args.input,
        output_dir=args.output,
        skip_existing=not args.overwrite_all,
        copy_images=not args.no_copy_images,
    )

    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f" Total images:    {summary['total_images']}")
    print(f" Skipped existing: {summary['skipped_existing']} (preserved human labels)")
    print(f" Newly labeled:   {summary['newly_labeled']}")
    print(f" Errors:          {summary['errors']}")
    print(f" Output directory: {summary['output_dir']}")
    print("=" * 60)
    print("Open AnyLabeling and select 'Open Dir' -> 'labbled_images' to review.")


if __name__ == "__main__":
    main()
