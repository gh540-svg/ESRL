"""Download selected Qwen3 backbones and benchmark splits for offline jobs."""
from __future__ import annotations

import os
import argparse


MODELS = {
    "0p6b": "Qwen/Qwen3-0.6B-Base",
    "4b": "Qwen/Qwen3-4B-Base",
    "8b": "Qwen/Qwen3-8B-Base",
}
DATASETS = (
    ("HuggingFaceH4/MATH-500", None, ("test",)),
    ("cais/mmlu", "all", ("test",)),
    ("mkieffer/MedQA-USMLE", None, ("train", "test")),
    ("rawsh/amc23", None, ("train",)),
    ("yentinglin/aime_2025", None, ("train",)),
    ("allenai/ai2_arc", "ARC-Challenge", ("test",)),
    ("Idavidrein/gpqa", "gpqa_diamond", ("train",)),
    ("openlifescienceai/medmcqa", None, ("validation",)),
    ("qiaojin/PubMedQA", "pqa_labeled", ("train",)),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_sets",
        nargs="*",
        default=("4b",),
        choices=("0p6b", "4b", "8b", "all"),
        help="one or more model sizes; 'all' selects every configured backbone",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="download model snapshots without revisiting benchmark datasets",
    )
    args = parser.parse_args()
    offline_flags = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
    enabled = [name for name in offline_flags if os.environ.get(name) == "1"]
    if enabled:
        raise SystemExit(
            "Caching requires network access. Unset: " + ", ".join(enabled)
        )

    from huggingface_hub import snapshot_download

    if not args.models_only:
        from datasets import load_dataset
        import transformers

        major_minor = tuple(
            int(part) for part in transformers.__version__.split(".")[:2]
        )
        if major_minor < (4, 51):
            raise SystemExit(
                f"transformers {transformers.__version__} is too old for Qwen3; "
                "install transformers>=4.51,<5 first"
            )

    print(f"HF_HOME={os.environ.get('HF_HOME', '(default)')}", flush=True)
    if "all" in args.model_sets:
        selected_keys = tuple(MODELS)
    else:
        selected_keys = tuple(dict.fromkeys(args.model_sets))
    selected_models = (MODELS[key] for key in selected_keys)
    for model in selected_models:
        print(f"[cache] model {model}", flush=True)
        snapshot_download(model)

    if not args.models_only:
        for dataset, config, splits in DATASETS:
            for split in splits:
                print(
                    f"[cache] dataset {dataset} {config or ''} split={split}",
                    flush=True,
                )
                try:
                    load_dataset(dataset, config, split=split)
                except Exception as error:
                    if dataset == "Idavidrein/gpqa":
                        raise SystemExit(
                            "GPQA download failed. Accept the official GPQA access "
                            "conditions in your Hugging Face account, run `hf auth login`, "
                            "and then rerun this cache command. Original error: "
                            f"{error}"
                        ) from error
                    raise
    scope = "models" if args.models_only else "models and benchmark datasets"
    print(f"[cache] selected Qwen3 {scope} are ready", flush=True)


if __name__ == "__main__":
    main()
