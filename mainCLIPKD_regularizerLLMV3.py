#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thin wrapper around mainCLIPKD_regularizerV2.py:
- Monkey-patch trainer_regularizerV2 -> trainer_regularizerLLMV3
- Extend argparse with LLM-related flags
- Call V2 main() so training logic remains unchanged
"""

import importlib
import sys

# 1) Make V2 import our LLM trainer instead of the original trainer_regularizerV2
trainer_llm = importlib.import_module("trainer_regularizerLLMV3")
sys.modules["trainer_regularizerV2"] = trainer_llm

# 2) Import V2 main AFTER monkey-patching
import mainCLIPKD_regularizerV2 as v2  # noqa: E402


def _safe_add_arg(*args, **kwargs):
    """Avoid 'conflicting option string' if you re-run in notebooks."""
    try:
        v2.parser.add_argument(*args, **kwargs)
    except Exception:
        pass


# 3) Extend V2 parser with LLM args (minimal invasive)
_safe_add_arg(
    "--relax_source",
    type=str,
    default="wordnet",
    choices=["wordnet", "llm"],
    help="Source of relaxed nouns used in WordNet-branch code path. "
         "Set to llm to use llm_candidates_all.json + Algorithm1 Stage B.",
)
_safe_add_arg(
    "--llm_data_dir",
    type=str,
    default="/home/ps/scratch/CLIP_KD/data_LLM/CIFAR100",
    help="Directory for LLM json/npy caches (e.g., .../data_LLM/CIFAR100).",
)
_safe_add_arg(
    "--llm_candidates_json",
    type=str,
    default="",
    help="Path to llm_candidates_all.json (optional; default uses llm_data_dir).",
)
_safe_add_arg(
    "--llm_cache_npy",
    type=str,
    default="",
    help="Path to LLM_filtered_nouns_embedding.npy (optional; default uses llm_data_dir).",
)
_safe_add_arg(
    "--llm_cluster_num",
    type=int,
    default=150,
    help="KMeans cluster num for Algorithm1 Stage B on training images (LLM branch).",
)
_safe_add_arg(
    "--llm_topk_per_cluster",
    type=int,
    default=5,
    help="Select top-K nouns per cluster (LLM branch).",
)


if __name__ == "__main__":
    # 4) Call V2 entry
    if hasattr(v2, "main"):
        v2.main()
    else:
        # Fallback: execute V2 as script if no main() exists
        import runpy
        runpy.run_module("mainCLIPKD_regularizerV2", run_name="__main__")
