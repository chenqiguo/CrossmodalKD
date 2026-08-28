#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thin wrapper around mainCLIPKD_regularizerV2.py (minimal-invasive):
- Monkey-patch trainer_regularizerV2 -> trainer_regularizerLLMtextonlyV4
- Extend argparse with LLM-related flags
- Call V2 main() so training logic remains unchanged

V4 change:
- relax_source=llm_textonly will use ALL LLM candidate nouns (unique deduped) as the noun-embedding pool
- No k-means alignment with training images
"""

import importlib
import sys

# 1) Make V2 import our LLM trainer instead of the original trainer_regularizerV2
trainer_llm = importlib.import_module("trainer_regularizerLLMtextonlyV4")
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
    choices=["wordnet", "llm", "llm_textonly"],
    help="Source of relaxed nouns used in WordNet-branch code path. "
         "Set to llm_textonly to use llm_candidates_all.json and encode ALL unique candidates "
         "(no kmeans filtering).",
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
    help="Path to cached LLM noun embeddings npy (optional; default uses llm_data_dir).",
)

if __name__ == "__main__":
    # 4) Call V2 entry
    if hasattr(v2, "main"):
        v2.main()
    else:
        import runpy
        runpy.run_module("mainCLIPKD_regularizerV2", run_name="__main__")
