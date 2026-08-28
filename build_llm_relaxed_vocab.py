#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge + validate LLM candidate JSON batches into a single llm_candidates_all.json.

Expected input schema (per batch):
{
  "dataset": "CIFAR100",
  "K": 10,
  "classes": [
    {"class_name": "...", "candidates": [{"term":"...","relation_type":"..."} x10]},
    ...
  ]
}

Output:
- llm_candidates_all.json (same schema; classes merged & sorted by class_name)
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

REL_TYPES = {"synonym", "hypernym", "co_hyponym", "general_noun"}
TERM_RE = re.compile(r"^[a-z_]+$")


def _validate_candidate_term(term: str, class_name: str) -> Tuple[bool, str]:
    if not isinstance(term, str) or not term:
        return False, "term is not a non-empty string"
    if TERM_RE.fullmatch(term) is None:
        return False, f"term has invalid chars: {term}"
    words = term.split("_")
    if len(words) > 6:
        return False, f"term has >6 words: {term}"
    # disallow exact class_name or trivial inflections (very rough guard)
    if term == class_name:
        return False, f"term equals class_name: {term}"
    if term.replace("_", "") == class_name.replace("_", ""):
        return False, f"term is trivial variation of class_name: {term}"
    return True, ""


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_batch(obj: Dict, expected_dataset: str, expected_k: int) -> None:
    if not isinstance(obj, dict):
        raise ValueError("Batch JSON is not a dict.")
    if obj.get("dataset") != expected_dataset:
        raise ValueError(f"dataset mismatch: {obj.get('dataset')} != {expected_dataset}")
    if obj.get("K") != expected_k:
        raise ValueError(f"K mismatch: {obj.get('K')} != {expected_k}")
    classes = obj.get("classes")
    if not isinstance(classes, list) or len(classes) == 0:
        raise ValueError("classes must be a non-empty list.")
    for item in classes:
        if not isinstance(item, dict) or "class_name" not in item or "candidates" not in item:
            raise ValueError("Each classes[i] must have class_name and candidates.")
        if not isinstance(item["class_name"], str) or not item["class_name"]:
            print('***** item = ' + str(item))
            raise ValueError("class_name must be a non-empty string.")
        cands = item["candidates"]
        if not isinstance(cands, list) or len(cands) != expected_k:
            raise ValueError(f"Each class must have exactly K={expected_k} candidates.")
        for c in cands:
            if not isinstance(c, dict) or "term" not in c or "relation_type" not in c:
                raise ValueError("Each candidate must have term and relation_type.")
            if c["relation_type"] not in REL_TYPES:
                raise ValueError(f"Invalid relation_type: {c['relation_type']}")


def _dedupe_within_class(class_name: str, candidates: List[Dict], expected_k: int) -> List[Dict]:
    seen = set()
    out = []
    for c in candidates:
        term = c["term"]
        ok, msg = _validate_candidate_term(term, class_name)
        if not ok:
            raise ValueError(f"[{class_name}] invalid term '{term}': {msg}")
        # case-insensitive dedupe (though all lowercase by constraint)
        key = term
        if key in seen:
            continue
        seen.add(key)
        out.append({"term": term, "relation_type": c["relation_type"]})

    if len(out) != expected_k:
        raise ValueError(
            f"[{class_name}] candidates after dedupe = {len(out)} != {expected_k}. "
            f"Please regenerate this class to keep exactly {expected_k} unique terms."
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsons",
        nargs="+",
        required=True,
        help="List of batch json paths (e.g., llm_candidates_batch1.json ...).",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory (will write llm_candidates_all.json there).",
    )
    parser.add_argument("--dataset", default="CIFAR100")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--expected_num_classes", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    merged: Dict[str, Dict] = {}

    for p in args.input_jsons:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        print(p)
        obj = _load_json(p)
        _validate_batch(obj, expected_dataset=args.dataset, expected_k=args.K)

        for item in obj["classes"]:
            cn = item["class_name"]
            if cn in merged:
                raise ValueError(f"Duplicate class_name across batches: {cn}")
            merged[cn] = {
                "class_name": cn,
                "candidates": _dedupe_within_class(cn, item["candidates"], args.K),
            }

    if len(merged) != args.expected_num_classes:
        raise ValueError(
            f"Merged class count = {len(merged)} != expected {args.expected_num_classes}. "
            f"Check your batch splits."
        )

    out = {
        "dataset": args.dataset,
        "K": args.K,
        "classes": [merged[k] for k in sorted(merged.keys())],
    }

    out_path = os.path.join(args.out_dir, "llm_candidates_all.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
