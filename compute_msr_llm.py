#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 19:43:11 2026

@author: ps
"""


"""
Compute LLM mapping success rate (MSR_LLM) based on WordNet synset lookup.

Given LLM_selected_terms_with_class.json:
- For each class c with selected_terms list:
  1_LLM(c) = I( exists t in selected_terms s.t. Syn(g(t)) != empty OR Syn(head(g(t))) != empty )
- MSR_LLM = mean_c 1_LLM(c)

Normalization:
- g(): lowercase, '_'->' ', trim spaces, collapse multiple spaces
- head(): last token of normalized phrase
- Optional: restrict synset lookup to noun synsets only (recommended)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_label(s: str) -> str:
    s = s.strip().lower().replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def head_noun(phrase: str) -> str:
    phrase = phrase.strip()
    if not phrase:
        return ""
    parts = phrase.split(" ")
    return parts[-1] if parts else ""


def ensure_wordnet():
    import nltk
    try:
        from nltk.corpus import wordnet as wn  # noqa: F401
        _ = wn.synsets("dog")
    except Exception:
        print("[INFO] Downloading NLTK WordNet data...")
        nltk.download("wordnet")
        nltk.download("omw-1.4")


def has_synset(q: str, noun_only: bool) -> bool:
    from nltk.corpus import wordnet as wn
    if not q:
        return False
    if noun_only:
        return len(wn.synsets(q, pos=wn.NOUN)) > 0
    return len(wn.synsets(q)) > 0


def load_llm_selected_json(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return json.loads(p.read_text(encoding="utf-8"))


def compute_msr_llm(obj: Dict, noun_only: bool = True) -> Tuple[float, List[str], List[Dict]]:
    classes = obj.get("classes", [])
    if not isinstance(classes, list) or len(classes) == 0:
        raise ValueError("Invalid LLM_selected_terms_with_class.json: classes missing/empty")

    failed = []
    per_class = []
    ok_count = 0

    for item in classes:
        cn = item.get("class_name", "")
        terms = item.get("selected_terms", None)

        if not isinstance(cn, str) or not cn:
            raise ValueError("Invalid class_name in JSON.")
        if terms is None:
            raise ValueError(f"Missing selected_terms for class {cn}")
        if not isinstance(terms, list):
            raise ValueError(f"selected_terms must be a list for class {cn}")

        hit_terms = []
        ok = False

        for t in terms:
            if not isinstance(t, str) or not t.strip():
                continue
            q_full = normalize_label(t)
            q_head = head_noun(q_full)

            if has_synset(q_full, noun_only=noun_only) or has_synset(q_head, noun_only=noun_only):
                ok = True
                hit_terms.append(t)

        if ok:
            ok_count += 1
        else:
            failed.append(cn)

        per_class.append({
            "class_name": cn,
            "num_selected": int(item.get("num_selected", len(terms))),
            "llm_ok": bool(ok),
            "llm_hit_terms": hit_terms,  # may be empty if not ok
        })

    msr = ok_count / max(1, len(classes))
    return msr, failed, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_json", type=str, required=True,
                        help="Path to LLM_selected_terms_with_class.json")
    parser.add_argument("--noun_only", action="store_true",
                        help="Restrict synset lookup to nouns only (recommended).")
    parser.add_argument("--out_json", type=str, default="",
                        help="Optional: write detailed results to JSON.")
    args = parser.parse_args()

    ensure_wordnet()

    obj = load_llm_selected_json(args.llm_json)
    msr, failed, per_class = compute_msr_llm(obj, noun_only=args.noun_only)

    dataset = obj.get("dataset", "UNKNOWN")
    print(f"[MSR_LLM] dataset={dataset} num_classes={len(per_class)} noun_only={args.noun_only} msr={msr:.4f}")
    print(f"[MSR_LLM] failed={len(failed)}")
    if failed:
        print("[MSR_LLM] failed class names:")
        for x in failed:
            print("  -", x)

    if args.out_json:
        out = {
            "dataset": dataset,
            "num_classes": len(per_class),
            "noun_only": bool(args.noun_only),
            "msr_llm": msr,
            "failed": failed,
            "per_class": per_class,
            "selection_meta": obj.get("selection", {}),
        }
        Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] wrote {args.out_json}")


if __name__ == "__main__":
    main()