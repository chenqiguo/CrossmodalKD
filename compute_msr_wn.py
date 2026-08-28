#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:41:23 2026

@author: ps
"""

"""
Compute WordNet label coverage / mapping success rate (MSR_WN) for a dataset.

Definition:
1_WN(c) = I( synsets(g(label)) non-empty OR synsets(head(g(label))) non-empty )
MSR_WN = mean_c 1_WN(c)

- g(): lowercase, '_'->' ', trim spaces, collapse multiple spaces
- head(): last token of normalized phrase
- Optional: restrict WordNet synsets to nouns only (recommended for your setting)
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict

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

def load_classnames(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() in [".json"]:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
        raise ValueError("JSON classnames must be a list of strings.")
    else:
        # txt: one class per line
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
        return [ln for ln in lines if ln]

def ensure_wordnet():
    import nltk
    try:
        from nltk.corpus import wordnet as wn  # noqa: F401
        # simple probe
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

def compute_msr_wn(class_names: List[str], noun_only: bool = True) -> Tuple[float, List[str], List[Dict]]:
    per_class = []
    failed = []
    success_count = 0

    for cn in class_names:
        q_full = normalize_label(cn)
        q_head = head_noun(q_full)

        ok_full = has_synset(q_full, noun_only=noun_only)
        ok_head = has_synset(q_head, noun_only=noun_only) if (not ok_full) else False
        ok = ok_full or ok_head

        if ok:
            success_count += 1
        else:
            failed.append(cn)

        per_class.append({
            "class_name": cn,
            "normalized": q_full,
            "head": q_head,
            "wn_ok": bool(ok),
            "matched_by": "full" if ok_full else ("head" if ok_head else "none"),
        })

    msr = success_count / max(1, len(class_names))
    return msr, failed, per_class

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classnames", type=str, required=True,
                        help="Path to classnames (.txt one per line, or .json list).")
    parser.add_argument("--noun_only", action="store_true",
                        help="Restrict synset lookup to nouns only (recommended).")
    parser.add_argument("--out_json", type=str, default="",
                        help="Optional: write detailed results to JSON.")
    args = parser.parse_args()

    ensure_wordnet()

    class_names = load_classnames(args.classnames)
    if len(class_names) == 0:
        raise ValueError("No class names loaded.")

    msr, failed, per_class = compute_msr_wn(class_names, noun_only=args.noun_only)

    print(f"[MSR_WN] num_classes={len(class_names)} noun_only={args.noun_only} msr={msr:.4f}")
    print(f"[MSR_WN] failed={len(failed)}")
    if failed:
        print("[MSR_WN] failed class names:")
        for x in failed:
            print("  -", x)

    if args.out_json:
        out = {
            "num_classes": len(class_names),
            "noun_only": bool(args.noun_only),
            "msr_wn": msr,
            "failed": failed,
            "per_class": per_class,
        }
        Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] wrote {args.out_json}")

if __name__ == "__main__":
    main()






