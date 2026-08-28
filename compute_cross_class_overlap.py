#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 20:23:00 2026

@author: ps
"""

"""
Compute cross-class term overlap statistics.

Input (LLM-selected per-class file):
{
  "dataset": "...",
  "classes": [
    {"class_name": "...", "selected_terms": ["...", ...], ...},
    ...
  ]
}

Outputs:
- term -> frequency (#classes containing the term)
- overlap metrics (global + per-class)
"""

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median

def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return json.loads(p.read_text(encoding="utf-8"))

def normalize_term(t: str) -> str:
    # Keep it minimal & deterministic; your terms should already be lowercase + underscores.
    return t.strip().lower()

def compute_overlap_stats(obj: dict, term_field: str = "selected_terms") -> dict:
    classes = obj.get("classes", [])
    if not isinstance(classes, list) or len(classes) == 0:
        raise ValueError("Invalid JSON: missing/empty classes.")

    class_to_terms = {}
    all_assignments = 0
    term_to_classes = defaultdict(set)

    for item in classes:
        cn = item.get("class_name", "")
        terms = item.get(term_field, None)
        if not cn:
            raise ValueError("Invalid class_name.")
        if terms is None:
            raise ValueError(f"Missing field '{term_field}' for class {cn}.")
        if not isinstance(terms, list):
            raise ValueError(f"Field '{term_field}' must be a list for class {cn}.")

        norm_terms = [normalize_term(t) for t in terms if isinstance(t, str) and t.strip()]
        # treat per-class as a SET (a term either appears in class or not)
        uniq = sorted(set(norm_terms))
        class_to_terms[cn] = uniq
        all_assignments += len(uniq)

        for t in uniq:
            term_to_classes[t].add(cn)

    num_classes = len(class_to_terms)
    unique_terms = list(term_to_classes.keys())
    num_unique_terms = len(unique_terms)

    # term frequency distribution
    term_freq = {t: len(cls_set) for t, cls_set in term_to_classes.items()}

    # global overlap rate: fraction of terms shared by >=2 classes
    shared_terms = [t for t, f in term_freq.items() if f >= 2]
    frac_terms_shared_ge2 = len(shared_terms) / max(1, num_unique_terms)

    # assignment-level overlap: how much of assignments come from shared terms
    shared_assignment_count = sum(term_freq[t] for t in shared_terms)  # counts term occurrences across classes
    frac_assignments_from_shared = shared_assignment_count / max(1, all_assignments)

    # average classes per term (weighted/unweighted)
    avg_classes_per_term = mean(term_freq.values()) if term_freq else 0.0
    med_classes_per_term = median(term_freq.values()) if term_freq else 0.0

    # per-class shared fraction
    per_class_shared_frac = {}
    for cn, terms in class_to_terms.items():
        if not terms:
            per_class_shared_frac[cn] = 0.0
            continue
        shared_in_class = sum(1 for t in terms if term_freq.get(t, 0) >= 2)
        per_class_shared_frac[cn] = shared_in_class / len(terms)

    # some useful summaries
    per_class_sizes = [len(v) for v in class_to_terms.values()]
    avg_terms_per_class = mean(per_class_sizes) if per_class_sizes else 0.0
    med_terms_per_class = median(per_class_sizes) if per_class_sizes else 0.0

    # top shared terms
    top_shared_terms = sorted(shared_terms, key=lambda t: (term_freq[t], t), reverse=True)[:50]

    return {
        "dataset": obj.get("dataset", "UNKNOWN"),
        "num_classes": num_classes,
        "term_field": term_field,
        "num_unique_terms": num_unique_terms,
        "total_class_term_assignments": all_assignments,
        "avg_terms_per_class": avg_terms_per_class,
        "median_terms_per_class": med_terms_per_class,
        "frac_terms_shared_ge2": frac_terms_shared_ge2,
        "frac_assignments_from_shared_terms": frac_assignments_from_shared,
        "avg_classes_per_term": avg_classes_per_term,
        "median_classes_per_term": med_classes_per_term,
        "per_class_shared_fraction": per_class_shared_frac,
        "top_shared_terms": [{"term": t, "num_classes": term_freq[t], "classes": sorted(term_to_classes[t])}
                             for t in top_shared_terms],
        "term_frequency": term_freq,  # large; keep if you want
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to LLM_selected_terms_with_class.json (or similar).")
    ap.add_argument("--term_field", default="selected_terms",
                    help="Which field to use per class (default: selected_terms).")
    ap.add_argument("--out", default="", help="Optional: write stats JSON to this path.")
    ap.add_argument("--print_top", type=int, default=20, help="How many top shared terms to print.")
    args = ap.parse_args()

    obj = load_json(args.json)
    stats = compute_overlap_stats(obj, term_field=args.term_field)

    print(f"[Overlap] dataset={stats['dataset']} classes={stats['num_classes']} unique_terms={stats['num_unique_terms']}")
    print(f"[Overlap] avg_terms_per_class={stats['avg_terms_per_class']:.2f}  median={stats['median_terms_per_class']:.2f}")
    print(f"[Overlap] frac_terms_shared_ge2={stats['frac_terms_shared_ge2']:.4f}  (lower is usually better)")
    print(f"[Overlap] frac_assignments_from_shared_terms={stats['frac_assignments_from_shared_terms']:.4f}")
    print(f"[Overlap] avg_classes_per_term={stats['avg_classes_per_term']:.2f}  median={stats['median_classes_per_term']:.2f}")

    top = stats["top_shared_terms"][:max(0, args.print_top)]
    if top:
        print(f"\n[Overlap] Top shared terms (term -> #classes):")
        for row in top:
            print(f"  {row['term']} -> {row['num_classes']}")

    if args.out:
        Path(args.out).write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[OK] wrote {args.out}")

if __name__ == "__main__":
    main()