#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 20:49:50 2026

@author: ps
"""

"""
Estimate cross-class term overlap (CCO) for WordNet-selected nouns by
automatically assigning each noun to dataset classes using WordNet synset similarity.

Method (Option A):
- Map each dataset class label and each selected noun term to WordNet noun synsets.
- Compute similarity sim(t,c) = max_{s in Syn(t), s' in Syn(label(c))} wup_similarity(s,s')
  (fallback to head noun for class labels if needed)
- Assignment modes:
  * single: assign each term to argmax class if max_sim >= tau else none
  * multi : assign each term to all classes with sim >= tau (can be empty -> none)

Then compute:
- frac_terms_shared_ge2 = (#unique terms assigned to >=2 classes) / (#unique terms assigned to >=1 class)
  (terms assigned to none are excluded by default, but can be included via flag)

Outputs:
- prints CCO + summary stats
- writes per-class JSON: WN_estimated_terms_with_class.json
- optionally writes stats JSON
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean, median

# -------------------------
# text utils
# -------------------------
def normalize(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def head_noun(phrase: str) -> str:
    phrase = phrase.strip()
    if not phrase:
        return ""
    parts = phrase.split(" ")
    return parts[-1] if parts else ""

# -------------------------
# WordNet utils
# -------------------------
def ensure_wordnet():
    import nltk
    try:
        from nltk.corpus import wordnet as wn  # noqa: F401
        _ = wn.synsets("dog")
    except Exception:
        print("[INFO] Downloading NLTK WordNet data...")
        nltk.download("wordnet")
        nltk.download("omw-1.4")

def synsets_noun(q: str):
    from nltk.corpus import wordnet as wn
    if not q:
        return []
    return wn.synsets(q, pos=wn.NOUN)

def wup_max(syn_a, syn_b) -> float:
    """max Wu-Palmer similarity across synset pairs"""
    best = 0.0
    for sa in syn_a:
        for sb in syn_b:
            v = sa.wup_similarity(sb)
            if v is None:
                continue
            if v > best:
                best = float(v)
    return best

# -------------------------
# IO
# -------------------------
def load_classnames(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, list):
            raise ValueError("classnames JSON must be a list of strings")
        return [str(x).strip() for x in obj if str(x).strip()]
    else:
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
        return [ln for ln in lines if ln]

"""
def parse_wordnet_terms_file(path: str):
    
    #Accepts:
    #- one term per line
    #- OR a python/numpy-ish print like: ['a' 'b' 'c']
    
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return []

    # If looks like bracketed list, extract quoted tokens
    if ("[" in txt and "]" in txt) and ("'" in txt):
        terms = re.findall(r"'([^']+)'", txt)
        # also handle double quotes if any
        if not terms:
            terms = re.findall(r"\"([^\"]+)\"", txt)
        return [t.strip() for t in terms if t.strip()]

    # Otherwise treat as one per line
    lines = [ln.strip() for ln in txt.splitlines()]
    # allow lines like: term1 term2 term3 (space separated)
    out = []
    for ln in lines:
        if not ln:
            continue
        if " " in ln and "_" not in ln and len(ln.split()) > 1:
            # could be accidental multi-column; keep as-is (it will normalize)
            out.append(ln)
        else:
            out.append(ln)
    return out
"""

import re
from pathlib import Path

def parse_wordnet_terms_file(path: str):
    """
    Robust parser for tokens like:
      'mess-up' 'rocket_firing' ... "pope's_nose" 'old_man's_beard' ...
    Key idea: split by whitespace, then strip only OUTER quotes.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    txt = p.read_text(encoding="utf-8")

    # Split by any whitespace (handles newlines/indentation)
    raw_tokens = re.split(r"\s+", txt.strip())

    terms = []
    for tok in raw_tokens:
        tok = tok.strip()
        if not tok:
            continue

        # Strip trailing commas or brackets if any
        tok = tok.strip(",")
        tok = tok.strip()

        # Strip only the outer quote if present (keep internal apostrophes)
        if len(tok) >= 2 and tok[0] == "'" and tok[-1] == "'":
            tok = tok[1:-1]
        elif len(tok) >= 2 and tok[0] == '"' and tok[-1] == '"':
            tok = tok[1:-1]

        tok = tok.strip()
        if tok:
            terms.append(tok)

    return terms

# -------------------------
# core
# -------------------------
def build_class_synsets(class_names):
    """
    For each class, store synsets for full phrase and head backoff.
    """
    cls_syn = {}
    for c in class_names:
        q_full = normalize(c)
        q_head = head_noun(q_full)
        syn_full = synsets_noun(q_full)
        syn_head = synsets_noun(q_head) if (not syn_full) else []
        syn = syn_full if syn_full else syn_head
        cls_syn[c] = {
            "norm": q_full,
            "head": q_head,
            "synsets": syn,
            "matched_by": "full" if syn_full else ("head" if syn_head else "none"),
        }
    return cls_syn

def assign_terms_to_classes(terms, class_names, cls_syn, tau_abs, topk, margin, mode: str):
    """
    Return:
    - term_to_classes: dict term -> list of assigned classes (empty means none)
    - term_to_scores: dict term -> list of (class, score) for debugging (top few)
    """
    term_to_classes = {}
    term_to_scores = {}

    for t in terms:
        t_norm = normalize(t)
        t_syn = synsets_noun(t_norm) or synsets_noun(head_noun(t_norm))
        if not t_syn:
            # cannot map term -> assign none
            term_to_classes[t] = []
            term_to_scores[t] = []
            continue

        scores = []
        best_c = None
        best_s = -1.0

        for c in class_names:
            syn_c = cls_syn[c]["synsets"]
            if not syn_c:
                continue
            s = wup_max(t_syn, syn_c)
            if s > 0:
                scores.append((c, s))
            if s > best_s:
                best_s = s
                best_c = c

        scores.sort(key=lambda x: x[1], reverse=True)
        term_to_scores[t] = scores[:10]

        # scores already sorted desc: [(c, s), ...]
        if mode == "single":
            if best_c is not None and best_s >= tau_abs:
                term_to_classes[t] = [best_c]
            else:
                term_to_classes[t] = []
        else:
            # NEW: multi via top-k + relative margin (prevents huge over-assignment)
            if best_s < tau_abs:
                term_to_classes[t] = []
            else:
                top = scores[:topk]
                keep = [c for c, s in top if s >= best_s - margin and s >= tau_abs]
                term_to_classes[t] = keep

    return term_to_classes, term_to_scores

def compute_cgo_stats(term_to_classes, include_none: bool):
    """
    Compute frac_terms_shared_ge2.
    - By default exclude terms assigned to none from denominator.
    """
    # term frequency = number of classes it is assigned to
    freqs = {}
    for t, cls in term_to_classes.items():
        freqs[t] = len(set(cls))

    if include_none:
        denom_terms = list(freqs.keys())
    else:
        denom_terms = [t for t, f in freqs.items() if f >= 1]

    if not denom_terms:
        return {
            "frac_terms_shared_ge2": 0.0,
            "num_terms_denom": 0,
            "num_terms_shared_ge2": 0,
            "avg_classes_per_term": 0.0,
        }

    num_shared_ge2 = sum(1 for t in denom_terms if freqs[t] >= 2)
    frac_shared_ge2 = num_shared_ge2 / len(denom_terms)

    # avg classes per term (only denom)
    avg_classes = mean(freqs[t] for t in denom_terms)

    return {
        "frac_terms_shared_ge2": frac_shared_ge2,
        "num_terms_denom": len(denom_terms),
        "num_terms_shared_ge2": num_shared_ge2,
        "avg_classes_per_term": avg_classes,
    }

def export_per_class_json(dataset: str, class_names, term_to_classes, out_path: str):
    class_to_terms = defaultdict(list)
    for t, cls_list in term_to_classes.items():
        for c in cls_list:
            class_to_terms[c].append(t)

    out = {
        "dataset": dataset,
        "classes": []
    }
    for c in sorted(class_names):
        out["classes"].append({
            "class_name": c,
            "selected_terms": sorted(set(class_to_terms.get(c, [])))
        })

    Path(out_path).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--classnames", type=str, required=True, help="txt/json list of class names")
    ap.add_argument("--wordnet_terms", type=str, required=True, help="txt containing selected WordNet nouns list")
    ap.add_argument("--tau", type=float, default=0.35, help="similarity threshold for assignment (default 0.35)")
    ap.add_argument("--mode", type=str, choices=["single", "multi"], default="multi",
                    help="single: one best class; multi: all classes with sim>=tau (default multi)")
    ap.add_argument("--include_none", action="store_true",
                    help="Include terms assigned to no class in denominator (default exclude).")
    ap.add_argument("--out_terms_with_class", type=str, default="WN_estimated_terms_with_class.json")
    ap.add_argument("--out_stats", type=str, default="", help="Optional stats json output")
    ap.add_argument("--topk", type=int, default=3, help="Keep at most top-k classes per term (default 3).")
    ap.add_argument("--margin", type=float, default=0.05, help="Assign classes with sim >= best_sim - margin (default 0.05).")
    ap.add_argument("--tau_abs", type=float, default=0.40, help="Minimum absolute similarity to accept (default 0.40).")
    args = ap.parse_args()

    ensure_wordnet()

    class_names = load_classnames(args.classnames)
    terms = parse_wordnet_terms_file(args.wordnet_terms)
    # global dedupe terms
    terms = sorted(set(t for t in terms if str(t).strip()))

    print(f"[Input] dataset={args.dataset} classes={len(class_names)} unique_terms={len(terms)}")
    print(f"[Config] mode={args.mode} tau={args.tau} include_none={args.include_none}")

    # build class synsets
    cls_syn = build_class_synsets(class_names)
    # assign
    term_to_classes, term_to_scores = assign_terms_to_classes(
        terms, class_names, cls_syn, tau_abs=args.tau_abs, topk=args.topk, margin=args.margin, mode=args.mode
    )

    # compute overlap-like stats
    stats = compute_cgo_stats(term_to_classes, include_none=args.include_none)
    print(f"[CCO] frac_terms_shared_ge2={stats['frac_terms_shared_ge2']:.4f} "
          f"(denom_terms={stats['num_terms_denom']}, shared_ge2={stats['num_terms_shared_ge2']})")
    print(f"[CCO] avg_classes_per_term={stats['avg_classes_per_term']:.2f}")

    # export per-class mapping json
    export_per_class_json(args.dataset, class_names, term_to_classes, args.out_terms_with_class)
    print(f"[OK] wrote per-class mapping: {args.out_terms_with_class}")

    if args.out_stats:
        out = {
            "dataset": args.dataset,
            "num_classes": len(class_names),
            "num_unique_terms": len(terms),
            "tau": args.tau,
            "mode": args.mode,
            "include_none": args.include_none,
            "cco": stats,
            "note": "Estimated via WordNet Wu-Palmer similarity between noun synsets and class-label synsets (with head backoff)."
        }
        Path(args.out_stats).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] wrote stats: {args.out_stats}")

if __name__ == "__main__":
    main()