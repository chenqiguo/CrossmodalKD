#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM text-only extension on top of trainer_regularizerV2.py (minimal-invasive).

Key idea (V4):
- Keep training logic unchanged.
- Switch source via args.relax_source in get_CLIP_text_embeddings_WordNet().
- For relax_source == 'llm_textonly':
    candidates -> CLIP text embeddings (multi-template avg) -> use ALL unique candidates as noun pool
    (NO kmeans alignment with training images)

Artifacts saved to args.llm_data_dir:
- LLM_textonly_nouns_embedding.npy (cache)
- LLM_textonly_terms.txt
- LLM_textonly_terms_with_class.json (per-class candidates view + reproducibility metadata)
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

import trainer_regularizerV2 as base  # reuse everything

# Re-export all symbols so mainV2 can keep using them
from trainer_regularizerV2 import *  # noqa: F401,F403


# ---- LLM prompt: treat underscores as spaces (like GT) ----
def get_prompt_LLM(words, index, device="cuda"):
    prompt = [base.SIMPLE_IMAGENET_TEMPLATES[index](w.replace("_", " ")) for w in words]
    text = base.clip.tokenize(prompt, truncate=True).to(device)
    return text


def _default_llm_dir(args) -> str:
    return getattr(args, "llm_data_dir", "/home/ps/scratch/CLIP_KD/data_LLM/CIFAR100")


def _llm_candidates_json_path(args) -> str:
    p = getattr(args, "llm_candidates_json", None)
    if p:
        return p
    return os.path.join(_default_llm_dir(args), "llm_candidates_all.json")


def _llm_cache_npy_path_textonly(args) -> str:
    p = getattr(args, "llm_cache_npy", None)
    if p:
        return p
    return os.path.join(_default_llm_dir(args), "LLM_textonly_nouns_embedding.npy")


def _load_llm_candidates_with_class(args) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    """
    Load llm_candidates_all.json and return:
      - class_to_terms: dict[class_name] -> list of candidate terms (as stored)
      - raw_pairs: list[(class_name, term)] (in original order)
    """
    path = _llm_candidates_json_path(args)
    if not os.path.exists(path):
        raise FileNotFoundError(f"LLM candidates json not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if "classes" not in obj or not isinstance(obj["classes"], list) or len(obj["classes"]) == 0:
        raise ValueError("Invalid llm_candidates_all.json: classes is empty or missing")

    class_to_terms: Dict[str, List[str]] = {}
    raw_pairs: List[Tuple[str, str]] = []

    for item in obj["classes"]:
        cn = item.get("class_name", None)
        cands = item.get("candidates", None)
        if cn is None or cands is None:
            raise ValueError("Invalid llm_candidates_all.json: missing class_name or candidates")
        if not isinstance(cands, list):
            raise ValueError(f"Invalid candidates list for class {cn}")

        terms = []
        for c in cands:
            term = c.get("term", None)
            if term is None:
                raise ValueError(f"Invalid candidate entry for class {cn}: missing term")
            terms.append(term)
            raw_pairs.append((cn, term))

        class_to_terms[cn] = terms

    return class_to_terms, raw_pairs


def _dedupe_terms_preserve_order(raw_pairs: List[Tuple[str, str]]) -> List[str]:
    """Deduplicate terms globally while preserving first-seen order."""
    seen = set()
    uniq = []
    for _, term in raw_pairs:
        if term not in seen:
            seen.add(term)
            uniq.append(term)
    if len(uniq) == 0:
        raise ValueError("No candidate terms found after dedupe.")
    return uniq


def _encode_terms_clip_text(args, model_CLIP, device, terms: List[str]) -> np.ndarray:
    """
    Encode a list of noun terms into CLIP text embeddings with multi-template averaging.
    Output: (N, 512) float32, L2-normalized.
    """
    nouns = np.array(terms, dtype=object)
    nouns_num = nouns.shape[0]

    # Keep batch large to be fast; adjust if OOM
    batch_size = getattr(args, "llm_text_batch", 2048)

    nouns_embedding = np.zeros((nouns_num, 512), dtype=np.float32)

    for index in range(len(base.SIMPLE_IMAGENET_TEMPLATES)):
        features_list = []
        print("[LLM-textonly] Inferring text features for template index", index)
        for i in range(nouns_num // batch_size + 1):
            start = i * batch_size
            end = min(start + batch_size, nouns_num)
            if start >= end:
                continue
            nouns_batch = nouns[start:end].tolist()

            with torch.no_grad():
                prompt = get_prompt_LLM(nouns_batch, index, device=device)
                feat = model_CLIP.encode_text(prompt)
                features_list.append(feat.cpu().numpy())

            if i % 50 == 0:
                print(f"[LLM-textonly]   completed {min(i * batch_size, nouns_num)}/{nouns_num}")

        feats_index = np.concatenate(features_list, axis=0)
        nouns_embedding += feats_index

    nouns_embedding = nouns_embedding / len(base.SIMPLE_IMAGENET_TEMPLATES)
    nouns_embedding = nouns_embedding / np.linalg.norm(nouns_embedding, axis=1, keepdims=True)
    return nouns_embedding


def get_CLIP_text_embeddings_LLM_textonly(args, model_CLIP, device, train_loader):
    """
    V4: LLM Stage B (text-only)
      candidates (per class) -> global dedupe -> CLIP text embedding (multi-template avg)
      -> return ALL unique candidate embeddings (no kmeans selection)
      -> save cache + terms list + terms_with_class json
    """
    out_dir = _default_llm_dir(args)
    os.makedirs(out_dir, exist_ok=True)

    cache_npy = _llm_cache_npy_path_textonly(args)

    # 0) load cache
    if os.path.exists(cache_npy):
        emb = np.load(cache_npy)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        print(f"[LLM-textonly] Loaded cache: {cache_npy} (N={emb.shape[0]})")
        return emb

    # 1) load raw candidates with class mapping
    class_to_terms, raw_pairs = _load_llm_candidates_with_class(args)

    # 2) global dedupe (preserve order)
    uniq_terms = _dedupe_terms_preserve_order(raw_pairs)
    print(f"[LLM-textonly] Unique candidates after dedupe: {len(uniq_terms)}")

    # 3) encode ALL unique candidates
    emb = _encode_terms_clip_text(args, model_CLIP, device, uniq_terms)

    # 4) save cache
    np.save(cache_npy, emb)
    print(f"[LLM-textonly] Saved embedding cache: {cache_npy}")

    # 5) save term list (aligned with embedding rows)
    terms_txt = os.path.join(out_dir, "LLM_textonly_terms.txt")
    with open(terms_txt, "w", encoding="utf-8") as f:
        for t in uniq_terms:
            f.write(str(t) + "\n")
    print(f"[LLM-textonly] Saved term list: {terms_txt}")

    # 6) save per-class json (reference / reproducibility)
    meta = {
        "relax_source": "llm_textonly",
        "candidates_json": _llm_candidates_json_path(args),
        "cache_npy": cache_npy,
        "num_unique_candidates": int(len(uniq_terms)),
        "note": "V4 uses ALL unique candidates (no kmeans alignment / no selection).",
    }
    out_json = {
        "dataset": "CIFAR100",
        "K": int(getattr(args, "K", 10)) if hasattr(args, "K") else 10,
        "vocab": meta,
        "classes": []
    }

    for cn in sorted(class_to_terms.keys()):
        out_json["classes"].append({
            "class_name": cn,
            "num_candidates": int(len(class_to_terms[cn])),
            "candidates": class_to_terms[cn],
        })

    json_path = os.path.join(out_dir, "LLM_textonly_terms_with_class.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"[LLM-textonly] Saved per-class mapping: {json_path}")

    return emb


# ---- Entry-point hijack: keep name used by V2 training logic ----
_base_get_wordnet = base.get_CLIP_text_embeddings_WordNet


def get_CLIP_text_embeddings_WordNet(args, model_CLIP, device, train_loader):
    """
    V2 training code calls this function whenever useWhatModal contains 'WordNet'.
    We keep the same name, but switch the source via args.relax_source:
      - wordnet (default): call original V2 implementation
      - llm: (kept for compatibility) call your V3 path if you still use that file
      - llm_textonly: use candidates from llm_candidates_all.json and encode ALL unique candidates
    """
    relax_source = getattr(args, "relax_source", "wordnet").lower()
    if relax_source == "llm_textonly":
        return get_CLIP_text_embeddings_LLM_textonly(args, model_CLIP, device, train_loader)

    # If someone mistakenly points relax_source=llm here, we fallback to V2 WordNet;
    # (Your V3 file handled relax_source=llm explicitly.)
    if relax_source == "llm":
        print("[WARN] relax_source=llm requested but trainer is V4(textonly). "
              "Falling back to WordNet in V2. If you need V3 behavior, use trainer_regularizerLLMV3.py.")
        return _base_get_wordnet(args, model_CLIP, device, train_loader)

    return _base_get_wordnet(args, model_CLIP, device, train_loader)
