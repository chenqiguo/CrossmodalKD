#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM extension on top of trainer_regularizerV2.py (minimal-invasive).

Key idea:
- Keep training logic unchanged.
- Reuse Algorithm1 Stage B (image-embedding kmeans alignment) for LLM candidates.
- Switch source via args.relax_source in get_CLIP_text_embeddings_WordNet().
"""

import json
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import trainer_regularizerV2 as base  # reuse everything

# Re-export all symbols so mainV2 can keep using them
from trainer_regularizerV2 import *  # noqa: F401,F403


# ---- LLM prompt: treat underscores as spaces (like GT) ----
def get_prompt_LLM(words, index, device="cuda"):
    # base.SIMPLE_IMAGENET_TEMPLATES exists in V2 :contentReference[oaicite:2]{index=2}
    prompt = [base.SIMPLE_IMAGENET_TEMPLATES[index](w.replace("_", " ")) for w in words]
    text = base.clip.tokenize(prompt, truncate=True).to(device)
    return text


def _default_llm_dir(args) -> str:
    # user requested CIFAR100 path; keep it as default fallback
    return getattr(args, "llm_data_dir", "/home/ps/scratch/CLIP_KD/data_LLM/CIFAR100")


def _llm_candidates_json_path(args) -> str:
    p = getattr(args, "llm_candidates_json", None)
    if p:
        return p
    return os.path.join(_default_llm_dir(args), "llm_candidates_all.json")


def _llm_cache_npy_path(args) -> str:
    p = getattr(args, "llm_cache_npy", None)
    if p:
        return p
    return os.path.join(_default_llm_dir(args), "LLM_filtered_nouns_embedding.npy")


def _load_llm_terms(args) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Return:
      - unique_terms: list[str] unique candidate terms across all classes
      - raw_pairs: list[(class_name, term)] for debugging (optional)
    """
    path = _llm_candidates_json_path(args)
    if not os.path.exists(path):
        raise FileNotFoundError(f"LLM candidates json not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    #if obj.get("dataset") != "CIFAR100":
    #    # keep strict for now; can relax later
    #    raise ValueError(f"Unexpected dataset in {path}: {obj.get('dataset')}")

    classes = obj.get("classes", [])
    if not isinstance(classes, list) or len(classes) == 0:
        raise ValueError("Invalid llm_candidates_all.json: classes is empty")

    seen = set()
    unique_terms = []
    raw_pairs = []

    for item in classes:
        cn = item["class_name"]
        for c in item["candidates"]:
            term = c["term"]
            raw_pairs.append((cn, term))
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

    if len(unique_terms) == 0:
        raise ValueError("No candidate terms found in llm_candidates_all.json")

    return unique_terms, raw_pairs


def get_CLIP_text_embeddings_LLM(args, model_CLIP, device, train_loader):
    """
    LLM Stage B (reuse Algorithm1):
      candidates -> CLIP text embeddings (multi-template avg)
      -> use training images CLIP embeddings -> kmeans alignment -> select topK per cluster
      -> save npy
    """
    out_dir = _default_llm_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    cache_npy = _llm_cache_npy_path(args)

    # 0) load cache
    if os.path.exists(cache_npy):
        emb = np.load(cache_npy)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb

    # 1) load & dedupe terms
    #terms, _ = _load_llm_terms(args)
    terms, raw_pairs = _load_llm_terms(args)  # raw_pairs: [(class_name, term), ...]
    nouns = np.array(terms, dtype=object)
    nouns_num = nouns.shape[0]

    # 2) encode text (multi templates)
    batch_size = 2048
    nouns_embedding = np.zeros((nouns_num, 512), dtype=np.float32)

    for index in range(len(base.SIMPLE_IMAGENET_TEMPLATES)):
        features_list = []
        print("[LLM] Inferring text features for template index", index)
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
                print(f"[LLM]   completed {min(i * batch_size, nouns_num)}/{nouns_num}")

        feats_index = np.concatenate(features_list, axis=0)
        nouns_embedding += feats_index

    nouns_embedding = nouns_embedding / len(base.SIMPLE_IMAGENET_TEMPLATES)
    nouns_embedding = nouns_embedding / np.linalg.norm(nouns_embedding, axis=1, keepdims=True)

    # 3) Algorithm1 Step2: image alignment via kmeans (reuse V2 helpers) :contentReference[oaicite:3]{index=3}
    images_embedding = base.get_CLIP_image_embeddings_all(args, model_CLIP, train_loader, device)

    nouns_embedding_t = torch.from_numpy(nouns_embedding).cuda().half()
    images_embedding_t = torch.from_numpy(images_embedding).cuda().half()

    cluster_num = getattr(args, "llm_cluster_num", 150)  # default matches V2 WordNet :contentReference[oaicite:4]{index=4}
    topK = getattr(args, "llm_topk_per_cluster", 5)

    preds = base.kmeans(images_embedding_t.cpu().numpy(), cluster_num)

    image_centers = torch.zeros((cluster_num, 512), dtype=torch.float16).cuda()
    for k in range(cluster_num):
        if (preds == k).sum() == 0:
            continue
        image_centers[k] = images_embedding_t[preds == k].mean(dim=0)
    image_centers = F.normalize(image_centers, dim=1)

    similarity = torch.matmul(image_centers, nouns_embedding_t.T)
    softmax_nouns = torch.softmax(similarity, dim=0).cpu().float()
    class_pred = torch.argmax(softmax_nouns, dim=0).long()

    selected_idx = torch.zeros_like(class_pred, dtype=torch.bool)
    for k in range(cluster_num):
        if (class_pred == k).sum() == 0:
            continue
        class_index = torch.where(class_pred == k)[0]
        softmax_class = softmax_nouns[:, class_index]
        confidence = softmax_class.max(dim=0)[0]
        rank = torch.argsort(confidence, descending=True)
        selected_idx[class_index[rank[:topK]]] = True

    selected_idx_np = selected_idx.cpu().numpy()
    print(f"[LLM] {selected_idx_np.sum()} nouns selected by kmeans alignment.")

    nouns_embedding_selected = nouns_embedding_t[selected_idx].cpu().numpy()
    nouns_embedding_selected = nouns_embedding_selected / np.linalg.norm(
        nouns_embedding_selected, axis=1, keepdims=True
    )

    # save selected embeddings + selected terms list for debugging
    np.save(cache_npy, nouns_embedding_selected)
    sel_terms = nouns[selected_idx_np].tolist()
    with open(os.path.join(out_dir, "LLM_selected_terms.txt"), "w", encoding="utf-8") as f:
        for t in sel_terms:
            f.write(str(t) + "\n")

    print(f"[LLM] Saved: {cache_npy}")
    print(f"[LLM] Saved selected term list: {os.path.join(out_dir, 'LLM_selected_terms.txt')}")
    
    
    # ---- NEW: save selected terms with class mapping (per-class view) ----
    selected_terms_set = set(sel_terms)
    
    # build class -> [selected terms] based on original per-class candidate lists
    class_to_selected = {}
    class_to_all = {}
    for cn, term in raw_pairs:
        class_to_all.setdefault(cn, []).append(term)
        if term in selected_terms_set:
            class_to_selected.setdefault(cn, []).append(term)
    
    # ensure deterministic order & schema
    out_json = {
        "dataset": args.dataset, #"CIFAR100",
        "K": 20,
        "selection": {
            "relax_source": "llm",
            "cluster_num": int(cluster_num),
            "topk_per_cluster": int(topK),
            "num_unique_candidates": int(nouns_num),
            "num_selected_unique_terms": int(len(sel_terms)),
        },
        "classes": []
    }
    
    for cn in sorted(class_to_all.keys()):
        all_terms = class_to_all[cn]
        sel_cls = class_to_selected.get(cn, [])
        out_json["classes"].append({
            "class_name": cn,
            "num_candidates": len(all_terms),          # should be 10
            "num_selected": len(sel_cls),
            "selected_terms": sel_cls                  # keep original order from json batches
        })
    
    json_path = os.path.join(out_dir, "LLM_selected_terms_with_class.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    
    print(f"[LLM] Saved selected-term mapping: {json_path}")
    # ---- END NEW ----

    
    
    return nouns_embedding_selected


# ---- Entry-point hijack: keep name used by V2 training logic ----
_base_get_wordnet = base.get_CLIP_text_embeddings_WordNet


def get_CLIP_text_embeddings_WordNet(args, model_CLIP, device, train_loader):
    """
    V2 training code calls this function whenever useWhatModal contains 'WordNet'. :contentReference[oaicite:5]{index=5}

    We keep the same name, but switch the source via args.relax_source:
      - wordnet (default): call original V2 implementation
      - llm: use candidates from llm_candidates_all.json + Algorithm1 Stage B
    """
    relax_source = getattr(args, "relax_source", "wordnet").lower()
    if relax_source == "llm":
        return get_CLIP_text_embeddings_LLM(args, model_CLIP, device, train_loader)
    return _base_get_wordnet(args, model_CLIP, device, train_loader)
