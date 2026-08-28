### for WordNet: estimate it!

python estimate_wordnet_cco_by_wup.py \
  --dataset CIFAR100 \
  --classnames CCO/classnames_cifar100.txt \
  --wordnet_terms CCO/selected_wordnet_nouns_cifar100.txt \
  --mode multi \
  --topk 3 \
  --margin 0.001 \
  --tau_abs 0.7 \
  --out_terms_with_class CCO/WN_estimated_terms_with_class_cifar100.json \
  --out_stats CCO/WN_estimated_stats_cifar100.json

python estimate_wordnet_cco_by_wup.py \
  --dataset ImageNet \
  --classnames CCO/classnames_ImageNet.txt \
  --wordnet_terms CCO/selected_wordnet_nouns_ImageNet.txt \
  --mode multi \
  --topk 3 \
  --margin 0.001 \
  --tau_abs 0.7 \
  --out_terms_with_class CCO/WN_estimated_terms_with_class_ImageNet.json \
  --out_stats CCO/WN_estimated_stats_ImageNet.json


python estimate_wordnet_cco_by_wup.py \
  --dataset scene \
  --classnames CCO/classnames_scene.txt \
  --wordnet_terms CCO/selected_wordnet_nouns_scene.txt \
  --mode multi \
  --topk 3 \
  --margin 0.001 \
  --tau_abs 0.7 \
  --out_terms_with_class CCO/WN_estimated_terms_with_class_scene.json \
  --out_stats CCO/WN_estimated_stats_scene.json

python estimate_wordnet_cco_by_wup.py \
  --dataset UTKFace \
  --classnames CCO/classnames_UTKFace.txt \
  --wordnet_terms CCO/selected_wordnet_nouns_UTKFace.txt \
  --mode multi \
  --topk 3 \
  --margin 0.001 \
  --tau_abs 0.7 \
  --out_terms_with_class CCO/WN_estimated_terms_with_class_UTKFace.json \
  --out_stats CCO/WN_estimated_stats_UTKFace.json


### for LLM:

python compute_cross_class_overlap.py \
  --json /home/ps/scratch/CLIP_KD/data_LLM/CIFAR100/LLM_selected_terms_with_class.json \
  --term_field selected_terms \
  --out /home/ps/scratch/CLIP_KD/CCO/overlap_llm_selected_CIFAR100.json

python compute_cross_class_overlap.py \
  --json /home/ps/scratch/CLIP_KD/data_LLM/CIFAR100_imb100/LLM_selected_terms_with_class.json \
  --term_field selected_terms \
  --out /home/ps/scratch/CLIP_KD/CCO/overlap_llm_selected_CIFAR100_imb100.json

python compute_cross_class_overlap.py \
  --json /home/ps/scratch/CLIP_KD/data_LLM/ImageNet/LLM_selected_terms_with_class.json \
  --term_field selected_terms \
  --out /home/ps/scratch/CLIP_KD/CCO/overlap_llm_selected_ImageNet.json

python compute_cross_class_overlap.py \
  --json /home/ps/scratch/CLIP_KD/data_LLM/ImageNet_LT/LLM_selected_terms_with_class.json \
  --term_field selected_terms \
  --out /home/ps/scratch/CLIP_KD/CCO/overlap_llm_selected_ImageNet_LT.json

python compute_cross_class_overlap.py \
  --json /home/ps/scratch/CLIP_KD/data_LLM/scene/LLM_selected_terms_with_class.json \
  --term_field selected_terms \
  --out /home/ps/scratch/CLIP_KD/CCO/overlap_llm_selected_scene.json









