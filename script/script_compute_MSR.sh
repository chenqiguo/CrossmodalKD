### for WordNet:

python compute_msr_wn.py \
  --classnames MSR/classnames_cifar100.txt \
  --noun_only \
  --out_json MSR/msr_cifar100_wn.json

python compute_msr_wn.py \
  --classnames MSR/classnames_ImageNet.txt \
  --noun_only \
  --out_json MSR/msr_ImageNet_wn.json

python compute_msr_wn.py \
  --classnames MSR/classnames_scene.txt \
  --noun_only \
  --out_json MSR/msr_scene_wn.json

python compute_msr_wn.py \
  --classnames MSR/classnames_UTKFace.txt \
  --noun_only \
  --out_json MSR/msr_UTKFace_wn.json


### for LLM:

python compute_msr_llm.py \
  --llm_json /home/ps/scratch/CLIP_KD/data_LLM/CIFAR100/LLM_selected_terms_with_class.json \
  --noun_only \
  --out_json /home/ps/scratch/CLIP_KD/MSR/msr_cifar100_llm.json

python compute_msr_llm.py \
  --llm_json /home/ps/scratch/CLIP_KD/data_LLM/CIFAR100_imb100/LLM_selected_terms_with_class.json \
  --noun_only \
  --out_json /home/ps/scratch/CLIP_KD/MSR/msr_CIFAR100_imb100_llm.json

python compute_msr_llm.py \
  --llm_json /home/ps/scratch/CLIP_KD/data_LLM/ImageNet/LLM_selected_terms_with_class.json \
  --noun_only \
  --out_json /home/ps/scratch/CLIP_KD/MSR/msr_ImageNet_llm.json

python compute_msr_llm.py \
  --llm_json /home/ps/scratch/CLIP_KD/data_LLM/ImageNet_LT/LLM_selected_terms_with_class.json \
  --noun_only \
  --out_json /home/ps/scratch/CLIP_KD/MSR/msr_ImageNet_LT_llm.json

python compute_msr_llm.py \
  --llm_json /home/ps/scratch/CLIP_KD/data_LLM/scene/LLM_selected_terms_with_class.json \
  --noun_only \
  --out_json /home/ps/scratch/CLIP_KD/MSR/msr_scene_llm.json







