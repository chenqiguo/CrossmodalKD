### Train the Teacher:


CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_regularizerLLMV3.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 --llm_topk_per_cluster 5 --llm_cluster_num 150 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --relax_source llm \
      --llm_data_dir /home/ps/scratch/CLIP_KD/data_LLM/CIFAR100 \
      --add_name get_Ts_image_textLLM_V3

## V4 NOT use!
#CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_regularizerLLMtextonlyV4.py --mode train --arch resnet50 --dataset CIFAR100 \
#      --epoch 61 \
#      --lr_steps 25 40 60 \
#      --batch_size 128 \
#      --train_add_strong  \
#      --useWhatModal image_textWordNet \
#      --relax_source llm_textonly \
#      --llm_data_dir /home/ps/scratch/CLIP_KD/data_LLM/CIFAR100 \
#      --add_name get_Ts_image_textLLMtextonly_V4

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_regularizerLLMV3.py --mode train --arch resnet50 --dataset CIFAR100_imb100 \
      --epoch 200 --llm_topk_per_cluster 5 --llm_cluster_num 150 \
      --lr_steps 50 100 150 \
      --batch_size 512 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --relax_source llm \
      --llm_data_dir /home/ps/scratch/CLIP_KD/data_LLM/CIFAR100_imb100 \
      --add_name get_Ts_image_textLLM_V3

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_regularizerLLMV3.py --mode train --arch resnet50 --dataset scene \
      --epoch 61 --lr 0.001 --llm_topk_per_cluster 5 --llm_cluster_num 150 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --relax_source llm \
      --llm_data_dir /home/ps/scratch/CLIP_KD/data_LLM/scene \
      --add_name get_Ts_image_textLLM_V3



CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_regularizerLLMV3.py --mode train --arch resnet50 --dataset ImageNet \
      --epoch 35 --llm_topk_per_cluster 5 --llm_cluster_num 1100 \
      --lr_CosineAnnealing 30 0 \
      --batch_size 1024 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --relax_source llm \
      --llm_data_dir /home/ps/scratch/CLIP_KD/data_LLM/ImageNet \
      --add_name get_Ts_image_textLLM_V3


CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_regularizerLLMV3.py --mode train --arch resnet50 --dataset ImageNet_LT \
      --epoch 60 --llm_topk_per_cluster 5 --llm_cluster_num 1100 \
      --lr_steps 35 50 0 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --relax_source llm \
      --llm_data_dir /home/ps/scratch/CLIP_KD/data_LLM/ImageNet_LT \
      --add_name get_Ts_image_textLLM_V3


### Distill the Student:


CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_regularizerLLMV3.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T4sImgCATllmV3_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_regularizerLLMV3.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATllmV3_Ss

## V4 NOT use!
#CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_regularizerLLMtextonlyV4.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
#   --epoch 500 --temp 3.0 --alpha 0.6 \
#   --lr_steps 300 400 0 \
#   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
#   --relax_source llm_textonly \
#   --llm_data_dir /home/ps/scratch/CLIP_KD/data_LLM/CIFAR100 \
#   --add_name T1s_T1w_T4sImgCATllmtextonlyV4_Ss


CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_regularizerLLMV3.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T4sImgCATllmV3_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_regularizerLLMV3.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATllmV3_Ss


CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_regularizerLLMV3.py --mode distil --arch resnet18 --dataset scene --teacher_num 1 \
   --epoch 200 --temp 3.0 --alpha 0.6 --lr 0.001 \
   --lr_steps 190 195 0 \
   --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T4sImgCATllmV3_Ss

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_regularizerLLMV3.py --mode distil --arch resnet18 --dataset scene --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 --lr 0.001 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATllmV3_Ss



CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_regularizerLLMV3.py --mode distil --arch resnet18 --dataset ImageNet --teacher_num 3 \
   --epoch 100 --temp 3.0 --alpha 0.6 \
   --lr_CosineAnnealing 100 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATllmV3_Ss \
   --next_continue runs/ImageNet_distil_T1s_T1w_T4sImgCATllmV3_Ss/run-1-epoch100/checkpoint_each-epoch.pth.tar

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_regularizerLLMV3.py --mode distil --arch resnet18 --dataset ImageNet --teacher_num 1 \
   --epoch 100 --temp 3.0 --alpha 0.6 \
   --lr_CosineAnnealing 100 0 \
   --batch_size 512 \
   --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T4sImgCATllmV3_Ss \
   --next_continue runs/ImageNet_distil_T4sImgCATllmV3_Ss/run-1-epoch100/checkpoint_each-epoch.pth.tar


CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_regularizerLLMV3.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 1 \
   --epoch 165 --temp 3.0 --alpha 0.6 \
   --lr_steps 150 155 160 \
   --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T4sImgCATllmV3_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_regularizerLLMV3.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 3 \
   --epoch 165 --temp 3.0 --alpha 0.6 \
   --lr_steps 150 155 160 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATllmV3_Ss









