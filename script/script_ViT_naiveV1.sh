### Train the Teacher:



+++++++++++++++++++++++++++



### for CIFAR100:

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1_ViTMLP.py --mode train --arch MLP --dataset CIFAR100 \
      --epoch 200 \
      --lr 0.1 \
      --lr_steps 120 150 0 \
      --train_add_strong  \
      --batch_size 1024 \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet_V1_MLP


### for ImageNet-mini:

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1_ViTMLP.py --mode train --arch MLP --dataset ImageNet-mini \
      --epoch 100 \
      --lr 0.05 \
      --lr_steps 70 80 90 \
      --train_add_strong  \
      --batch_size 1024 \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet_V1_MLP


### for scene:

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1_ViTMLP.py --mode train --arch MLP --dataset scene \
      --epoch 200 \
      --lr 0.1 \
      --lr_steps 120 150 0 \
      --train_add_strong  \
      --batch_size 1024 \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet_V1_MLP


### for UTKFace:

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1_ViTMLP.py --mode train --arch MLP --dataset UTKFace \
      --epoch 200 \
      --lr 0.1 \
      --lr_steps 120 150 0 \
      --train_add_strong  \
      --batch_size 1024 \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet_V1_MLP


### for ImageNet-LT:

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1_ViTMLP.py --mode train --arch MLP --dataset ImageNet_LT \
      --epoch 120 \
      --lr 0.3 \
      --lr_steps 75 100 0 \
      --train_add_weak \
      --batch_size 1024 \
      --useWhatModal image_textWordNet \
      --add_name get_Tw_image_textWordNet_V1_MLP


### for CIFAR100_imb100:

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1_ViTMLP.py --mode train --arch MLP --dataset CIFAR100_imb100 \
      --epoch 500 \
      --lr 0.3 \
      --lr_steps 250 350 450 \
      --train_add_strong  \
      --batch_size 1024 \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet_V1_MLP






+++++++++++++++++++++++++++


### Distill the Student:



+++++++++++++++++++++++++++

### for CIFAR100:

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1_ViTMLP.py --mode distil --arch vit_base_patch32_224 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T4sImgCATwordnetV1MLP_Ss_ViT


### for ImageNet-mini:

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1_ViTMLP.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T4sImgCATwordnetV1MLP_Ss_ViT


### for scene:

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1_ViTMLP.py --mode distil --arch vit_base_patch32_224 --dataset scene --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.001 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T4sImgCATwordnetV1MLP_Ss_ViT


### for UTKFace:

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1_ViTMLP.py --mode distil --arch vit_base_patch32_224 --dataset UTKFace --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.001 \
   --lr_steps 200 250 0 \
   --batch_size 512 \
   --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T4sImgCATwordnetV1MLP_Ss_ViT


### for ImageNet-LT:

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1_ViTMLP.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet_LT --teacher_num 1 \
   --epoch 240 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 210 220 230 \
   --batch_size 512 \
   --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T4sImgCATwordnetV1MLP_Ss_ViT


### for CIFAR100_imb100:

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1_ViTMLP.py --mode distil --arch vit_base_patch32_224 --dataset CIFAR100_imb100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 160 165 170 \
   --batch_size 512 \
   --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T4sImgCATwordnetV1MLP_Ss_ViT










++++++++++++++++++++






# #####################



















