### Train the Teacher:



+++++++++++++++++++++++++++



### for CIFAR100:
CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch vit_base_patch16_224 --dataset CIFAR100 \
      --epoch 200 \
      --lr 0.001 \
      --lr_steps 120 150 0 \
      --batch_size 256 \
      --train_add_strong \
      --useWhatModal rawImg \
      --add_name get_Ts_image_ViT

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch vit_base_patch16_224 --dataset CIFAR100 \
      --epoch 200 \
      --lr 0.001 \
      --lr_steps 120 150 0 \
      --batch_size 256 \
      --train_add_weak \
      --useWhatModal rawImg \
      --add_name get_Tw_image_ViT

#CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch vit_base_patch16_224 --dataset CIFAR100 \
#      --epoch 200 \
#      --lr 0.001 \
#      --lr_steps 120 150 0 \
#      --batch_size 128 \
#      --train_add_strong  \
#      --useWhatModal image_textWordNet \
#      --add_name get_Ts_image_textWordNet_V2_ViT

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset CIFAR100 \
      --epoch 200 \
      --lr 0.1 \
      --lr_steps 120 150 0 \
      --train_add_strong  \
      --batch_size 1024 \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet_V2_MLP


### for ImageNet:
CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch vit_base_patch16_224 --dataset ImageNet \
      --epoch 35 \
      --lr 0.001 \
      --lr_CosineAnnealing 30 0 \
      --batch_size 288 \
      --train_add_strong \
      --useWhatModal rawImg \
      --add_name get_Ts_image_ViT

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch vit_base_patch16_224 --dataset ImageNet \
      --epoch 35 \
      --lr 0.001 \
      --lr_CosineAnnealing 30 0 \
      --batch_size 288 \
      --train_add_weak \
      --useWhatModal rawImg \
      --add_name get_Tw_image_ViT

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset ImageNet \
      --epoch 120 \
      --lr 0.1 \
      --lr_steps 75 100 0 \
      --train_add_weak \
      --batch_size 1024 \
      --useWhatModal image_textWordNet \
      --add_name get_Tw_image_textWordNet_V2_MLP \
      --next_continue runs/ImageNet_train_get_Tw_image_textWordNet_V2_MLP/run-1-epoch200/checkpoint_bestAcc1.pth.tar


### for ImageNet-LT:

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch vit_base_patch16_224 --dataset ImageNet_LT \
      --epoch 35 \
      --lr 0.01 \
      --lr_CosineAnnealing 30 0 \
      --batch_size 288 \
      --train_add_strong \
      --useWhatModal rawImg \
      --add_name get_Ts_image_ViT

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch vit_base_patch16_224 --dataset ImageNet_LT \
      --epoch 35 \
      --lr 0.01 \
      --lr_CosineAnnealing 30 0 \
      --batch_size 288 \
      --train_add_weak \
      --useWhatModal rawImg \
      --add_name get_Tw_image_ViT

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset ImageNet_LT \
      --epoch 120 \
      --lr 0.3 \
      --lr_steps 75 100 0 \
      --train_add_weak \
      --batch_size 1024 \
      --useWhatModal image_textWordNet \
      --add_name get_Tw_image_textWordNet_V2_MLP


### for scene:

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch vit_base_patch16_224 --dataset scene \
      --epoch 61 \
      --lr 0.001 \
      --lr_steps 25 40 60 \
      --batch_size 256 \
      --train_add_strong \
      --useWhatModal rawImg \
      --add_name get_Ts_image_ViT

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch vit_base_patch16_224 --dataset scene \
      --epoch 61 \
      --lr 0.001 \
      --lr_steps 25 40 60 \
      --batch_size 256 \
      --train_add_weak \
      --useWhatModal rawImg \
      --add_name get_Tw_image_ViT

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset scene \
      --epoch 200 \
      --lr 0.1 \
      --lr_steps 120 150 0 \
      --train_add_strong  \
      --batch_size 1024 \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet_V2_MLP


### for UTKFace:

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch vit_base_patch16_224 --dataset UTKFace \
      --epoch 61 \
      --lr 0.001 \
      --lr_steps 25 40 60 \
      --batch_size 256 \
      --train_add_strong \
      --useWhatModal rawImg \
      --add_name get_Ts_image_ViT

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch vit_base_patch16_224 --dataset UTKFace \
      --epoch 61 \
      --lr 0.001 \
      --lr_steps 25 40 60 \
      --batch_size 256 \
      --train_add_weak \
      --useWhatModal rawImg \
      --add_name get_Tw_image_ViT

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset UTKFace \
      --epoch 200 \
      --lr 0.1 \
      --lr_steps 120 150 0 \
      --train_add_strong  \
      --batch_size 1024 \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet_V2_MLP


### for CIFAR100_imb100:

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch vit_base_patch16_224 --dataset CIFAR100_imb100 \
      --epoch 61 \
      --lr 0.01 \
      --lr_steps 25 40 60 \
      --batch_size 256 \
      --train_add_strong \
      --useWhatModal rawImg \
      --add_name get_Ts_image_ViT

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch vit_base_patch16_224 --dataset CIFAR100_imb100 \
      --epoch 61 \
      --lr 0.01 \
      --lr_steps 25 40 60 \
      --batch_size 256 \
      --train_add_weak \
      --useWhatModal rawImg \
      --add_name get_Tw_image_ViT

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset CIFAR100_imb100 \
      --epoch 500 \
      --lr 0.3 \
      --lr_steps 250 350 450 \
      --train_add_strong  \
      --batch_size 1024 \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet_V2_MLP


### for ImageNet-mini:

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch vit_base_patch16_224 --dataset ImageNet-mini \
      --epoch 61 \
      --lr 0.001 \
      --lr_steps 25 40 60 \
      --batch_size 256 \
      --train_add_strong \
      --useWhatModal rawImg \
      --add_name get_Ts_image_ViT

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch vit_base_patch16_224 --dataset ImageNet-mini \
      --epoch 61 \
      --lr 0.001 \
      --lr_steps 25 40 60 \
      --batch_size 256 \
      --train_add_weak \
      --useWhatModal rawImg \
      --add_name get_Tw_image_ViT

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset ImageNet-mini \
      --epoch 30 \
      --lr 0.1 \
      --lr_steps 5 10 20 \
      --train_add_strong  \
      --batch_size 1024 \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet_V2_MLP



+++++++++++++++++++++++++++


### Distill the Student:



+++++++++++++++++++++++++++

### for CIFAR100:

#CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2.py --mode distil --arch vit_base_patch32_224 --dataset CIFAR100 --teacher_num 3 \
#   --epoch 300 --temp 3.0 --alpha 0.6 \
#   --lr 0.001 \
#   --lr_steps 190 195 0 \
#   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
#   --add_name T1s_T1w_T4sImgCATwordnetV2_Ss_ViT

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 300 400 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnetV2MLP_Ss_ViT

# Use the setup adopted by HuggingFace baseline:
#CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_ViT_regularizerV2.py --mode distil --arch vit_base_patch32_224 --dataset CIFAR100 --teacher_num 3 \
#   --epoch 300 --temp 3.0 --alpha 0.6 \
#   --lr 0.003 \
#   --lr_CosineAnnealing 300 0 \
#   --batch_size 1024 \
#   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
#   --add_name T1s_T1w_T4sImgCATwordnetV2MLP_Ss_ViT

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset CIFAR100 --teacher_num 2 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --S_add_strong \
   --add_name T1s_T1w_Ss_ViT

# Use the setup adopted by HuggingFace baseline:
#CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2.py --mode distil --arch vit_base_patch32_224 --dataset CIFAR100 --teacher_num 2 \
#   --epoch 300 --temp 3.0 --alpha 0.6 \
#   --lr 0.003 \
#   --lr_CosineAnnealing 300 0 \
#   --batch_size 1024 \
#   --T1_add_strong --T1_add_weak --S_add_strong \
#   --add_name T1s_T1w_Ss_ViT 


### for ImageNet:

#CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet --teacher_num 3 \
#   --epoch 60 --temp 3.0 --alpha 0.6 \
#   --lr 0.1 \
#   --lr_CosineAnnealing 60 0 \
#   --batch_size 512 \
#   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
#   --add_name T1s_T1w_T4sImgCATwordnetV2MLP_Ss_ViT

# Use the setup adopted by HuggingFace baseline:
CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet --teacher_num 3 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.0003 \
   --lr_CosineAnnealing 300 0 \
   --batch_size 768 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnetV2MLP_Ss_ViT


#CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_ViT_regularizerV2.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet --teacher_num 2 \
#   --epoch 60 --temp 3.0 --alpha 0.6 \
#   --lr 0.1 \
#   --lr_CosineAnnealing 60 0 \
#   --batch_size 512 \
#   --T1_add_strong --T1_add_weak --S_add_strong \
#   --add_name T1s_T1w_Ss_ViT \
#   --next_continue runs/ImageNet_distil_T1s_T1w_Ss_ViT/run-2-epoch70/#checkpoint_bestAcc1.pth.tar

# Use the setup adopted by HuggingFace baseline:
CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_ViT_regularizerV2.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet --teacher_num 2 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.0003 \
   --lr_CosineAnnealing 300 0 \
   --batch_size 768 \
   --T1_add_strong --T1_add_weak --S_add_strong \
   --add_name T1s_T1w_Ss_ViT 
#   --next_continue runs/ImageNet_distil_T1s_T1w_Ss_ViT/run-3-epoch60/checkpoint_bestAcc1.pth.tar


### for ImageNet-LT:

# Use the setup adopted by HuggingFace baseline:
#CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_ViT_regularizerV2.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet_LT --teacher_num 3 \
#   --epoch 1000 --temp 3.0 --alpha 0.6 \
#   --lr 0.05 \
#   --lr_CosineAnnealing 1000 0 \
#   --batch_size 1024 \
#   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
#   --add_name T1s_T1w_T4sImgCATwordnetV2MLP_Ss_ViT

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet_LT --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 300 400 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnetV2MLP_Ss_ViT

# Use the setup adopted by HuggingFace baseline:
#CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet_LT --teacher_num 2 \
#   --epoch 1000 --temp 3.0 --alpha 0.6 \
#   --lr 0.05 \
#   --lr_CosineAnnealing 1000 0 \
#   --batch_size 1024 \
#   --T1_add_strong --T1_add_weak --S_add_strong \
#   --add_name T1s_T1w_Ss_ViT 

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet_LT --teacher_num 2 \
   --epoch 240 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 210 220 230 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --S_add_strong \
   --add_name T1s_T1w_Ss_ViT


### for scene:

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset scene --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr 0.001 \
   --lr_steps 300 400 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnetV2MLP_Ss_ViT

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset scene --teacher_num 2 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.001 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --S_add_strong \
   --add_name T1s_T1w_Ss_ViT



### for UTKFace:

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset UTKFace --teacher_num 3 \
   --epoch 600 --temp 3.0 --alpha 0.6 \
   --lr 0.001 \
   --lr_steps 300 450 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnetV2MLP_Ss_ViT

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset UTKFace --teacher_num 2 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.001 \
   --lr_steps 200 250 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --S_add_strong \
   --add_name T1s_T1w_Ss_ViT


### for CIFAR100_imb100:

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 300 400 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnetV2MLP_Ss_ViT

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset CIFAR100_imb100 --teacher_num 2 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 160 165 170 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --S_add_strong \
   --add_name T1s_T1w_Ss_ViT


### for ImageNet-mini:

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 300 400 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnetV2MLP_Ss_ViT

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 2 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --S_add_strong \
   --add_name T1s_T1w_Ss_ViT

# Use the setup adopted by HuggingFace baseline:
#CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_ViT_regularizerV2.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 3 \
#   --epoch 500 --temp 3.0 --alpha 0.6 \
#   --lr 0.0003 \
#   --lr_CosineAnnealing 500 0 \
#   --batch_size 768 \
#   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
#   --add_name T1s_T1w_T4sImgCATwordnetV2MLP_Ss_ViT

# Use the setup adopted by HuggingFace baseline:
#CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 2 \
#   --epoch 500 --temp 3.0 --alpha 0.6 \
#   --lr 0.0003 \
#   --lr_CosineAnnealing 500 0 \
#   --batch_size 768 \
#   --T1_add_strong --T1_add_weak --S_add_strong \
#   --add_name T1s_T1w_Ss_ViT









CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_regularizerV2.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCAT10gt90wordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCAT10gt90wordnetV2_Ss




## for rebuttal:

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_regularizerV2.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet_mlp --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnetV2_Ss_mlpTx

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_regularizerV2.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet_vit --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnetV2_Ss_vitTx








CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATWNtreeV1 --S_add_strong \
   --add_name T1s_T1w_T4sImgCATWNtreeV1_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCAT10gt90WNtreeV1 --S_add_strong \
   --add_name T1s_T1w_T4sImgCAT10gt90WNtreeV1_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCAT1gt99WNtreeV1 --S_add_strong \
   --add_name T1s_T1w_T4sImgCAT1gt99WNtreeV1_Ss






CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATWNtreeV2 --S_add_strong \
   --add_name T1s_T1w_T4sImgCATWNtreeV2_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCAT10gt90WNtreeV2 --S_add_strong \
   --add_name T1s_T1w_T4sImgCAT10gt90WNtreeV2_Ss




++++++++++++++++++++






# #####################



















