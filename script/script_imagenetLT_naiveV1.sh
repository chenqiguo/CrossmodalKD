### Train the Teacher:

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset ImageNet_LT \
      --epoch 60 \
      --lr_steps 35 50 0 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet


CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset ImageNet_LT \
      --epoch 60 \
      --lr_steps 35 50 0 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal textGT \
      --add_name get_Ts_textGT


CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset ImageNet_LT \
      --epoch 60 \
      --lr_steps 35 50 0 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal textWordNet \
      --add_name get_Ts_textWordNet

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset ImageNet_LT \
      --epoch 100 \
      --lr_steps 70 90 0 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal textWordNet \
      --add_name get_Ts_textWordNet \
      --next_continue runs/ImageNet_LT_train_get_Ts_textWordNet/run-1-epoch60/checkpoint_step1.pth.tar

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset ImageNet_LT \
      --epoch 170 \
      --lr_steps 100 125 150 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal textWordNet \
      --add_name get_Ts_textWordNet \
      --next_continue runs/ImageNet_LT_train_get_Ts_textWordNet/run-2-epoch100/checkpoint_step1.pth.tar


##### USE this:
CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100_imb100 \
      --epoch 200 \
      --lr_steps 50 100 150 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset ImageNet_LT \
      --epoch 170 \
      --lr_steps 100 125 150 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal textWordNet \
      --add_name get_Ts_textWordNet \
      --next_continue runs/ImageNet_LT_train_get_Ts_textWordNet/run-2-epoch100/checkpoint_step1.pth.tar




### Distill the Student:

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T4sImgCATwordnet_Ss

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 200 240 280 \
   --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T4sImgCATwordnet_Ss




CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 2 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T4_add_strong_imgCATwordnet --T1_add_weak --S_add_strong \
   --add_name T4sImgCATwordnet_T1w_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 2 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 200 240 280 \
   --T4_add_strong_imgCATwordnet --T1_add_weak --S_add_strong \
   --add_name T4sImgCATwordnet_T1w_Ss


CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 300 --temp 3.0 --alpha 0.7 \
   --lr_steps 200 240 280 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnet_Ss



CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 3 \
   --epoch 165 --temp 3.0 --alpha 0.6 \
   --lr_steps 150 155 160 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnet_Ss






CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 3 \
   --epoch 200 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 170 180 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnet_Ss \
   --next_continue runs/ImageNet_LT_distil_T1s_T1w_T4sImgCATwordnet_Ss/run-1-epoch165/checkpoint_step1.pth.tar

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 3 \
   --epoch 200 --temp 3.0 --alpha 0.6 \
   --lr_steps 175 185 195 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnet_Ss \
   --next_continue runs/ImageNet_LT_distil_T1s_T1w_T4sImgCATwordnet_Ss/run-2-epoch200/checkpoint_step1.pth.tar

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 3 \
   --epoch 215 --temp 3.0 --alpha 0.6 \
   --lr_steps 185 195 205 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnet_Ss \
   --next_continue runs/ImageNet_LT_distil_T1s_T1w_T4sImgCATwordnet_Ss/run-3-epoch200/checkpoint_step1.pth.tar

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 3 \
   --epoch 230 --temp 3.0 --alpha 0.6 \
   --lr_steps 200 210 220 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnet_Ss \
   --next_continue runs/ImageNet_LT_distil_T1s_T1w_T4sImgCATwordnet_Ss/run-4-epoch215/checkpoint_step1.pth.tar
   
   
+++++++++++++++++++++++++++

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 1 \
   --epoch 240 --temp 3.0 --alpha 0.6 \
   --lr_steps 210 220 230 \
   --T3_add_gt --S_add_strong \
   --add_name T3gt_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 1 \
   --epoch 240 --temp 3.0 --alpha 0.6 \
   --lr_steps 210 220 230 \
   --T3_add_wordnet --S_add_strong \
   --add_name T3wordnet_Ss

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 2 \
   --epoch 240 --temp 3.0 --alpha 0.6 \
   --lr_steps 210 220 230 \
   --T4_add_strong_imgCATwordnet --T1_add_weak --S_add_strong \
   --add_name T4sImgCATwordnet_T1w_Ss


+++++++++++++++++++++++++++





##### USE this:

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T4sImgCATwordnet_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 2 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 200 240 280 \
   --T4_add_strong_imgCATwordnet --T1_add_weak --S_add_strong \
   --add_name T4sImgCATwordnet_T1w_Ss

#run6: gives the best!:
CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 3 \
   --epoch 240 --temp 3.0 --alpha 0.6 \
   --lr_steps 210 220 230 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnet_Ss \
   --next_continue runs/ImageNet_LT_distil_T1s_T1w_T4sImgCATwordnet_Ss/run-5-epoch230/checkpoint_step1.pth.tar



CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_regularizerV2_embdDistill.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 3 \
   --epoch 240 --temp 3.0 --alpha 0.6 \
   --lr_steps 210 220 230 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnet_Ss_embd



# #####################



















