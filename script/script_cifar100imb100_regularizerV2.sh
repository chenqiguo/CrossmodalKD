### Train the Teacher:

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100_imb100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100_imb100 \
      --epoch 100 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet
      
CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100_imb100 \
      --epoch 100 \
      --lr_steps 40 80 0 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100_imb100 \
      --epoch 100 \
      --lr_steps 25 50 75 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100_imb100 \
      --epoch 100 \
      --lr_steps 8 23 50 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet \
      --next_continue runs/CIFAR100_imb100_train_get_Ts_image_textWordNet/run-4-epoch100/checkpoint_step1.pth.tar

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100_imb100 \
      --epoch 150 \
      --lr_steps 50 75 100 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet


##### USE this:
CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100_imb100 \
      --epoch 200 \
      --lr_steps 50 100 150 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet


CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_regularizerV2.py --mode train --arch resnet50 --dataset CIFAR100_imb100 \
      --epoch 200 \
      --lr_steps 50 100 150 \
      --batch_size 512 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet_V2





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
CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnet_Ss





CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_regularizerV2.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnetV2_Ss


CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_regularizerV2_embdDistill.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnetV2_Ss_embd




#CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
#   --epoch 300 --temp 3.0 --alpha 0.6 \
#   --lr_steps 160 165 170 \
#   --T1_add_strong --T2_add_strong --T4_add_strong_imgCATwordnet --S_add_strong \
#   --add_name T1s_T2s_T4sImgCATwordnet_Ss


CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 600 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 500 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnet_Ss


# #####################



















