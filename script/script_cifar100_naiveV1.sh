### Train the Teacher:

python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_weak  \
      --useWhatModal image \
      --add_name get_Tw_image

python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image \
      --add_name get_Ts_imageCLIP

python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_weak  \
      --useWhatModal textGT \
      --add_name get_Tw_textGT

python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_weak  \
      --useWhatModal textWordNet \
      --add_name get_Tw_textWordNet      

python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal textWordNet \
      --add_name get_Ts_textWordNet

python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_weak  \
      --useWhatModal image_textGT \
      --add_name get_Tw_image_textGT

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textGT \
      --add_name get_Ts_image_textGT

python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_weak  \
      --useWhatModal image_textWordNet \
      --add_name get_Tw_image_textWordNet  

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet

+++++++++++++++++++++++++++

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal text_70gt30wordnet \
      --add_name get_Ts_text_70gt30wordnet


CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal text_30gt70wordnet \
      --add_name get_Ts_text_30gt70wordnet


CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal text_80gt20wordnet \
      --add_name get_Ts_text_80gt20wordnet

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal text_50gt50wordnet \
      --add_name get_Ts_text_50gt50wordnet

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal text_20gt80wordnet \
      --add_name get_Ts_text_20gt80wordnet

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong \
      --useWhatModal textGT \
      --noise_percent 20 \
      --add_name get_Ts_textGT_20noise

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal text_50gt50wordnet \
      --add_name get_Ts_text_50gt50wordnet

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal text_90gt10wordnet \
      --add_name get_Ts_text_90gt10wordnet

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal text_10gt90wordnet \
      --add_name get_Ts_text_10gt90wordnet

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal text_95gt5wordnet \
      --add_name get_Ts_text_95gt5wordnet

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal text_5gt95wordnet \
      --add_name get_Ts_text_5gt95wordnet

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal text_1gt99wordnet \
      --add_name get_Ts_text_1gt99wordnet

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal text_3gt97wordnet \
      --add_name get_Ts_text_3gt97wordnet

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal text_0gt100wordnet \
      --add_name get_Ts_text_0gt100wordnet


CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal text_60gt40wordnet \
      --add_name get_Ts_text_60gt40wordnet

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal text_40gt60wordnet \
      --add_name get_Ts_text_40gt60wordnet


CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_text_10gt90wordnet \
      --add_name get_Ts_image_text_10gt90wordnet

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_text_1gt99wordnet \
      --add_name get_Ts_image_text_1gt99wordnet

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_text_20gt80wordnet \
      --add_name get_Ts_image_text_20gt80wordnet



CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong \
      --useWhatModal image_textGT \
      --noise_percent 80 \
      --add_name get_Ts_image_textGT_80noise

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong \
      --useWhatModal image_textGT \
      --noise_percent 20 \
      --add_name get_Ts_image_textGT_20noise


CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong \
      --useWhatModal image_textGT \
      --noise_percent 50 \
      --add_name get_Ts_image_textGT_50noise


CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong \
      --useWhatModal image_textGT \
      --noise_percent 100 \
      --add_name get_Ts_image_textGT_100noise


CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 200 \
      --lr_steps 100 140 180 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet
CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 400 \
      --lr_steps 250 300 350 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet \
      --next_continue runs/CIFAR100_train_get_Ts_image_textWordNet/run-2-epoch200/checkpoint_step1.pth.tar


CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 400 \
      --lr_steps 250 300 350 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal text_0gt100wordnet \
      --add_name get_Ts_text_0gt100wordnet





CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWNtreeV1 \
      --add_name get_Ts_image_textWNtreeV1

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_text_10gt90WNtreeV1 \
      --add_name get_Ts_image_text_10gt90WNtreeV1

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_text_1gt99WNtreeV1 \
      --add_name get_Ts_image_text_1gt99WNtreeV1



CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWNtreeV2 \
      --add_name get_Ts_image_textWNtreeV2

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_text_10gt90WNtreeV2 \
      --add_name get_Ts_image_text_10gt90WNtreeV2






CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_text_80gt20wordnet \
      --add_name get_Ts_image_text_80gt20wordnet

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_text_50gt50wordnet \
      --add_name get_Ts_image_text_50gt50wordnet




+++++++++++++++++++++++++++


### Distill the Student:
python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 200 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T1_add_weak --T2_add_weak_CLIPimg --T3_add_gt --S_add_weak \
   --add_name T1w_T2CLIPimg_T3gt_Sw 

python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 200 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T1_add_weak --T2_add_weak_CLIPimg --T3_add_wordnet --S_add_weak \
   --add_name T1w_T2CLIPimg_T3wordnet_Sw 


CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 200 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T1_add_strong --T4_add_weak_imgCATgt --T4_add_weak_imgCATwordnet --S_add_strong \
   --add_name T1s_T4wImgCATgt_T4wImgCATwordnet_Ss
CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 250 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T1_add_strong --T4_add_strong_imgCATgt --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T4sImgCATgt_T4sImgCATwordnet_Ss 
CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 2 \
   --epoch 250 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T1_add_strong --T4_add_strong_imgCATgt --S_add_strong \
   --add_name T1s_T4sImgCATgt_Ss 

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 200 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T1_add_strong --T3_add_gt --T4_add_weak_imgCATgt --S_add_strong \
   --add_name T1s_T3gt_T4wImgCATgt_Ss 



CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 400 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnet_Ss \
   --next_continue runs/CIFAR100_distil_T1s_T1w_T4sImgCATwordnet_Ss/run-2-epoch300/checkpoint_each-epoch.pth.tar
CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 400 --temp 3.0 --alpha 0.6 \
   --lr_steps 250 350 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnet_Ss 
===================
CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnet_Ss \
   --next_continue runs/CIFAR100_distil_T1s_T1w_T4sImgCATwordnet_Ss/run-4-epoch400/checkpoint_step1.pth.tar
=====================
CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 400 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 300 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnet_Ss \
   --next_continue runs/CIFAR100_distil_T1s_T1w_T4sImgCATwordnet_Ss/run-1-epoch250/checkpoint_step1.pth.tar

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T4sImgCATwordnet_Ss \
   --next_continue runs/CIFAR100_distil_T4sImgCATwordnet_Ss/run-1-epoch200/checkpoint_each-epoch.pth.tar




CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 300 --temp 3.0 --alpha 0.6 --lr 0.2 \
   --lr_steps 190 240 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnet_Ss \
   --next_continue runs/CIFAR100_distil_T1s_T1w_T4sImgCATwordnet_Ss/run-7-epoch200/checkpoint_step1.pth.tar



CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 600 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnet_Ss \
   --next_continue runs/CIFAR100_distil_T1s_T1w_T4sImgCATwordnet_Ss/run-6-epoch500/checkpoint_bestAcc1.pth.tar



CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 2 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T4_add_strong_imgCATwordnet --T1_add_weak --S_add_strong \
   --add_name T4sImgCATwordnet_T1w_Ss






-------------this!:
CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T3_add_wordnet --S_add_strong \
   --add_name T3wordnet_Ss

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T3_add_wordnet --S_add_strong \
   --add_name T3wordnet_Ss \
   --next_continue runs/CIFAR100_distil_T3wordnet_Ss/run-1-epoch300/checkpoint_step1.pth.tar




CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T2_add_strong_CLIPimg --S_add_strong \
   --add_name T2sCLIPimg_Ss

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 2 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T2_add_strong_CLIPimg --T3_add_wordnet --S_add_strong \
   --add_name T2sCLIPimg_T3wordnet_Ss

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T1_add_strong --T2_add_strong_CLIPimg --T3_add_wordnet --S_add_strong \
   --add_name T1s_T2sCLIPimg_T3wordnet_Ss

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T3_add_gt --S_add_strong \
   --add_name T3gt_Ss



+++++++++++++++++++++++++++

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T3_add_70gt30wordnet --S_add_strong \
   --add_name T3-70gt30wordnet_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T3_add_30gt70wordnet --S_add_strong \
   --add_name T3-30gt70wordnet_Ss

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T3_add_80gt20wordnet --S_add_strong \
   --add_name T3-80gt20wordnet_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T3_add_20gt80wordnet --S_add_strong \
   --add_name T3-20gt80wordnet_Ss \
   --next_continue runs/CIFAR100_distil_T3-20gt80wordnet_Ss/run-2-epoch300/checkpoint_step1.pth.tar



CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T3_add_gt_noise --S_add_strong \
   --noise_percent 20 \
   --add_name T3gt20noise_Ss

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T3_add_90gt10wordnet --S_add_strong \
   --add_name T3-90gt10wordnet_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T3_add_10gt90wordnet --S_add_strong \
   --add_name T3-10gt90wordnet_Ss

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T3_add_95gt5wordnet --S_add_strong \
   --add_name T3-95gt5wordnet_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T3_add_5gt95wordnet --S_add_strong \
   --add_name T3-5gt95wordnet_Ss

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T3_add_50gt50wordnet --S_add_strong \
   --add_name T3-50gt50wordnet_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T3_add_1gt99wordnet --S_add_strong \
   --add_name T3-1gt99wordnet_Ss


CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T3_add_3gt97wordnet --S_add_strong \
   --add_name T3-3gt97wordnet_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T3_add_0gt100wordnet --S_add_strong \
   --add_name T3-0gt100wordnet_Ss



CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T3_add_60gt40wordnet --S_add_strong \
   --add_name T3-60gt40wordnet_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T3_add_40gt60wordnet --S_add_strong \
   --add_name T3-40gt60wordnet_Ss


CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCAT10gt90wordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCAT10gt90wordnet_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCAT1gt99wordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCAT1gt99wordnet_Ss



CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATgt --S_add_strong \
   --add_name T1s_T1w_T4sImgCATgt_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCAT20gt80wordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCAT20gt80wordnet_Ss




## for Table 2 in CLIP-KD paper:

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T4_add_strong_imgCATgt --S_add_strong \
   --add_name T4sImgCATgt_Ss

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T4_add_strong_imgCAT20gt80wordnet --S_add_strong \
   --add_name T4sImgCAT20gt80wordnet_Ss

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T4_add_strong_imgCAT10gt90wordnet --S_add_strong \
   --add_name T4sImgCAT10gt90wordnet_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T4_add_strong_imgCAT1gt99wordnet --S_add_strong \
   --add_name T4sImgCAT1gt99wordnet_Ss

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T4sImgCATwordnet_Ss_ep500



CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T4_add_strong_imgCAT80gt20wordnet --S_add_strong \
   --add_name T4sImgCAT80gt20wordnet_Ss

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T4_add_strong_imgCAT50gt50wordnet --S_add_strong \
   --add_name T4sImgCAT50gt50wordnet_Ss


CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCAT80gt20wordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCAT80gt20wordnet_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCAT50gt50wordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCAT50gt50wordnet_Ss






CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATgtNoise --S_add_strong \
   --noise_percent 80 \
   --add_name T1s_T1w_T4sImgCAT20gt80noise_Ss
#T3_add_gt_noise

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATgtNoise --S_add_strong \
   --noise_percent 20 \
   --add_name T1s_T1w_T4sImgCAT80gt20noise_Ss


CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATgtNoise --S_add_strong \
   --noise_percent 100 \
   --add_name T1s_T1w_T4sImgCAT0gt100noise_Ss



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




+++++++++++++++++++++++++++



##### USE this:

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T4sImgCATwordnet_Ss

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 2 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T4_add_strong_imgCATwordnet --T1_add_weak --S_add_strong \
   --add_name T4sImgCATwordnet_T1w_Ss

#run6: gives the best!:
CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnet_Ss







# #####################



















