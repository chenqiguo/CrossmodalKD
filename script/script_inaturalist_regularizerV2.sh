### Train the Teacher:



+++++++++++++++++++++++++++


CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_regularizerV2.py --mode train --arch resnet50 --dataset inaturalist \
      --epoch 45 \
      --lr_steps 15 30 45 \
      --batch_size 512 \
      --train_add_strong  \
      --useWhatModal rawImg \
      --add_name get_Ts \
      --next_continue /home/ps/scratch/CLIP_KD/runs/inaturalist_train_get_Ts/run-1-epoch61/checkpoint_bestAcc1.pth.tar

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_regularizerV2.py --mode train --arch resnet50 --dataset inaturalist \
      --epoch 45 \
      --lr_steps 15 30 45 \
      --batch_size 512 \
      --train_add_weak  \
      --useWhatModal rawImg \
      --add_name get_Tw \
      --next_continue /home/ps/scratch/CLIP_KD/runs/inaturalist_train_get_Tw/run-1-epoch61/checkpoint_bestAcc1.pth.tar



CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_regularizerV2.py --mode train --arch resnet50 --dataset inaturalist \
      --epoch 100 \
      --lr_steps 30 60 90 \
      --batch_size 1024 \
      --train_add_weak  \
      --useWhatModal image_textWordNet \
      --add_name get_Tw_image_textWordNet_V2

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset inaturalist \
      --epoch 45 \
      --lr_steps 15 30 45 \
      --batch_size 512 \
      --train_add_weak  \
      --useWhatModal image_textWordNet \
      --add_name get_Tw_image_textWordNet











CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_regularizerV2.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_text_10gt90wordnet \
      --add_name get_Ts_image_text_10gt90wordnet_V2







## for rebuttal:

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_regularizerV2.py --mode train --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --TxArch mlp \
      --add_name get_Ts_image_textWordNet_V2_mlp

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_regularizerV2.py --mode train --dataset CIFAR100 \
      --epoch 140 \
      --lr_steps 120 130 0 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --TxArch vit \
      --add_name get_Ts_image_textWordNet_V2_vit


+++++++++++++++++++++++++++


### Distill the Student:



+++++++++++++++++++++++++++


CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_regularizerV2.py --mode distil --arch resnet18 --dataset inaturalist --teacher_num 2 \
   --epoch 200 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --S_add_strong \
   --add_name T1s_T2w_Ss


CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_regularizerV2.py --mode distil --arch resnet18 --dataset inaturalist --teacher_num 3 \
   --epoch 200 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --T4_add_weak_imgCATwordnet --S_add_weak \
   --add_name T1s_T1w_T4wImgCATwordnetV2_Sw_v2_invLoss

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset inaturalist --teacher_num 3 \
   --epoch 200 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --T4_add_weak_imgCATwordnet --S_add_weak \
   --add_name T1s_T1w_T4wImgCATwordnet_Sw_v1






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



















