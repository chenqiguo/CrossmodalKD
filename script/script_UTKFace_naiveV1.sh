### Train the Teacher:




CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset UTKFace \
      --epoch 61 --lr 0.001 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet_V1





CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset UTKFace --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 --lr 0.001 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnetV1_Ss


+++++++++++++++++++++++++++

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_regularizerV2.py --mode train --arch resnet50 --dataset UTKFace \
      --epoch 61 --lr 0.001 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal rawImg \
      --add_name get_Ts

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_regularizerV2.py --mode train --arch resnet50 --dataset UTKFace \
      --epoch 61 --lr 0.001 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_weak  \
      --useWhatModal rawImg \
      --add_name get_Tw


CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_regularizerV2.py --mode train --arch resnet50 --dataset UTKFace \
      --epoch 61 --lr 0.001 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet_V2



#CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_regularizerV2.py --mode train --arch resnet50 --dataset CIFAR100 \
#      --epoch 61 \
#      --lr_steps 25 40 60 \
#      --batch_size 128 \
#      --train_add_strong  \
#      --useWhatModal image_text_10gt90wordnet \
#      --add_name get_Ts_image_text_10gt90wordnet_V2









+++++++++++++++++++++++++++


### Distill the Student:



+++++++++++++++++++++++++++


#CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_regularizerV2.py --mode distil --arch resnet18 --dataset UTKFace --teacher_num 3 \
#   --epoch 200 --temp 3.0 --alpha 0.6 --lr 0.001 \
#   --lr_steps 190 195 0 \
#   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
#   --add_name T1s_T1w_T4sImgCATwordnetV2_Ss

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_regularizerV2.py --mode distil --arch resnet18 --dataset UTKFace --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 --lr 0.001 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnetV2_Ss

# T1s_T2w_T3s_Ss
CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_regularizerV2.py --mode distil --arch resnet18 --dataset UTKFace --teacher_num 3 \
   --epoch 200 --temp 3.0 --alpha 0.6 --lr 0.001 \
   --lr_steps 190 195 0 \
   --T1_add_strong --T1_add_weak --T2_add_strong --S_add_strong \
   --add_name T1s_T1w_T2s_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_regularizerV2.py --mode distil --arch resnet18 --dataset UTKFace --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 --lr 0.001 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T2_add_strong --S_add_strong \
   --add_name T1s_T1w_T2s_Ss




#CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_regularizerV2.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
#   --epoch 500 --temp 3.0 --alpha 0.6 \
#   --lr_steps 300 400 0 \
#   --T1_add_strong --T1_add_weak --T4_add_strong_imgCAT10gt90wordnet --S_add_strong \
#   --add_name T1s_T1w_T4sImgCAT10gt90wordnetV2_Ss












++++++++++++++++++++






# #####################



















