### Train the Teacher:

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image \
      --add_name get_Ts_imageCLIP

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100_imb100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image \
      --add_name get_Ts_imageCLIP

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset scene \
      --epoch 61 --lr 0.001 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image \
      --add_name get_Ts_imageCLIP

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset UTKFace \
      --epoch 61 --lr 0.001 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image \
      --add_name get_Ts_imageCLIP

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset ImageNet \
      --epoch 35 \
      --lr_CosineAnnealing 30 0 \
      --batch_size 1024 \
      --train_add_strong  \
      --useWhatModal image \
      --add_name get_Ts_imageCLIP

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset ImageNet_LT \
      --epoch 60 \
      --lr_steps 35 50 0 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image \
      --add_name get_Ts_imageCLIP



----------------------------

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textGT \
      --add_name get_Ts_image_textGT

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset ImageNet \
      --epoch 35 \
      --lr_CosineAnnealing 30 0 \
      --batch_size 1024 \
      --train_add_strong  \
      --useWhatModal image_textGT \
      --add_name get_Ts_image_textGT

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset CIFAR100_imb100 \
      --epoch 61 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textGT \
      --add_name get_Ts_image_textGT

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset scene \
      --epoch 61 --lr 0.001 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textGT \
      --add_name get_Ts_image_textGT

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset UTKFace \
      --epoch 61 --lr 0.001 \
      --lr_steps 25 40 60 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textGT \
      --add_name get_Ts_image_textGT

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode train --arch resnet50 --dataset ImageNet_LT \
      --epoch 60 \
      --lr_steps 35 50 0 \
      --batch_size 128 \
      --train_add_strong  \
      --useWhatModal image_textGT \
      --add_name get_Ts_image_textGT


### Distill the Student:

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 190 195 0 \
   --T2_add_strong_CLIPimg --S_add_strong \
   --add_name T2sCLIPimg_Ss

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T2_add_strong_CLIPimg --S_add_strong \
   --add_name T2sCLIPimg_Ss

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset scene --teacher_num 1 \
   --epoch 500 --temp 3.0 --alpha 0.6 --lr 0.001 \
   --lr_steps 300 400 0 \
   --T2_add_strong_CLIPimg --S_add_strong \
   --add_name T2sCLIPimg_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset UTKFace --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 --lr 0.001 \
   --lr_steps 200 250 0 \
   --T2_add_strong_CLIPimg --S_add_strong \
   --add_name T2sCLIPimg_Ss

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset ImageNet --teacher_num 1 \
   --epoch 60 --temp 3.0 --alpha 0.6 \
   --lr_CosineAnnealing 60 0 \
   --batch_size 512 \
   --T2_add_strong_CLIPimg --S_add_strong \
   --add_name T2sCLIPimg_Ss

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 1 \
   --epoch 240 --temp 3.0 --alpha 0.6 \
   --lr_steps 210 220 230 \
   --T2_add_strong_CLIPimg --S_add_strong \
   --add_name T2sCLIPimg_Ss


----------------------------

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 1 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T4_add_strong_imgCATgt --S_add_strong \
   --add_name T4sImgCATgt_Ss

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100 --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr_steps 300 400 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATgt --S_add_strong \
   --add_name T1s_T1w_T4sImgCATgt_Ss


CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T4_add_strong_imgCATgt --S_add_strong \
   --add_name T4sImgCATgt_Ss

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset CIFAR100_imb100 --teacher_num 3 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr_steps 160 165 170 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATgt --S_add_strong \
   --add_name T1s_T1w_T4sImgCATgt_Ss



CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset scene --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 --lr 0.001 \
   --lr_steps 160 165 170 \
   --T4_add_strong_imgCATgt --S_add_strong \
   --add_name T4sImgCATgt_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset scene --teacher_num 3 \
   --epoch 300 --temp 3.0 --alpha 0.6 --lr 0.001 \
   --lr_steps 160 165 170 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATgt --S_add_strong \
   --add_name T1s_T1w_T4sImgCATgt_Ss


CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset UTKFace --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 --lr 0.001 \
   --lr_steps 200 250 0 \
   --T4_add_strong_imgCATgt --S_add_strong \
   --add_name T4sImgCATgt_Ss

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset UTKFace --teacher_num 3 \
   --epoch 300 --temp 3.0 --alpha 0.6 --lr 0.001 \
   --lr_steps 200 250 0 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATgt --S_add_strong \
   --add_name T1s_T1w_T4sImgCATgt_Ss



CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset ImageNet --teacher_num 1 \
   --epoch 60 --temp 3.0 --alpha 0.6 \
   --lr_CosineAnnealing 60 0 \
   --batch_size 512 \
   --T4_add_strong_imgCATgt --S_add_strong \
   --add_name T4sImgCATgt_Ss

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset ImageNet --teacher_num 3 \
   --epoch 60 --temp 3.0 --alpha 0.6 \
   --lr_CosineAnnealing 60 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATgt --S_add_strong \
   --add_name T1s_T1w_T4sImgCATgt_Ss


CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 1 \
   --epoch 240 --temp 3.0 --alpha 0.6 \
   --lr_steps 210 220 230 \
   --T4_add_strong_imgCATgt --S_add_strong \
   --add_name T4sImgCATgt_Ss

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_naiveV1.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 3 \
   --epoch 240 --temp 3.0 --alpha 0.6 \
   --lr_steps 210 220 230 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATgt --S_add_strong \
   --add_name T1s_T1w_T4sImgCATgt_Ss



# #####################



















