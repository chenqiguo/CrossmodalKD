### Train the Teacher:



+++++++++++++++++++++++++++


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

#CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset ImageNet-mini \
#      --epoch 30 \
#      --lr 0.1 \
#      --lr_steps 5 10 20 \
#      --train_add_strong  \
#      --batch_size 1024 \
#      --useWhatModal image_textWordNet \
#      --add_name get_Ts_image_textWordNet_V2_MLP

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset ImageNet-mini \
      --epoch 100 \
      --lr 0.05 \
      --lr_steps 70 80 90 \
      --train_add_strong  \
      --batch_size 1024 \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet_V2_MLP


CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset ImageNet-mini \
      --epoch 100 \
      --lr 0.05 \
      --lr_steps 70 80 90 \
      --train_add_strong  \
      --batch_size 1024 \
      --useWhatModal textWordNet \
      --add_name get_Ts_textWordNet_V2_MLP



-----------------------

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset ImageNet-mini \
      --epoch 100 \
      --lr 0.05 \
      --lr_steps 70 80 90 \
      --batch_size 1024 \
      --train_add_strong  \
      --useWhatModal image_text_20gt80wordnet \
      --add_name get_Ts_image_text_20gt80wordnet_V2_MLP

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset ImageNet-mini \
      --epoch 100 \
      --lr 0.05 \
      --lr_steps 70 80 90 \
      --train_add_strong  \
      --batch_size 1024 \
      --useWhatModal text_20gt80wordnet \
      --add_name get_Ts_text_20gt80wordnet_V2_MLP



CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset ImageNet-mini \
      --epoch 30 \
      --lr 0.1 \
      --lr_steps 5 10 20 \
      --batch_size 1024 \
      --train_add_strong  \
      --useWhatModal image_text_20gt80wordnet \
      --add_name get_Ts_image_text_20gt80wordnet_V2_MLP

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset ImageNet-mini \
      --epoch 100 \
      --lr 0.05 \
      --lr_steps 70 80 90 \
      --batch_size 1024 \
      --train_add_strong  \
      --useWhatModal image_text_50gt50wordnet \
      --add_name get_Ts_image_text_50gt50wordnet_V2_MLP

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset ImageNet-mini \
      --epoch 100 \
      --lr 0.05 \
      --lr_steps 70 80 90 \
      --train_add_strong  \
      --batch_size 1024 \
      --useWhatModal text_50gt50wordnet \
      --add_name get_Ts_text_50gt50wordnet_V2_MLP


CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset ImageNet-mini \
      --epoch 100 \
      --lr 0.05 \
      --lr_steps 70 80 90 \
      --batch_size 1024 \
      --train_add_strong  \
      --useWhatModal image_text_80gt20wordnet \
      --add_name get_Ts_image_text_80gt20wordnet_V2_MLP

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset ImageNet-mini \
      --epoch 100 \
      --lr 0.05 \
      --lr_steps 70 80 90 \
      --train_add_strong  \
      --batch_size 1024 \
      --useWhatModal text_80gt20wordnet \
      --add_name get_Ts_text_80gt20wordnet_V2_MLP


CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset ImageNet-mini \
      --epoch 100 \
      --lr 0.05 \
      --lr_steps 70 80 90 \
      --batch_size 1024 \
      --train_add_strong  \
      --useWhatModal image_textGT \
      --add_name get_Ts_image_textGT_V2_MLP

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset ImageNet-mini \
      --epoch 100 \
      --lr 0.05 \
      --lr_steps 70 80 90 \
      --train_add_strong  \
      --batch_size 1024 \
      --useWhatModal text_100gt0wordnet \
      --add_name get_Ts_textGT_V2_MLP



-----------------------

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset ImageNet-mini \
      --epoch 100 \
      --lr 0.05 \
      --lr_steps 70 80 90 \
      --batch_size 1024 \
      --train_add_strong \
      --useWhatModal image_textGT \
      --noise_percent 20 \
      --add_name get_Ts_image_textGT_20noise_V2_MLP

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset ImageNet-mini \
      --epoch 100 \
      --lr 0.05 \
      --lr_steps 70 80 90 \
      --batch_size 1024 \
      --train_add_strong \
      --useWhatModal image_textGT \
      --noise_percent 100 \
      --add_name get_Ts_image_textGT_100noise_V2_MLP




CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset ImageNet-mini \
      --epoch 100 \
      --lr 0.05 \
      --lr_steps 70 80 90 \
      --batch_size 1024 \
      --train_add_strong \
      --useWhatModal image_textGT \
      --noise_percent 50 \
      --add_name get_Ts_image_textGT_50noise_V2_MLP

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_ViT_regularizerV2.py --mode train --arch MLP --dataset ImageNet-mini \
      --epoch 100 \
      --lr 0.05 \
      --lr_steps 70 80 90 \
      --batch_size 1024 \
      --train_add_strong \
      --useWhatModal image_textGT \
      --noise_percent 80 \
      --add_name get_Ts_image_textGT_80noise_V2_MLP


+++++++++++++++++++++++++++


### Distill the Student:



+++++++++++++++++++++++++++


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


CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T4sImgCATwordnetV2MLP_Ss_ViT


CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --useWhatModal text_wordnet \
   --T4_add_strong_wordnet --S_add_strong \
   --add_name T4sWordnetV2MLP_Ss_ViT



-----------------------


CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 300 400 0 \
   --batch_size 512 \
   --useWhatModal image_text_20gt80wordnet \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCAT20gt80wordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCAT20gt80wordnetV2MLP_Ss_ViT

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --useWhatModal image_text_20gt80wordnet \
   --T4_add_strong_imgCAT20gt80wordnet --S_add_strong \
   --add_name T4sImgCAT20gt80wordnetV2MLP_Ss_ViT


CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --useWhatModal text_20gt80wordnet \
   --T4_add_strong_20gt80wordnet --S_add_strong \
   --add_name T4s20gt80wordnetV2MLP_Ss_ViT



CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 300 400 0 \
   --batch_size 512 \
   --useWhatModal image_text_50gt50wordnet \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCAT50gt50wordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCAT50gt50wordnetV2MLP_Ss_ViT

CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --useWhatModal image_text_50gt50wordnet \
   --T4_add_strong_imgCAT50gt50wordnet --S_add_strong \
   --add_name T4sImgCAT50gt50wordnetV2MLP_Ss_ViT


CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --useWhatModal text_50gt50wordnet \
   --T4_add_strong_50gt50wordnet --S_add_strong \
   --add_name T4s50gt50wordnetV2MLP_Ss_ViT



CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 3 \
   --epoch 500 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 300 400 0 \
   --batch_size 512 \
   --useWhatModal image_text_80gt20wordnet \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCAT80gt20wordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCAT80gt20wordnetV2MLP_Ss_ViT

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --useWhatModal image_text_80gt20wordnet \
   --T4_add_strong_imgCAT80gt20wordnet --S_add_strong \
   --add_name T4sImgCAT80gt20wordnetV2MLP_Ss_ViT


CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --useWhatModal text_80gt20wordnet \
   --T4_add_strong_80gt20wordnet --S_add_strong \
   --add_name T4s80gt20wordnetV2MLP_Ss_ViT



CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 3 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --useWhatModal image_textGT \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATgt --S_add_strong \
   --add_name T1s_T1w_T4sImgCATgtV2MLP_Ss_ViT

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --useWhatModal image_textGT \
   --T4_add_strong_imgCATgt --S_add_strong \
   --add_name T4sImgCATgtV2MLP_Ss_ViT


CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --useWhatModal textGT \
   --T4_add_strong_gt --S_add_strong \
   --add_name T4sGtV2MLP_Ss_ViT




-----------------------


CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 3 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --useWhatModal image_textGT \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATgtNoise --S_add_strong \
   --noise_percent 20 \
   --add_name T1s_T1w_T4sImgCAT80gt20noiseV2MLP_Ss_ViT

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --useWhatModal image_textGT \
   --T4_add_strong_imgCATgtNoise --S_add_strong \
   --noise_percent 20 \
   --add_name T4sImgCAT80gt20noiseV2MLP_Ss_ViT





CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 3 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --useWhatModal image_textGT \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATgtNoise --S_add_strong \
   --noise_percent 100 \
   --add_name T1s_T1w_T4sImgCAT0gt100noiseV2MLP_Ss_ViT

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --useWhatModal image_textGT \
   --T4_add_strong_imgCATgtNoise --S_add_strong \
   --noise_percent 100 \
   --add_name T4sImgCAT0gt100noiseV2MLP_Ss_ViT







CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 3 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --useWhatModal image_textGT \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATgtNoise --S_add_strong \
   --noise_percent 50 \
   --add_name T1s_T1w_T4sImgCAT50gt50noiseV2MLP_Ss_ViT

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --useWhatModal image_textGT \
   --T4_add_strong_imgCATgtNoise --S_add_strong \
   --noise_percent 50 \
   --add_name T4sImgCAT50gt50noiseV2MLP_Ss_ViT




CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 3 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --useWhatModal image_textGT \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATgtNoise --S_add_strong \
   --noise_percent 80 \
   --add_name T1s_T1w_T4sImgCAT20gt80noiseV2MLP_Ss_ViT

CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_ViT_regularizerV2_sgd.py --mode distil --arch vit_base_patch32_224 --dataset ImageNet-mini --teacher_num 1 \
   --epoch 300 --temp 3.0 --alpha 0.6 \
   --lr 0.1 \
   --lr_steps 190 195 0 \
   --batch_size 512 \
   --useWhatModal image_textGT \
   --T4_add_strong_imgCATgtNoise --S_add_strong \
   --noise_percent 80 \
   --add_name T4sImgCAT20gt80noiseV2MLP_Ss_ViT





++++++++++++++++++++






# #####################



















