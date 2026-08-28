### Train the Teacher:



+++++++++++++++++++++++++++


#CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_regularizerV2.py --mode train --arch resnet50 --dataset ImageNet \
#      --epoch 60 \
#      --lr_CosineAnnealing 35 50 \
#      --batch_size 1024 \
#      --train_add_strong  \
#      --useWhatModal image_textWordNet \
#      --add_name get_Ts_image_textWordNet_V2

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_regularizerV2.py --mode train --arch resnet50 --dataset ImageNet \
      --epoch 35 \
      --lr_CosineAnnealing 30 0 \
      --batch_size 1024 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet_V2

CUDA_VISIBLE_DEVICES=0 python mainCLIPKD_regularizerV2.py --mode train --arch resnet50 --dataset ImageNet \
      --epoch 35 \
      --lr_CosineAnnealing 30 0 \
      --batch_size 1024 \
      --train_add_strong  \
      --useWhatModal image_text_10gt90wordnet \
      --add_name get_Ts_image_text_10gt90wordnet_V2



CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_regularizerV2.py --mode train --arch resnet50 --dataset ImageNet_LT \
      --epoch 60 \
      --lr_steps 18 36 54 \
      --batch_size 1024 \
      --train_add_strong  \
      --useWhatModal image_textWordNet \
      --add_name get_Ts_image_textWordNet_V2 \
      --next_continue runs/ImageNet_LT_train_get_Ts_image_textWordNet_V2/run-1-epoch60/checkpoint_bestAcc1.pth.tar


+++++++++++++++++++++++++++


### Distill the Student:



+++++++++++++++++++++++++++


CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_regularizerV2.py --mode distil --arch resnet18 --dataset ImageNet --teacher_num 3 \
   --epoch 60 --temp 3.0 --alpha 0.6 \
   --lr_CosineAnnealing 60 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --T2_add_strong --S_add_strong \
   --add_name T1s_T1w_T2s_Ss 




CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_regularizerV2.py --mode distil --arch resnet18 --dataset ImageNet --teacher_num 3 \
   --epoch 100 --temp 3.0 --alpha 0.6 \
   --lr_CosineAnnealing 60 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnetV2_Ss
#   --next_continue runs/ImageNet_distil_T1s_T1w_T4sImgCATwordnetV2_Ss/run-1-epoch60/checkpoint_each-epoch.pth.tar

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_regularizerV2.py --mode distil --arch resnet18 --dataset ImageNet --teacher_num 3 \
   --epoch 100 --temp 3.0 --alpha 0.6 \
   --lr_CosineAnnealing 100 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnetV2_Ss




CUDA_VISIBLE_DEVICES=2 python mainCLIPKD_regularizerV2.py --mode distil --arch resnet18 --dataset ImageNet --teacher_num 3 \
   --epoch 100 --temp 3.0 --alpha 0.6 \
   --lr_CosineAnnealing 100 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnetV2_Ss_invLoss

CUDA_VISIBLE_DEVICES=1 python mainCLIPKD_regularizerV2.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 3 \
   --epoch 230 --temp 3.0 --alpha 0.6 \
   --lr_steps 200 210 220 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnetV2_Ss_invLoss


CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_regularizerV2.py --mode distil --arch resnet18 --dataset ImageNet_LT --teacher_num 1 \
   --epoch 230 --temp 3.0 --alpha 0.6 \
   --lr_steps 200 210 220 \
   --batch_size 512 \
   --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T4sImgCATwordnetV2_Ss





CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_regularizerV2.py --mode distil --arch resnet18 --dataset ImageNet --teacher_num 3 \
   --epoch 60 --temp 3.0 --alpha 0.6 \
   --lr_CosineAnnealing 60 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCAT10gt90wordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCAT10gt90wordnetV2_Ss


python main.py --mode distil --arch resnet18 --dataset ImageNet --teacher_num 2 \
   --epoch 60 --temp 3.0 --alpha 0.6 \
   --lr_CosineAnnealing 60 0 \
   --T1_add_strong --T2_add_strong --S_add_weak \
   --add_name T1s_T2s_Sw \


CUDA_VISIBLE_DEVICES=3 python mainCLIPKD_regularizerV2_embdDistill.py --mode distil --arch resnet18 --dataset ImageNet --teacher_num 3 \
   --epoch 100 --temp 3.0 --alpha 0.6 \
   --lr_CosineAnnealing 100 0 \
   --batch_size 512 \
   --T1_add_strong --T1_add_weak --T4_add_strong_imgCATwordnet --S_add_strong \
   --add_name T1s_T1w_T4sImgCATwordnetV2_Ss_embd




++++++++++++++++++++






# #####################



















