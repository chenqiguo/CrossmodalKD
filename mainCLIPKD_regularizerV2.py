#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:51:55 2024

@author: ps
"""

# V2: employ the WordNet-relaxed nouns CLIP text embeddings as regularizer.


import argparse
import os

import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models
from torchvision import transforms as T

from trainer_regularizerV2 import train_model_withCLIP,get_logfile_name,print_write
from data_loader.DataLoaderCIFAR import Load_CIFAR100, Load_CIFAR100_noised, Load_CIFAR100_clean_noised
from data_loader.DataLoaderImageNet import Load_ImageNet_emb, Load_ImageNet
from data_loader.DataLoaderScene import Load_scene
from data_loader.DataLoaderUTKFace import Load_UTKFace
from data_loader.DataLoaderInaturalist import Load_inaturalist


import timm


resnets = ['resnet18', 'resnet34', 'resnet50']
students = ['resnet18', 'resnet34']
modes = ['train', 'distil'] 
data_root = {
        'CIFAR100':  '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img',
        'ImageNet':'/home/ps/scratch/KD_imbalance/LFME/my_data/ILSVRC/Data/CLS-LOC',
        'scene': '/home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/scene/cleaned',
        'UTKFace': '/home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/UTKFace/cls_by_race',
        'inaturalist': '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/inaturalist-2019/all',
        'ImageNet-mini': '/home/ps/scratch/CLIP_KD/data/ImageNet-mini',
        
        # 'CIFAR100': 'H:\Dataset\cifar100\clean_img',
        # 'ImageNet': 'H:\Dataset\ImageNet',
        }
teacher_path = {
    
    ### for inaturalist:
    #"weak" : "runs/inaturalist_train_get_Tw/run-2-epoch45/checkpoint_bestAcc1.pth.tar",
    #"strong" : "runs/inaturalist_train_get_Ts/run-2-epoch45/checkpoint_bestAcc1.pth.tar",
    
    #"image_textWordNet": "runs/inaturalist_train_get_Tw_image_textWordNet_V2/run-2-epoch100/checkpoint_bestAcc1.pth.tar", #"runs/CIFAR100_train_get_Tw_image_textWordNet/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    ##"image_textWordNet": "runs/inaturalist_train_get_Tw_image_textWordNet/run-1-epoch45/checkpoint_bestAcc1.pth.tar", #"runs/CIFAR100_train_get_Tw_image_textWordNet/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    
    
    ### for CIFAR100:
    #"weak" : "runs/CIFAR100_train_get_Tw/run-1/checkpoint_bestAcc1.pth.tar",
    #"strong" : "runs/CIFAR100_train_get_Ts/run-1/checkpoint_bestAcc1.pth.tar",
    
    #"image_textWordNet": "runs/CIFAR100_train_get_Ts_image_textWordNet_V2/run-1-epoch61_canonicalVer/checkpoint_bestAcc1.pth.tar", #"runs/CIFAR100_train_get_Tw_image_textWordNet/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    ##"image_textWordNet_s": "runs/CIFAR100_train_get_Ts_image_textWordNet/run-3-epoch400/checkpoint_bestAcc1.pth.tar", #"runs/CIFAR100_train_get_Ts_image_textWordNet/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    
    # newly added for rebuttal:
    #"image_textWordNet_mlp": "runs/CIFAR100_train_get_Ts_image_textWordNet_V2_mlp/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    #"image_textWordNet_vit": "runs/CIFAR100_train_get_Ts_image_textWordNet_V2_vit/run-1-epoch140/checkpoint_bestAcc1.pth.tar",
    
    # for ESWA LLM-relaxed:
    #"image_textWordNet": "runs/CIFAR100_train_get_Ts_image_textLLM_V3/run-4-epoch61/checkpoint_bestAcc1.pth.tar",
    #"image_textWordNet": "runs/CIFAR100_train_get_Ts_image_textLLMtextonly_V4/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    
    
    #"image_text_10gt90wordnet": "runs/CIFAR100_train_get_Ts_image_text_10gt90wordnet_V2/run-4-epoch61/checkpoint_bestAcc1.pth.tar", #"runs/CIFAR100_train_get_Ts_image_text_10gt90wordnet/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    
    #"image_textWNtreeV1": "runs/CIFAR100_train_get_Ts_image_textWNtreeV1/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    #"image_text_10gt90WNtreeV1": "runs/CIFAR100_train_get_Ts_image_text_10gt90WNtreeV1/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    
    #"image_textWNtreeV2": "runs/CIFAR100_train_get_Ts_image_textWNtreeV2/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    #"image_text_10gt90WNtreeV2": "runs/CIFAR100_train_get_Ts_image_text_10gt90WNtreeV2/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    
    
    ### for ImageNet
    #"weak" : "runs/ImageNet_train_get_Tw/run-10-epoch30/checkpoint_bestAcc1.pth.tar",
    #"strong" : "runs/ImageNet_train_get_Ts/run-12-epoch30/checkpoint_bestAcc1.pth.tar",
    #"strong_2" : "runs/ImageNet_train_get_Ts/run-11-epoch30/checkpoint_bestAcc1.pth.tar",
    
    #"image_textWordNet": "runs/ImageNet_train_get_Ts_image_textWordNet_V2/run-4-epoch35/checkpoint_bestAcc1.pth.tar", 
    #"image_text_10gt90wordnet": "runs/ImageNet_train_get_Ts_image_text_10gt90wordnet_V2/run-1-epoch35/checkpoint_bestAcc1.pth.tar", 
    
    # for ESWA LLM-relaxed:
    #"image_textWordNet": "runs/ImageNet_train_get_Ts_image_textLLM_V3/run-1-epoch35/checkpoint_bestAcc1.pth.tar",
    
    ### for CIFAR100_imb100:
    #"weak" : "runs/CIFAR100_imb100_train_get_T2w/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    #"strong" : "runs/CIFAR100_imb100_train_get_T2s/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    #"strong_2": "runs/CIFAR100_imb100_train_get_T1s/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    
    #"image_textWordNet": "runs/CIFAR100_imb100_train_get_Ts_image_textWordNet_V2/run-1-epoch200/checkpoint_bestAcc1.pth.tar", #"runs/CIFAR100_imb100_train_get_Ts_image_textWordNet/run-6-epoch100/checkpoint_bestAcc1.pth.tar",
    
    # for ESWA LLM-relaxed:
    #"image_textWordNet": "runs/CIFAR100_imb100_train_get_Ts_image_textLLM_V3/run-1-epoch200/checkpoint_bestAcc1.pth.tar",
    
    
    ### for ImageNet_LT:
    "weak" : "runs/ImageNet_LT_train_get_Tw/run-2-epoch60/checkpoint_bestAcc1.pth.tar",
    "strong" : "runs/ImageNet_LT_train_get_Ts/run-1-epoch60/checkpoint_bestAcc1.pth.tar",
    
    #"gt_CLIPtext": "runs/ImageNet_LT_train_get_Ts_textGT/run-1-epoch60/checkpoint_bestAcc1.pth.tar",
    #"wordnet_CLIPtext": "runs/ImageNet_LT_train_get_Ts_textWordNet/run-3-epoch170/checkpoint_bestAcc1.pth.tar",
    
    #"strong_2": "runs/ImageNet_LT_train_get_Ts/run-2-epoch60/checkpoint_bestAcc1.pth.tar",
    
    #"image_textWordNet": "runs/ImageNet_LT_train_get_Ts_image_textWordNet_V2/run-2-epoch60/checkpoint_bestAcc1.pth.tar", 
    
    # for ESWA LLM-relaxed:
    "image_textWordNet": "runs/ImageNet_LT_train_get_Ts_image_textLLM_V3/run-1-epoch60/checkpoint_bestAcc1.pth.tar",
    
    ### for scene
    #"weak" : "runs/scene_train_get_Tw/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    #"strong" : "runs/scene_train_get_Ts/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    ##"strong_2" : "runs/scene_train_get_Ts/run-2-epoch61/checkpoint_bestAcc1.pth.tar",
    
    #"image_textWordNet": "runs/scene_train_get_Ts_image_textWordNet_V2/run-1-epoch61/checkpoint_bestAcc1.pth.tar", 
    ##"image_text_10gt90wordnet": "runs/ImageNet_train_get_Ts_image_text_10gt90wordnet_V2/run-1-epoch35/checkpoint_bestAcc1.pth.tar", 
    
    # for ESWA LLM-relaxed:
    #"image_textWordNet": "runs/scene_train_get_Ts_image_textLLM_V3/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    
    ### for UTKFace
    #"weak" : "runs/UTKFace_train_get_Tw/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    #"strong" : "runs/UTKFace_train_get_Ts/run-2-epoch61/checkpoint_bestAcc1.pth.tar",
    ##"strong_2" : "runs/UTKFace_train_get_Ts/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    
    #"image_textWordNet": "runs/UTKFace_train_get_Ts_image_textWordNet_V2/run-2-epoch61/checkpoint_bestAcc1.pth.tar", 
    ##"image_text_10gt90wordnet": "runs/ImageNet_train_get_Ts_image_text_10gt90wordnet_V2/run-1-epoch35/checkpoint_bestAcc1.pth.tar", 
    
    ### for ImageNet-mini:
    #"weak" : "runs/ImageNet-mini_train_get_Tw_image_ViT/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    #"strong" : "runs/ImageNet-mini_train_get_Ts_image_ViT/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    #"image_textWordNet": "runs/ImageNet-mini_train_get_Ts_image_textWordNet_V2_MLP/run-2-epoch100/checkpoint_bestAcc1.pth.tar",
    
    
}


parser = argparse.ArgumentParser(description='CLIP KD naive v1')
parser.add_argument('--mode', default='distil', choices=modes,
                    help='program mode: ' +
                        ' | '.join(resnets) +
                        ' (default: train)')
parser.add_argument('--arch', default='resnet18', choices=resnets,
                    help='model architecture: ' +
                        ' | '.join(resnets) +
                        ' (default: resnet18)')
parser.add_argument('--teacher_num', default=0, type=int,
                    help='the number of teacher models')
parser.add_argument('--epochs', default=120, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int,
                    help='batch size')
parser.add_argument('--dataset', default="ImageNet_LT", # eg.CIFAR100_imb100
                    help='choose which dataset to use.')
parser.add_argument('-t', '--temp', default=10., type=float,
                    help='temperature for distillation')

parser.add_argument('--alpha', default=0.2, type=float,
                    help='weighting for hard loss during distillation')

parser.add_argument('--add_name', default="01",
                    help='Add the name of the model.')

parser.add_argument('--lr_steps', default=[0,0,0], nargs='+', type=int,
                    help="LambdaLR's step1, step2, and step3.")
parser.add_argument('--lr_CosineAnnealing', default=[0,0], nargs='+', type=int,
                    help="CosineAnnealingLR's T_max and eta_min")

parser.add_argument('--train_add_weak',default=False, action='store_true',
                    help="When training the teacher model, opt for training with a weak augmentation approach.")
parser.add_argument('--train_add_strong',default=False, action='store_true',
                    help="When training the teacher model, opt for training with a strong augmentation approach.")

# T1 is the Teacher trained with raw images, with weak or strong data-aug:
parser.add_argument('--T1_add_weak',default=False, action='store_true',
                    help="When training the student model, opt for using weak augmentation training with the T1 teacher.")
parser.add_argument('--T1_add_strong',default=False, action='store_true',
                    help="When training the student model, opt for using strong augmentation with the T1 teacher.")

parser.add_argument('--T2_add_strong',default=False, action='store_true',
                    help="When training the student model, opt for using strong augmentation with the T2 teacher.")

# T2 is the Teacher trained with CLIP image embeddings. Note: we only trained with weak aug!:
parser.add_argument('--T2_add_weak_CLIPimg',default=False, action='store_true',
                    help="When training the student model, opt for using weak augmentation CLIP image embeddings with the T2 teacher.")
parser.add_argument('--T2_add_strong_CLIPimg',default=False, action='store_true',
                    help="When training the student model, opt for using strong augmentation CLIP image embeddings with the T2 teacher.")

# T3 is the Teacher trained with CLIP text embeddings, either GT class names or WordNet nouns. Note: we only trained with weak aug!:
parser.add_argument('--T3_add_gt',default=False, action='store_true',
                    help="When training the student model, opt for using GT class names CLIP text embeddings with the T3 teacher.")
parser.add_argument('--T3_add_wordnet',default=False, action='store_true',
                    help="When training the student model, opt for using WordNet nouns CLIP text embeddings as TAC with the T3 teacher.")
#parser.add_argument('--T3_add_weak',default=False, action='store_true',
#                    help="When training the student model, opt for using weak augmentation with the T3 teacher.")
#parser.add_argument('--T3_add_strong',default=False, action='store_true',
#                    help="When training the student model, opt for using strong augmentation with the T3 teacher.")
parser.add_argument('--T3_add_70gt30wordnet',default=False, action='store_true',
                    help="When training the student model, opt for using 70percentGT 30percentWordNet class names CLIP text embeddings with the T3 teacher.")
parser.add_argument('--T3_add_30gt70wordnet',default=False, action='store_true',
                    help="When training the student model, opt for using 30percentGT 70percentWordNet nouns CLIP text embeddings as TAC with the T3 teacher.")
parser.add_argument('--T3_add_80gt20wordnet',default=False, action='store_true',
                    help="When training the student model, opt for using 80percentGT 20percentWordNet class names CLIP text embeddings with the T3 teacher.")
parser.add_argument('--T3_add_20gt80wordnet',default=False, action='store_true',
                    help="When training the student model, opt for using 20percentGT 80percentWordNet nouns CLIP text embeddings as TAC with the T3 teacher.")
parser.add_argument('--T3_add_50gt50wordnet',default=False, action='store_true',
                    help="When training the student model, opt for using 50percentGT 50percentWordNet nouns CLIP text embeddings as TAC with the T3 teacher.")
parser.add_argument('--T3_add_90gt10wordnet',default=False, action='store_true',
                    help="When training the student model, opt for using 90percentGT 10percentWordNet class names CLIP text embeddings with the T3 teacher.")
parser.add_argument('--T3_add_10gt90wordnet',default=False, action='store_true',
                    help="When training the student model, opt for using 10percentGT 90percentWordNet nouns CLIP text embeddings as TAC with the T3 teacher.")
parser.add_argument('--T3_add_95gt5wordnet',default=False, action='store_true',
                    help="When training the student model, opt for using 90percentGT 10percentWordNet class names CLIP text embeddings with the T3 teacher.")
parser.add_argument('--T3_add_5gt95wordnet',default=False, action='store_true',
                    help="When training the student model, opt for using 10percentGT 90percentWordNet nouns CLIP text embeddings as TAC with the T3 teacher.")
parser.add_argument('--T3_add_1gt99wordnet',default=False, action='store_true',
                    help="When training the student model, opt for using 10percentGT 90percentWordNet nouns CLIP text embeddings as TAC with the T3 teacher.")
parser.add_argument('--T3_add_3gt97wordnet',default=False, action='store_true',
                    help="When training the student model, opt for using 10percentGT 90percentWordNet nouns CLIP text embeddings as TAC with the T3 teacher.")
parser.add_argument('--T3_add_0gt100wordnet',default=False, action='store_true',
                    help="When training the student model, opt for using 10percentGT 90percentWordNet nouns CLIP text embeddings as TAC with the T3 teacher.")
parser.add_argument('--T3_add_60gt40wordnet',default=False, action='store_true',
                    help="When training the student model, opt for using 10percentGT 90percentWordNet nouns CLIP text embeddings as TAC with the T3 teacher.")
parser.add_argument('--T3_add_40gt60wordnet',default=False, action='store_true',
                    help="When training the student model, opt for using 10percentGT 90percentWordNet nouns CLIP text embeddings as TAC with the T3 teacher.")



parser.add_argument('--T3_add_gt_noise',default=False, action='store_true',
                    help="When training the student model, opt for using 20percent noised gt class names CLIP text embeddings with the T3 teacher.")


# T4 is the Teacher trained with CLIP image concate text embeddings. Note: we only trained with weak aug!:
parser.add_argument('--T4_add_weak_imgCATgt',default=False, action='store_true',
                    help="When training the student model, opt for using CLIP img embeddings cat GT class names CLIP text embeddings with the T4 teacher.")
parser.add_argument('--T4_add_weak_imgCATwordnet',default=False, action='store_true',
                    help="When training the student model, opt for using CLIP img embeddings cat WordNet nouns CLIP text embeddings as TAC with the T4 teacher.")
parser.add_argument('--T4_add_strong_imgCATgt',default=False, action='store_true',
                    help="When training the student model, opt for using CLIP img embeddings cat GT class names CLIP text embeddings with the T4 teacher.")
parser.add_argument('--T4_add_strong_imgCATwordnet',default=False, action='store_true',
                    help="When training the student model, opt for using CLIP img embeddings cat WordNet nouns CLIP text embeddings as TAC with the T4 teacher.")
parser.add_argument('--T4_add_strong_imgCAT10gt90wordnet',default=False, action='store_true',
                    help="When training the student model, opt for using CLIP img embeddings cat GT class names and wordnet nouns CLIP text embeddings with the T4 teacher.")
parser.add_argument('--T4_add_strong_imgCAT1gt99wordnet',default=False, action='store_true',
                    help="When training the student model, opt for using CLIP img embeddings cat GT class names and wordnet nouns CLIP text embeddings with the T4 teacher.")
parser.add_argument('--T4_add_strong_imgCAT20gt80wordnet',default=False, action='store_true',
                    help="When training the student model, opt for using CLIP img embeddings cat GT class names and wordnet nouns CLIP text embeddings with the T4 teacher.")

parser.add_argument('--T4_add_strong_imgCATWNtreeV1',default=False, action='store_true',
                    help="When training the student model, opt for using CLIP img embeddings cat GT class names and wordnet nouns CLIP text embeddings with the T4 teacher.")
parser.add_argument('--T4_add_strong_imgCAT10gt90WNtreeV1',default=False, action='store_true',
                    help="When training the student model, opt for using CLIP img embeddings cat GT class names and wordnet nouns CLIP text embeddings with the T4 teacher.")
parser.add_argument('--T4_add_strong_imgCAT1gt99WNtreeV1',default=False, action='store_true',
                    help="When training the student model, opt for using CLIP img embeddings cat GT class names and wordnet nouns CLIP text embeddings with the T4 teacher.")
parser.add_argument('--T4_add_strong_imgCATWNtreeV2',default=False, action='store_true',
                    help="When training the student model, opt for using CLIP img embeddings cat GT class names and wordnet nouns CLIP text embeddings with the T4 teacher.")
parser.add_argument('--T4_add_strong_imgCAT10gt90WNtreeV2',default=False, action='store_true',
                    help="When training the student model, opt for using CLIP img embeddings cat GT class names and wordnet nouns CLIP text embeddings with the T4 teacher.")

# for rebuttal:
parser.add_argument('--T4_add_strong_imgCATwordnet_mlp',default=False, action='store_true',
                    help="When training the student model, opt for using CLIP img embeddings cat WordNet nouns CLIP text embeddings as TAC with the T4 teacher.")
parser.add_argument('--T4_add_strong_imgCATwordnet_vit',default=False, action='store_true',
                    help="When training the student model, opt for using CLIP img embeddings cat WordNet nouns CLIP text embeddings as TAC with the T4 teacher.")



parser.add_argument('--T4_add_strong_imgCATgtNoise',default=False, action='store_true',
                    help="When training the student model, opt for using CLIP img embeddings cat GT class names with noise (randonly flipping) CLIP text embeddings with the T4 teacher.")


parser.add_argument('--S_add_weak',default=False, action='store_true',
                    help="When training the student model, choose to employ weak augmentation for training the student model.")
parser.add_argument('--S_add_strong',default=False, action='store_true',
                    help="When training the student model, choose to employ strong augmentation for training the student model.")

parser.add_argument('--next_continue',default=None,
                    help="Whether to load a checkpoint and resume training. Accepts the checkpoint file path as a parameter.")

parser.add_argument('--result', metavar='DIR',
                    help='path to results')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

# newly added by Chenqi: 
parser.add_argument('--useWhatModal', default="image", # or text...
                    help='choose which modal, image or text embeddings, to use.')
parser.add_argument('--noise_percent', default=0, type=int,
                    help='the (100) percentage of the training data labels get noised')

# newly added for rebuttal
parser.add_argument('--TxArch', default="resnet", # or vit, mlp
                    help='for rebuttal: try with different teacher arch.')






def learing_rate_scheduler(optimizer,args):

    if args.lr_steps[0] != 0:
        step1 = args.lr_steps[0]
        step2 = args.lr_steps[1]
        step3 = args.lr_steps[2]
        gamma = 0.1
        warmup_epoch = 8
        print("Scheduler step1, step2, warmup_epoch, gamma:", (step1,step2,step3, gamma))
        def lr_lambda(epoch):
            if step3 != 0 and epoch >= step3:
                lr = gamma * gamma * gamma
            elif epoch >=  step2:
                lr = gamma * gamma
            elif epoch >= step1:
                lr = gamma
            else:
                lr = 1

            """Warmup"""
            if epoch < warmup_epoch:
                lr = lr * float(1 + epoch) / warmup_epoch
            return lr
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif args.lr_CosineAnnealing[0] != 0:
        T_max = args.lr_CosineAnnealing[0]
        eta_min = args.lr_CosineAnnealing[1]
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        print("The learning rate scheduler is not defined, please check.")
        assert False
    return lr_scheduler


"""
class AdaptedResNet50(nn.Module):
    def __init__(self, resnet50_model):
        super(AdaptedResNet50, self).__init__()
        
        # Remove the firssst layer that expect image input:
        self.features = nn.Sequential(*list(resnet50_model.children())[4:]) # After the initial layers
        # Add an input layer to transform the 512-d CLIP embedding into a compatible feature map
        self.fc_input = nn.Linear(512, 64) # reshaping CLIP embedding to a larger size
        
        #self.resnet50 = resnet50_model
    
    def forward(self, x):
        x = self.fc_input(x) # Transform input CLIP embeddings to larger feature size
        print('x.shape = ' + str(x.shape))
        x = x.unsqueeze(-1).unsqueeze(-1) # Reshape output to simulate a "feature_map" [batch_size, 512, 1,1]
        print('x.shape = ' + str(x.shape))
        x = self.features(x) # Feed into the remaining ResNet layers
        print('x.shape = ' + str(x.shape))
        #x = self.resnet50(x)
        #print('x.shape = ' + str(x.shape))
        return x

class AdaptedResNet18(nn.Module):
    def __init__(self, resnet18_model):
        super(AdaptedResNet18, self).__init__()
        # Remove the firssst layer that expect image input:
        self.features = nn.Sequential(*list(resnet18_model.children())[4:]) # After the initial layers
        # Add an input layer to transform the 512-d CLIP embedding into a compatible feature map
        self.fc_input = nn.Linear(512, 512) # reshaping CLIP embedding to a larger size
    def forward(self, x):
        x = self.fc_input(x) # Transform input CLIP embeddings to larger feature size
        x = x.unsqueeze(-1).unsqueeze(-1) # Reshape output to simulate a "feature_map" [batch_size, 512, 1,1]
        x = self.features(x) # Feed into the remaining ResNet layers
        return x
"""




class MLPTeacher(nn.Module):
    """
    MLP‑512×2 teacher for 1024‑dim CLIP [image ‖ text] embeddings
    """
    def __init__(self, in_dim=1024, hidden=512, num_classes=100, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, vt_1024):          # vt_1024 : (B , 1024)
        return self.net(vt_1024)




class SingleTokenViT(nn.Module):
    """
    ViT‑B/32 that takes one projected [v‖t] token.
    """
    def __init__(self, feat_dim=1024, timm_name='vit_base_patch32_224', 
                 num_classes=100):
        super().__init__()

        # create vanilla ViT
        base = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)

        D = base.embed_dim            # 768 for ViT‑B
        # ------------- replace patch‑embed with identity ------------------
        base.patch_embed = nn.Identity()      # we supply our own token
        base.num_tokens = 1                   # only CLS
        # ------------- new learnable pieces --------------------------------
        self.proj      = nn.Linear(feat_dim, D)        # 1024 → D
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 1, D))  # CLS+1 token
        self.pos_drop  = base.pos_drop
        self.blocks    = base.blocks
        self.norm      = base.norm
        self.head      = base.head

    def forward(self, vt_1024):               # vt_1024 : (B ,1024)
        x = self.proj(vt_1024).unsqueeze(1)   # (B,1,D)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed
        x   = self.pos_drop(x)
        x   = self.blocks(x)
        x   = self.norm(x)
        return self.head(x[:, 0])             # logits





def train(args, path, log_file, device):

    if 'CIFAR100' in args.dataset:
        print("Function: load_CIFAR100.")
        # newly modified by Chenqi:
        if args.noise_percent ==0:
            data = {x: Load_CIFAR100(data_root=data_root[args.dataset.split("_")[0]], dataset=args.dataset, phase=x,
                         batch_size=args.batch_size, num_workers=4,
                         shuffle=True if x == 'train' else False)
                    for x in ['train', 'val', 'val_aug']} 
        
        elif args.mode == 'train' and args.noise_percent !=0:
            data = {x: Load_CIFAR100_noised(data_root=data_root[args.dataset.split("_")[0]], dataset=args.dataset, phase=x,
                         batch_size=args.batch_size, num_workers=4,
                         noise_percent=args.noise_percent,
                         shuffle=True if x == 'train' else False)
                    for x in ['train', 'val', 'val_aug']} 
        elif args.mode == 'distil' and args.noise_percent !=0:
            data = {x: Load_CIFAR100_clean_noised(data_root=data_root[args.dataset.split("_")[0]], dataset=args.dataset, phase=x,
                         batch_size=args.batch_size, num_workers=4,
                         noise_percent=args.noise_percent,
                         shuffle=True if x == 'train' else False)
                    for x in ['train', 'val', 'val_aug']}
            
    elif 'ImageNet' in args.dataset:
        print("Function: Load_ImageNet.")
        data = {x: Load_ImageNet(data_root=data_root[args.dataset.split("_")[0]], dataset=args.dataset, phase=x,
                    batch_size=args.batch_size, num_workers=4,
                    shuffle=True if x == 'train' else False)
            for x in ['train', 'val']}  
        #data = {x: Load_ImageNet_emb(data_root=data_root[args.dataset.split("_")[0]], emb_root='./data/'+args.dataset+'/CLIP_image_embeddings/',
        #                             dataset=args.dataset, phase=x,
        #            batch_size=args.batch_size, num_workers=4,
        #            shuffle=True) #if x == 'train' else False)
        #    for x in ['train', 'val', 'val_aug']}  
    
    
    elif 'scene' in args.dataset:
        print("Function: Load_scene.")
        data = {x: Load_scene(data_root=data_root[args.dataset.split("_")[0]], dataset=args.dataset, phase=x,
                    batch_size=args.batch_size, num_workers=4,
                    shuffle=True if x == 'train' else False)
            for x in ['train', 'val']}
    
    elif 'UTKFace' in args.dataset:
        print("Function: Load_UTKFace.")
        data = {x: Load_UTKFace(data_root=data_root[args.dataset.split("_")[0]], dataset=args.dataset, phase=x,
                    batch_size=args.batch_size, num_workers=4,
                    shuffle=True if x == 'train' else False)
            for x in ['train', 'val']}
    
    elif 'inaturalist' in args.dataset:
        print("Function: Load inaturalist.")
        data = {x: Load_inaturalist(data_root=data_root[args.dataset.split("_")[0]], dataset=args.dataset, phase=x,
                    batch_size=args.batch_size, num_workers=4,
                    shuffle=True if x == 'train' else False)
            for x in ['train', 'val']}
    
    
    
    if "CIFAR100" in args.dataset:
        class_num = 100
    elif "CIFAR10" in args.dataset:
        class_num = 10
    elif args.dataset == "ImageNet" or args.dataset == "ImageNet_LT":
        class_num = 1000
    elif args.dataset == "ImageNet-mini":
        class_num = 100
    elif "scene" in args.dataset:
        class_num = 6
    elif "UTKFace" in args.dataset:
        class_num = 5
    elif "inaturalist" in args.dataset:
        class_num = 1010
    else:
        class_num = None
        assert False

    
    if args.TxArch == 'resnet':
        model_ft = models.__dict__[args.arch](weights=None)
    elif args.TxArch == 'mlp':
        model_ft = MLPTeacher().to(device)
    elif args.TxArch == 'vit':
        model_ft = SingleTokenViT().to(device)
    
    
    # just for debug:
    #print('****** (1) model_ft = ' + str(model_ft))
    
    #model_ft = model
    if args.arch == 'resnet50' and args.TxArch == 'resnet' and args.mode == 'train':
        model_ft.load_state_dict(torch.load("./resnet50.pt"))
        # newly added by Chenqi: adapted for inputing the CLIP embeddings
        # with the shape of [batch_size, 512] instead of [batch_size, 3, H, W]:
        #model_ft = AdaptedResNet50(model_ft)
        
        if args.useWhatModal != 'rawImg':
            if args.useWhatModal == 'textGT' or args.useWhatModal == 'image' or args.useWhatModal == 'textWordNet'\
                or ('image' not in args.useWhatModal and 'text' in args.useWhatModal and 'gt' in args.useWhatModal and 'wordnet' in args.useWhatModal):
                inCh = 512
            elif 'image_text' in args.useWhatModal: #args.useWhatModal == 'image_textGT' or args.useWhatModal == 'image_textWordNet':
                inCh = 1024
            model_ft.conv1 = nn.Conv2d(in_channels=inCh, out_channels=64, kernel_size=7,
                                       stride=2, padding=3, bias=False)
        
    #elif args.arch == 'resnet18' and args.mode == 'distil':
    #    model_ft = AdaptedResNet18(model_ft)
        # ??? Add more...
    
    # just for debug:
    #print('****** (2) model_ft = ' + str(model_ft))
    #assert(False)
    
    if class_num != 1000 and args.TxArch == 'resnet': 
        #num_ftrs = model_ft.features[5].in_features
        #print('num_ftrs = ' + str(num_ftrs))
        #model_ft.features[5] = nn.Linear(num_ftrs, class_num)
        #num_ftrs = model_ft.resnet50.fc.in_features
        #model_ft.resnet50.fc = nn.Linear(num_ftrs, class_num)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, class_num)
    model_ft = model_ft.to(device)
    
    # just for debug:
    #print('****** model_ft = ' + str(model_ft))
    #assert(False)
    
    T_path = []
    key_strings = []
    if args.mode == 'distil':
        teacher = []
        if args.T1_add_weak:
            T_path.append(teacher_path['weak'])
            key_strings.append('weak')
        if args.T1_add_strong:
            T_path.append(teacher_path['strong'])
            key_strings.append('strong')
        
        if args.T2_add_strong:
            T_path.append(teacher_path['strong_2'])
            key_strings.append('strong')
        
        
        if args.T4_add_weak_imgCATgt:
            T_path.append(teacher_path['image_textGT'])
            key_strings.append('image_textGT')
        elif args.T4_add_strong_imgCATgt:
            T_path.append(teacher_path['image_textGT_s'])
            key_strings.append('image_textGT')
        elif args.T4_add_strong_imgCATwordnet or args.T4_add_weak_imgCATwordnet:
            T_path.append(teacher_path['image_textWordNet'])
            key_strings.append('image_textWordNet')
        
        elif args.T4_add_strong_imgCAT10gt90wordnet:
            T_path.append(teacher_path['image_text_10gt90wordnet'])
            key_strings.append('image_text_10gt90wordnet')
        elif args.T4_add_strong_imgCATwordnet_mlp: # for rebuttal
            T_path.append(teacher_path['image_textWordNet_mlp'])
            key_strings.append('image_textWordNet_mlp')
        elif args.T4_add_strong_imgCATwordnet_vit: # for rebuttal
            T_path.append(teacher_path['image_textWordNet_vit'])
            key_strings.append('image_textWordNet_vit')
        
        
        elif args.T4_add_strong_imgCATWNtreeV1:
            T_path.append(teacher_path['image_textWNtreeV1'])
            key_strings.append('image_textWNtreeV1')
        elif args.T4_add_strong_imgCAT10gt90WNtreeV1:
            T_path.append(teacher_path['image_text_10gt90WNtreeV1'])
            key_strings.append('image_text_10gt90WNtreeV1')
        elif args.T4_add_strong_imgCATWNtreeV2:
            T_path.append(teacher_path['image_textWNtreeV2'])
            key_strings.append('image_textWNtreeV2')
        elif args.T4_add_strong_imgCAT10gt90WNtreeV2:
            T_path.append(teacher_path['image_text_10gt90WNtreeV2'])
            key_strings.append('image_text_10gt90WNtreeV2')
        
        
        if len(T_path) != args.teacher_num or len(key_strings) != args.teacher_num:
            print("The number of loaded teachers does not match the preset. Please check.")
            assert(False)
        for i in range(args.teacher_num):
            print('----- i = ' + str(i))
            
            key_str = key_strings[i]
            
            if key_str == 'image_textWordNet_mlp':
                teacher_model = MLPTeacher().to(device)
            elif key_str == 'image_textWordNet_vit':
                teacher_model = SingleTokenViT().to(device)
            else:
                teacher_model = models.resnet50(pretrained=False) 
            
            
            #print('key_str = ' + key_str)
            # newly added by Chenqi:
            if key_str!='weak' and key_str!='strong' \
                and 'mlp' not in key_str and 'vit' not in key_str:
                if "image_text" not in key_str: # for T2, T3
                    inCh = 512
                elif "image_text" in key_str: # for T4
                    inCh = 1024
                #print('inCh = ' + str(inCh))
                teacher_model.conv1 = nn.Conv2d(in_channels=inCh, out_channels=64, kernel_size=7,
                                           stride=2, padding=3, bias=False)
            
            if 'mlp' not in key_str and 'vit' not in key_str:
                num_ftrs = teacher_model.fc.in_features
                print("class_num--->",class_num)
                teacher_model.fc = nn.Linear(num_ftrs, class_num) # orig code
            
            print("The loaded teacher model is:",T_path[i])
            teacher_model.load_state_dict(torch.load(T_path[i])['state_dict'])
            teacher.append(teacher_model)
        
    else:
        teacher = [] 
    
    
    # newly added by Chenqi: for V2 we updating the CLIP nouns embedding:
    nouns_embedding = None
    nouns_embedding_pretrained = None
    if args.mode == 'train' and args.useWhatModal != 'rawImg':
        from trainer_regularizerV2 import get_CLIP_text_embeddings_WordNet
        from TACmodels import CLIPModel
        model_CLIP = CLIPModel(model_name="ViT-B/32").to(device)
        model_CLIP.eval()
        nouns_embedding = get_CLIP_text_embeddings_WordNet(args, model_CLIP, device, data['train']) # data['train'] for CIFAR; data['val'] for ImageNet # make sure the file is already stored!
        nouns_embedding = torch.from_numpy(nouns_embedding).cuda().half()
        #nouns_embedding.requires_grad = True  # Set the noun embeddings as trainable: so that this is to be optimized!
        
        nouns_embedding_pretrained = nouns_embedding.clone().detach()  # Keep a copy of the original pretrained embeddings
        
        nouns_embedding = nn.Parameter(nouns_embedding)
        #optimizer_ft = optim.SGD(list(model_ft.parameters())+[nouns_embedding],
        #                 lr=args.lr, momentum=args.momentum)
        optimizer_ft = optim.SGD(
                    [{'params': model_ft.parameters()}, {'params': [nouns_embedding], 'weight_decay': 1e-4}],
                    lr=args.lr, momentum=args.momentum
                )
        
    else:
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=args.momentum)
    
    print(args.lr_steps)
    step1 = args.lr_steps[0]
    step2 = args.lr_steps[1]
    step3 = args.lr_steps[2]
    gamma = 0.1
    warmup_epoch = 8
    exp_lr_scheduler = learing_rate_scheduler(optimizer=optimizer_ft,args=args)


    # Resume from the checkpoint.
    if args.next_continue is not None:
        checkpoint_path = args.next_continue
        if args.add_name not in checkpoint_path:
            print("The loading path for resuming from the checkpoint is incorrect. Please check.")
            assert False
        # checkpoint = torch.load(checkpoint_path + '/checkpoint_state.pth.tar')
        checkpoint = torch.load(checkpoint_path)
        model_ft.load_state_dict(checkpoint['state_dict'])
        optimizer_ft.load_state_dict(checkpoint['optimizer'])
        exp_lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # start_epoch = checkpoint['epoch'] + 1 
        start_epoch = checkpoint['epoch'] - 1
    else:
        start_epoch = 0
        
    writer = SummaryWriter(path)

    if args.mode == 'train':
        print_str = [f"--teacher: {f'resnet50' if teacher else 'None'}",f"--teacher number: {args.teacher_num}\n"
                    f"--dataset: {args.dataset}",f"--epochs: {args.epochs}",f"--batch size: {args.batch_size}\n"
                    f"--train_add_weak: {args.train_add_weak}", f"--train_add_strong: {args.train_add_strong}\n"]
        if step1 != 0:
            print_str.append(
                f"learing_rate_scheduler: --scheduler: LambdaLR --step1: {step1} --step2: {step2} --step3: {step3} --warmup_epoch: {warmup_epoch}\n"
            )
        elif args.lr_CosineAnnealing[0] != 0:
            print_str.append(
                f"learing_rate_scheduler: --scheduler: CosineAnnealingLR --T_max: {args.lr_CosineAnnealing[0]} --eta_min: {args.lr_CosineAnnealing[1]}"
            )

    if args.mode == 'distil':
        print_str = [f" --teacher: {f'resnet50' if teacher else 'None'}",f"--teacher number: {args.teacher_num}\n"]
        
        for i in range(args.teacher_num):
            print_str.append(
                f"--teacher_{i}: {T_path[i]}\n"
            )
                        
        print_str.extend([
                f"--dataset: {args.dataset}",f"--epochs: {args.epochs}",f"--batch size: {args.batch_size}\n",
                #f"--T1_add_weak: {args.T1_add_weak}",f"--T1_add_strong: {args.T1_add_strong}\n",
                #f"--T2_add_weak: {args.T2_add_weak}",f"--T2_add_strong: {args.T2_add_strong}\n",
                #f"--T3_add_weak: {args.T3_add_weak}",f"--T3_add_strong: {args.T3_add_strong}\n",
                f"--S_add_weak: {args.S_add_weak}",f"--S_add_strong: {args.S_add_strong}\n",
                f"--distillation temperature: {args.temp}\n",
                f"--hard label weight: {args.alpha}",f"--soft label weight: {1. - args.alpha}\n",
        ]
                    )
        if step1 != 0:
            print_str.append(
                f"learing_rate_scheduler: --scheduler: LambdaLR --step1: {step1} --step2: {step2} --step3: {step3} --warmup_epoch: {warmup_epoch}\n"
            )
        elif args.lr_CosineAnnealing[0] != 0:
            print_str.append(
                f"learing_rate_scheduler: --scheduler: CosineAnnealingLR --T_max: {args.lr_CosineAnnealing[0]} --eta_min: {args.lr_CosineAnnealing[1]}"
            )

        # elif args.teacher_num == 1:
        #     print_str = [f"--teacher: {f'resnet50' if teacher else 'None'}",f"--teacher number: {args.teacher_num}\n"
        #                 f"--teacher_1: {T_path[0]}",f"--teacher_2: None\n"
        #                 f"--dataset: {args.dataset}",f"--epochs: {args.epochs}",f"--batch size: {args.batch_size}\n"
        #                 f"--T1_add_weak: {args.T1_add_weak}",f"--T1_add_strong: {args.T1_add_strong}\n"
        #                 f"--T2_add_weak: {args.T2_add_weak}",f"--T2_add_strong: {args.T2_add_strong}\n"
        #                 f"--S_add_weak: {args.S_add_weak}",f"--S_add_strong: {args.S_add_strong}\n"
        #                 f"--distillation temperature: {args.temp}\n"
        #                 f"--hard label weight: {args.alpha}",f"--soft label weight: {1. - args.alpha}\n"]
        #     if step1 != 0:
        #         print_str.append(
        #             f"learing_rate_scheduler: --scheduler: LambdaLR --step1: {step1} --step2: {step2} --step3: {step3} --warmup_epoch: {warmup_epoch}\n"
        #         )
        #     elif args.lr_CosineAnnealing[0] != 0:
        #         print_str.append(
        #             f"learing_rate_scheduler: --scheduler: CosineAnnealingLR --T_max: {args.lr_CosineAnnealing[0]} --eta_min: {args.lr_CosineAnnealing[1]}"
        #         )

    print_write(print_str, log_file)

    train_model_withCLIP(model_ft, data, optimizer_ft, nouns_embedding, nouns_embedding_pretrained,
                           exp_lr_scheduler, writer, device, args.temp, args.alpha, log_file,
                           args, key_strings, start_epoch, teacher, args.epochs)
    



def main():
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'Mode: {args.mode}')
    if args.mode == 'train':
        print_str = [f'Training model: {args.arch}...']
        path_1 = f'runs/{args.dataset}_{args.mode}_{args.add_name}'
        os.makedirs(path_1, exist_ok=True)

        subfolders = [f for f in os.listdir(path_1) if os.path.isdir(os.path.join(path_1, f)) and "run-" in f]
        path = path_1 + f"/run-{len(subfolders)+1}-epoch{args.epochs}"    

        log_file = get_logfile_name(path=path)
        print_write(print_str, log_file)
        args.result = path
        train(args, path, log_file, device)
    
    elif args.mode == 'distil':
        print_str = [f'Training student: {args.arch}...']
        if args.teacher_num >= 1:
            path_1 = f'runs/{args.dataset}_{args.mode}_{args.add_name}'
        # elif args.teacher_num==2:
        #     path_1 = f'runs/{args.dataset}_{args.mode}_{args.add_name}'
        os.makedirs(path_1, exist_ok=True)

        subfolders = [f for f in os.listdir(path_1) if os.path.isdir(os.path.join(path_1, f)) and "run-" in f]
        path = path_1 + f"/run-{len(subfolders)+1}-epoch{args.epochs}"      
        
        log_file = get_logfile_name(path=path)
        print_write(print_str, log_file)

        args.result = path
        train(args, path, log_file, device)




if __name__ == '__main__':
    main()
    
    
    
    



