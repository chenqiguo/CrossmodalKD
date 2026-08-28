#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:41:49 2024

@author: ps
"""

# This is my first try: a naive CLIP-KD framework: use the pre-trained CLIP model
# to generate image_embedding and text_embedding (as in TAC), and then input
# these embeddings into 2 teachers seoerately (one for img, the other for text).
# The teacher output logits are then input to one student model.

# What I expect: is to get a higher student val acc than the plain student training.
# If we can outperform the rethink-KD performance, then it would be even better!


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

from trainer_naiveV1_ViTMLP import train_model_withCLIP,get_logfile_name,print_write
from data_loader.DataLoaderCIFAR import Load_CIFAR100, Load_CIFAR100_noised, Load_CIFAR100_clean_noised
from data_loader.DataLoaderImageNet import Load_ImageNet
from data_loader.DataLoaderScene import Load_scene
from data_loader.DataLoaderUTKFace import Load_UTKFace
from data_loader.DataLoaderInaturalist import Load_inaturalist

import timm


vitnets = ['vit_base_patch32_224', 'vit_base_patch16_224', 'MLP'] # S, T arch

modes = ['train', 'distil'] 
data_root = {
        'CIFAR100':  '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img',
        'ImageNet':'/home/ps/scratch/KD_imbalance/LFME/my_data/ILSVRC/Data/CLS-LOC',
        'scene': '/home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/scene/cleaned',
        'UTKFace': '/home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/UTKFace/cls_by_race',
        'inaturalist': '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/inaturalist-2019/all',
        'ImageNet-mini': '/home/ps/scratch/CLIP_KD/data/ImageNet-mini',
        }
teacher_path = {
    ### for CIFAR100:
    #"image_textWordNet": "runs/CIFAR100_train_get_Ts_image_textWordNet_V1_MLP/run-1-epoch200/checkpoint_bestAcc1.pth.tar",
    
    ### for ImageNet-mini:
    #"image_textWordNet": "runs/ImageNet-mini_train_get_Ts_image_textWordNet_V1_MLP/run-1-epoch100/checkpoint_bestAcc1.pth.tar",
    
    ### for CIFAR100_imb100:
    "image_textWordNet": "runs/CIFAR100_imb100_train_get_Ts_image_textWordNet_V1_MLP/run-1-epoch500/checkpoint_bestAcc1.pth.tar",
    
    ### for ImageNet_LT:
    #"image_textWordNet": "runs/ImageNet_LT_train_get_Tw_image_textWordNet_V1_MLP/run-1-epoch120/checkpoint_bestAcc1.pth.tar",
    
    ### for scene
    #"image_textWordNet": "runs/scene_train_get_Ts_image_textWordNet_V1_MLP/run-1-epoch200/checkpoint_bestAcc1.pth.tar",
    
    ### for UTKFace
    #"image_textWordNet": "runs/UTKFace_train_get_Ts_image_textWordNet_V1_MLP/run-1-epoch200/checkpoint_bestAcc1.pth.tar",
    
}


parser = argparse.ArgumentParser(description='CLIP KD naive v1')
parser.add_argument('--mode', default='distil', choices=modes,
                    help='program mode: ' +
                        ' | '.join(vitnets) +
                        ' (default: train)')
parser.add_argument('--arch', default='resnet18', choices=vitnets,
                    help='model architecture: ' +
                        ' | '.join(vitnets) +
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
parser.add_argument('--T4_add_strong_imgCAT80gt20wordnet',default=False, action='store_true',
                    help="When training the student model, opt for using CLIP img embeddings cat GT class names and wordnet nouns CLIP text embeddings with the T4 teacher.")
parser.add_argument('--T4_add_strong_imgCAT50gt50wordnet',default=False, action='store_true',
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
parser.add_argument('--useWhatModal', default="rawImg", # or text...
                    help='choose which modal, image or text embeddings, to use.')
parser.add_argument('--noise_percent', default=0, type=int,
                    help='the (100) percentage of the training data labels get noised')



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


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_classes=10, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)






def train(args, path, log_file, device):

    if 'CIFAR100' in args.dataset:
        print("Function: load_CIFAR100.")
        # newly modified by Chenqi:
        if args.noise_percent ==0:
            data = {x: Load_CIFAR100(data_root=data_root[args.dataset.split("_")[0]], dataset=args.dataset, phase=x,
                         batch_size=args.batch_size, num_workers=4,
                         shuffle=True if x == 'train' else False)
                    for x in ['train', 'val']} 
        
        elif args.mode == 'train' and args.noise_percent !=0:
            data = {x: Load_CIFAR100_noised(data_root=data_root[args.dataset.split("_")[0]], dataset=args.dataset, phase=x,
                         batch_size=args.batch_size, num_workers=4,
                         noise_percent=args.noise_percent,
                         shuffle=True if x == 'train' else False)
                    for x in ['train', 'val']} 
        elif args.mode == 'distil' and args.noise_percent !=0:
            data = {x: Load_CIFAR100_clean_noised(data_root=data_root[args.dataset.split("_")[0]], dataset=args.dataset, phase=x,
                         batch_size=args.batch_size, num_workers=4,
                         noise_percent=args.noise_percent,
                         shuffle=True if x == 'train' else False)
                    for x in ['train', 'val']}
            
    elif 'ImageNet' in args.dataset:
        print("Function: Load_ImageNet.")
        if args.noise_percent ==0:
            data = {x: Load_ImageNet(data_root=data_root[args.dataset.split("_")[0]], dataset=args.dataset, phase=x,
                        batch_size=args.batch_size, num_workers=4,
                        shuffle=True if x == 'train' else False)
                for x in ['train', 'val']}  
            
    
    
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
    
    if args.mode == 'train':
        if args.useWhatModal == 'rawImg':
            model_ft = timm.create_model(args.arch, pretrained=False, num_classes=class_num)
            
            ckpt_path = 'timm_vit_model_ckpts/' + args.arch + '.bin'
            this_ckpt = torch.load(ckpt_path)
            
            # Remove classifier weights (they won't match due to different num_classes)
            this_ckpt.pop('head.weight', None)
            this_ckpt.pop('head.bias', None)
            
            model_ft.load_state_dict(this_ckpt, strict=False)
            #assert(False)
            
        elif 'image_text' in args.useWhatModal:
            #model_ft = SingleTokenViT(num_classes=class_num)
            if args.arch == 'MLP':
                model_ft = MLPClassifier(num_classes=class_num)
            
    
    elif args.mode == 'distil':
        model_ft = timm.create_model(args.arch, pretrained=False, num_classes=class_num)
        

    model_ft = model_ft.to(device)
    
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
        
        
        if args.T4_add_strong_imgCATgt: #T4_add_weak_imgCATgt:
            T_path.append(teacher_path['image_textGT'])
            key_strings.append('image_textGT')
        #elif args.T4_add_strong_imgCATgt:
        #    T_path.append(teacher_path['image_textGT_s'])
        #    key_strings.append('image_textGT')
        elif args.T4_add_strong_imgCATwordnet or args.T4_add_weak_imgCATwordnet:
            T_path.append(teacher_path['image_textWordNet'])
            key_strings.append('image_textWordNet')
        
        elif args.T4_add_strong_imgCAT10gt90wordnet:
            T_path.append(teacher_path['image_text_10gt90wordnet'])
            key_strings.append('image_text_10gt90wordnet')
        
        
        
        elif args.T4_add_strong_imgCAT20gt80wordnet:
            T_path.append(teacher_path['image_text_20gt80wordnet'])
            key_strings.append('image_text_20gt80wordnet')
        elif args.T4_add_strong_imgCAT50gt50wordnet:
            T_path.append(teacher_path['image_text_50gt50wordnet'])
            key_strings.append('image_text_50gt50wordnet')
        elif args.T4_add_strong_imgCAT80gt20wordnet:
            T_path.append(teacher_path['image_text_80gt20wordnet'])
            key_strings.append('image_text_80gt20wordnet')
        
        elif args.T4_add_strong_imgCATgtNoise:
            #T_path.append(teacher_path['image_text_gtNoise'])
            if args.noise_percent==80:
                T_path.append(teacher_path['image_text_20gt80noise'])
            if args.noise_percent==50:
                T_path.append(teacher_path['image_text_50gt50noise'])
            if args.noise_percent==20:
                T_path.append(teacher_path['image_text_80gt20noise'])
            if args.noise_percent==100:
                T_path.append(teacher_path['image_text_0gt100noise'])
            key_strings.append('image_text_gtNoise')
        
        
        if len(T_path) != args.teacher_num or len(key_strings) != args.teacher_num:
            print("The number of loaded teachers does not match the preset. Please check.")
            assert(False)
        for i in range(args.teacher_num):
            print('----- i = ' + str(i))
            
            key_str = key_strings[i]
            
            if key_str == 'weak' or key_str == 'strong':
                teacher_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=class_num).to(device)
            elif 'image_text' in key_str: #key_str == 'image_textWordNet':
                #teacher_model = SingleTokenViT(num_classes=class_num).to(device)
                #if args.arch == 'MLP':
                teacher_model = MLPClassifier(num_classes=class_num).to(device)
                
            print("The loaded teacher model is:",T_path[i])
            teacher_model.load_state_dict(torch.load(T_path[i])['state_dict'])
            teacher.append(teacher_model)
        
    else:
        teacher = [] 
    
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

    train_model_withCLIP(model_ft, data, optimizer_ft, 
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
        
        #print(path_1)
        #assert(False)

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
    
    
    
    


