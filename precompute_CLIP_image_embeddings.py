#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 13:52:00 2024

@author: ps
"""

# precompute the CLIP_image_embeddings for a dataset and
# save each into one npy file to ensure running efficiency during training.


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import csv
import os
import numpy as np

import pickle
#from trainer_regularizerV2 import retrieve_text_v2
from TACmodels import CLIPModel

from data_loader.DataLoaderCIFAR import Load_CIFAR100
from data_loader.DataLoaderImageNet import Load_ImageNet



data_root = {
        'CIFAR100':  '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img',
        'ImageNet':'/home/ps/scratch/KD_imbalance/LFME/my_data/ILSVRC/Data/CLS-LOC',
        }



dataset = 'ImageNet'
batch_size = 1024
train_add_strong = True
output_root = './data/ImageNet/CLIP_image_embeddings/'



def save_CLIP_image_embeddings_all(model_CLIP, data_loader, device, isTrain):
    print('---- getting CLIP image embeddings for the whole dataset...')
    
    features_list = []
    #labels_list = []
    
    #for batch_idx, (imgs_weak, imgs_strong, labels) in enumerate(data_loader):
    for batch_idx, this_batch in enumerate(data_loader):
        print(batch_idx)
        imgs_weak, imgs_strong = this_batch[0], this_batch[1]
        #this_path = this_batch[3]
        
        #labels = labels.to(device)
        if isTrain and train_add_strong:
            inputs = imgs_strong.to(device)
        else: #if isVal or train_add_weak:
            inputs = imgs_weak.to(device)
            
        # newly modified by Chenqi: for CLIP-KD:
        with torch.no_grad():
            this_feature = model_CLIP.encode_image(inputs)
        features_list.append(this_feature.cpu().numpy())
        #labels_list.append(labels.numpy())
    
    features_all = np.concatenate(features_list, axis=0)
    #labels_all = np.concatenate(labels_list, axis=0)
    #print("Feature shape:", features_all.shape) #, "Label shape:", labels_all.shape)
    
    features_all = features_all / np.linalg.norm(
                        features_all, axis=1, keepdims=True
                        )
    
    print('features_all.shape = ' + str(features_all.shape)) # (50000, 512) for val;
    
    if isTrain:
        npy_file_name = 'train.npy'
    else:
        npy_file_name = 'val.npy'
    
    save_path = os.path.join(output_root, npy_file_name)
    np.save(save_path, features_all)
    
    return #features_all



"""
def save_CLIP_image_embeddings_all(model_CLIP, data_loader, device, isTrain):
    print('---- getting CLIP image embeddings for the whole dataset...')
    
    features_list = []
    #labels_list = []
    
    #for batch_idx, (imgs_weak, imgs_strong, labels) in enumerate(data_loader):
    for batch_idx, this_batch in enumerate(data_loader):
        print(batch_idx)
        imgs_weak, imgs_strong = this_batch[0], this_batch[1]
        this_path = this_batch[3]
        
        #labels = labels.to(device)
        if isTrain and train_add_strong:
            inputs = imgs_strong.to(device)
        else: #if isVal or train_add_weak:
            inputs = imgs_weak.to(device)
            
        # newly modified by Chenqi: for CLIP-KD:
        with torch.no_grad():
            this_feature = model_CLIP.encode_image(inputs)
        
        #print('****this_feature.shape = ' + str(this_feature.shape))
        #print('this_feature = ' + str(this_feature))
        
        #image_embeddings = this_feature / np.linalg.norm(
        #                    this_feature, axis=1, keepdims=True
        #                    )
        
        # Compute norm per row (per sample)
        norms = this_feature.norm(dim=1, keepdim=True)  # shape: [batch_size, 1]
        
        image_embeddings = this_feature / norms
        
        #print('****image_embeddings.shape = ' + str(image_embeddings.shape))
        #print('image_embeddings = ' + str(image_embeddings))
        
        for idx, img_path in enumerate(this_path):
            # Get the relative path with respect to the known training directory:
            # Assuming your original training images are under:
            # '/home/ps/scratch/KD_imbalance/LFME/my_data/ILSVRC/Data/CLS-LOC/train/'
            train_root = '/home/ps/scratch/KD_imbalance/LFME/my_data/ILSVRC/Data/CLS-LOC/' # train/ or val/
            rel_path = os.path.relpath(img_path, train_root)  
            # rel_path might be 'train/n01440764/n01440764_8529.JPEG'
            
            # Create the new path under output_root, changing extension to '.npy'
            base, ext = os.path.splitext(rel_path)
            new_rel_path = base + '.npy'  # e.g. 'n01440764/n01440764_8529.npy'
            save_path = os.path.join(output_root, new_rel_path)
            
            #print('save_path = ' + str(save_path))
            
            # Make sure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
            # Extract the embedding from the tensor (assume image_embeddings is on CPU or move it)
            embedding = image_embeddings[idx].cpu().numpy()  # shape [512,]
            
            # Save the embedding as a .npy file
            np.save(save_path, embedding)
            
            #assert(False)
        
        # for debug!:
        #print('this_feature.shape = ' + str(this_feature.shape))
        #print('this_path = ' + str(this_path))
        #assert(False)
        
    
    return #features_all
"""




if __name__ == '__main__':
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_CLIP = CLIPModel(model_name="ViT-B/32").to(device)
    model_CLIP.eval()
    
    # load the dataset:
    train_loader = Load_ImageNet(data_root=data_root[dataset.split("_")[0]], dataset=dataset, phase='train',
                batch_size=batch_size, num_workers=4,
                shuffle=False)
    
    val_loader = Load_ImageNet(data_root=data_root[dataset.split("_")[0]], dataset=dataset, phase='val',
                batch_size=batch_size, num_workers=4,
                shuffle=False) 
    
    #save_CLIP_image_embeddings_all(model_CLIP, val_loader, device, isTrain=False)
    save_CLIP_image_embeddings_all(model_CLIP, train_loader, device, isTrain=True)
    
    
    
    