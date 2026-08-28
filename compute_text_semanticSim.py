#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:21:08 2024

@author: ps
"""

# For the k-NN retrieved WordNet nouns (both the pre-trained one and the best 
# one after loss optimization), compute their semantic distance between them and
# the gt's. Specifically:
# (1) Path Similarity between the retrieved pre-trained WordNet nouns and the 
#     gt class names, using wn.similarity.path --> this is to validate that 
#     the cosine similarity in (2) is consistent with the path similarity (by
#     comparing this (1) with (2.1)), so that the (2.2) metric is also representative
#     enough for measuring the semantic similarity between the retrieved best 
#     WordNet nouns and the gt's.
# (2) Cosine Similarity: (2.1) between the retrieved pre-trained WordNet nouns_embeddings
#     and the gt_embeddings; (2.2) between the best WordNet nouns_embeddings
#     and the gt_embeddings.
# NOTE: both (1) and (2) are averaged through all the training set.

# I want this to act as a metric of how strong the WordNet-relaxed
# regularizer is!!!


import wn
from wn.similarity import path

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import csv
import os
import numpy as np

from data_loader.DataLoaderCIFAR import Load_CIFAR100
import pickle
#from trainer_regularizerV2 import retrieve_text_v2
from TACmodels import CLIPModel


data_root = {
        'CIFAR100':  '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img',
        'ImageNet':'/home/ps/scratch/KD_imbalance/LFME/my_data/ILSVRC/Data/CLS-LOC',
        }


dataset = 'CIFAR100'
batch_size = 128
embDir_best = 'runs/CIFAR100_train_get_Ts_image_textWordNet_V2/run-1-epoch61/nouns_embeddings_bestAcc1.pt'
from WordNet_selected_nouns_CIFAR100 import WordNet_selected_nouns








def get_class_names():
    pkl_file_path = './data/' + dataset + '/infoMeta_dict.pkl'
    #print(pkl_file_path)
    assert(os.path.exists(pkl_file_path))
    with open(pkl_file_path, 'rb') as f:
        infoMeta_dict = pickle.load(f)
    meta_dict = infoMeta_dict['meta_dict']
    if "CIFAR100" in dataset:
        fine_label_names = meta_dict[b'fine_label_names']
    elif "ImageNet" in dataset:
        fine_label_names = meta_dict['fine_label_names']
    
    # for the CIFAR, we need to convert the byte class names into strings!:
    nouns = []
    for item_byte in fine_label_names:
        if "CIFAR100" in dataset:
            item_string = item_byte.decode('utf-8')
        elif "ImageNet" in dataset:
            item_string = item_byte
        nouns.append(item_string)
    
    return nouns


def get_CLIP_image_embeddings(model_CLIP, inputs, device):
    #print('---- getting CLIP image embeddings...')
    
    features_list = []
    
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
    
    return features_all


def retrieve_text_top1(images_embedding, nouns_embedding):
    tau = 0.005
    
    # Convert images_embedding to torch tensor and move to GPU
    images_embedding = torch.from_numpy(images_embedding).cuda().half()
    image_num = images_embedding.shape[0]
    
    top1_indices_list = []
    batch_size = 8192
    
    # Compute similarity over the dataset in batches
    for i in range(image_num // batch_size + 1):
        start = i * batch_size
        end = start + batch_size
        if end > image_num:
            end = image_num
        if start == end:
            break  # No more samples
        
        # Compute similarity scores: (batch_size, N) where N is the number of nouns
        similarity = torch.matmul(images_embedding[start:end], nouns_embedding.T)
        
        #print('similarity = ' + str(similarity))
        #print("Max similarity values per image:", similarity.max(dim=1).values)
        #print("Indices of max similarity per image:", similarity.max(dim=1).indices)
        
        similarity_sorted = torch.sort(similarity[0], descending=True)
        print("Top similarities for first image:", similarity_sorted.values[:10], similarity_sorted.indices[:10])
        
        #print("Similarity for first image:", similarity[0])
        #print('similarity[0,606] = ' + str(similarity[0,606]))
        
        assert(False)
        
        # Instead of softmax + weighted sum, we directly find the top-1 index
        # similarity shape: [batch_subset, nouns_num]
        values, indices = similarity.max(dim=1)  # values: [batch_subset], indices: [batch_subset]
        # indices now contains the top-1 noun index for each image in the batch subset
        
        top1_indices_list.append(indices.cpu())
        
    # Concatenate the indices for all batches
    top1_indices = torch.cat(top1_indices_list, dim=0)  # shape: [image_num]
    
    # Retrieve the top-1 noun embeddings for each image
    # Note: top1_indices[i] gives the index of the best matching noun for image i
    # nouns_embedding shape: [N, embedding_dim]
    # top1_indices shape: [image_num]
    selected_nouns_embedding = nouns_embedding[top1_indices]  # shape: [image_num, embedding_dim]

    # Move selected_nouns_embedding to CPU if needed
    selected_nouns_embedding = selected_nouns_embedding.cpu().float()

    # Optionally, normalize if desired
    selected_nouns_embedding = F.normalize(selected_nouns_embedding, dim=1)
    
    return selected_nouns_embedding, top1_indices




if __name__ == '__main__':
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_CLIP = CLIPModel(model_name="ViT-B/32").to(device)
    model_CLIP.eval()
    
    # load the training set:
    train_loader = Load_CIFAR100(data_root=data_root[dataset.split("_")[0]], dataset=dataset, phase='train_noAug',
                 batch_size=batch_size, num_workers=4,
                 shuffle=True)
    
    # get gt class names:
    all_gt_class_names = get_class_names()
    
    # 下载并加载 Open English WordNet 数据
    wn.download('oewn:2023')
    en = wn.Wordnet('oewn:2023')
    
    # get the gt class names text embeddings:
    result_dir = './data/' + dataset + "/GTnouns_embedding_ensemble.npy"
    assert(os.path.exists(result_dir))
    embeddings = np.load(result_dir)
    text_embedding_train_allCls_gt = embeddings / np.linalg.norm(
                        embeddings, axis=1, keepdims=True
                        )
    
    
    # get the pre-trained nouns_embedding:
    result_dir = './data/' + dataset + "/WordNet_filtered_nouns_embedding.npy"
    assert(os.path.exists(result_dir))
    nouns_embedding_selected = np.load(result_dir)
    nouns_embedding_pretrained = nouns_embedding_selected / np.linalg.norm(
                        nouns_embedding_selected, axis=1, keepdims=True
                        )
    nouns_embedding_pretrained = torch.from_numpy(nouns_embedding_pretrained).cuda().half()
    
    # get the best nouns_embedding:
    assert(os.path.exists(embDir_best))
    nouns_embedding_best = torch.load(embDir_best)
    nouns_embedding_best = nouns_embedding_best.cuda().half()
    
    #print(torch.all(nouns_embedding_pretrained == nouns_embedding_best))
    #print('***** nouns_embedding_pretrained = ' + str(nouns_embedding_pretrained))
    #print('***** nouns_embedding_best = ' + str(nouns_embedding_best))
    
    nouns_embedding = nouns_embedding_best
    #print("Mean of nouns_embedding:", nouns_embedding.mean(dim=0))
    #print("Std of nouns_embedding:", nouns_embedding.std(dim=0))
    # Check if rows are distinct
    #diff = (nouns_embedding.unsqueeze(1) - nouns_embedding.unsqueeze(0)).abs().sum(dim=-1).mean()
    #print("Average pairwise difference:", diff)
    
    #print("Embedding at index 606:", nouns_embedding[606])
    #print("Embedding at index 100:", nouns_embedding[100])
    
    #assert(False)
    
    
    for batch_idx, (imgs_weak, _, labels) in enumerate(train_loader):
        
        imgs_weak = imgs_weak.to(device)
        
        image_embedding_train_1batch = get_CLIP_image_embeddings(model_CLIP, imgs_weak, device)
        
        text_gtClass_embeddings = text_embedding_train_allCls_gt[labels, :]
        
        #retrieved_nouns_embedding_pretrained, top1_indices_pretrained = retrieve_text_top1(image_embedding_train_1batch, nouns_embedding_pretrained)
        retrieved_nouns_embedding_best, top1_indices_best = retrieve_text_top1(image_embedding_train_1batch, nouns_embedding_best)
        
        
        
        
        """
        
        # (1) compute path similarity:
        gt_class_names = np.array(all_gt_class_names)[labels]
        wnPretrain_class_names = WordNet_selected_nouns[top1_indices_pretrained]
        # Nevertheless, I decide to also check the path-sim for the ~best nouns embeddings,
        # though I know that this may be meaning less since the nouns_embedding_best
        # are NO longer corresponding to the WordNet_selected_nouns!:
        wnBest_class_names = WordNet_selected_nouns[top1_indices_best]
        
        print('***** retrieved_nouns_embedding_pretrained = ' + str(retrieved_nouns_embedding_pretrained))
        print('***** retrieved_nouns_embedding_best = ' + str(retrieved_nouns_embedding_best))
        
        print('***** top1_indices_pretrained = ' + str(top1_indices_pretrained))
        print('***** top1_indices_best = ' + str(top1_indices_best))
        
        
        print('***** gt_class_names = ' + str(gt_class_names))
        print('len(gt_class_names) = ' + str(len(gt_class_names))) # 128
        print('***** wnPretrain_class_names = ' + str(wnPretrain_class_names))
        print('len(wnPretrain_class_names) = ' + str(len(wnPretrain_class_names))) # 128
        print('***** wnBest_class_names = ' + str(wnBest_class_names))
        print('len(wnBest_class_names) = ' + str(len(wnBest_class_names))) # 128
        
        assert(False)
        
        
        
        # (2) compute cosine similarity:
        """




