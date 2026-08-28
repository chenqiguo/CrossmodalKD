#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:58:51 2023

@author: ps
"""

import copy
import datetime
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import wasserstein_distance as w_distance
from torchvision.transforms.functional import crop
import pickle

import numpy as np
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif #mutual_info_regression
from sklearn.metrics import mutual_info_score

from TACmodels import CLIPModel
import clip
import pandas as pd

import faiss

import matplotlib.pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)    


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def loss_kd(outputs, teacher_outputs, labels, temp, alpha):
    beta = 1. - alpha
    q = F.log_softmax(outputs/temp, dim=1)
    p = F.softmax(teacher_outputs/temp, dim=1)
    soft_loss = nn.KLDivLoss(reduction='batchmean')(q, p) * temp ** 2 
    hard_loss = nn.CrossEntropyLoss()(outputs, labels)
    KD_loss = alpha * hard_loss + beta * soft_loss 

    return KD_loss

def KL_divergence(model1_logits, model2_logits):

    probs2 = F.softmax(model2_logits, dim=1)
    log_probs1 = F.log_softmax(model1_logits, dim=1, dtype=torch.double)
    kl_div = F.kl_div(log_probs1, probs2, reduction='batchmean', log_target=False) * 10 
    return kl_div.item()


def get_logfile_name(path):
    get_time = datetime.datetime.now().strftime('%b%d_%H-%M') # 月 日 时 分
    file_name = get_time + '_log.txt'
    
    if not os.path.exists(path):  
        os.makedirs(path)  
        
    return os.path.join(path, file_name)


def print_write(print_str, log_file):
    print(*print_str)
    with open(log_file, 'a') as f:
        print(*print_str, file=f)



def get_fidelity(model1_logits, model2_logits):
    probs1 = F.softmax(model1_logits, dim=1)
    probs2 = F.softmax(model2_logits, dim=1)

    fidelity_num = torch.sum(torch.argmax(probs1, dim=1) == torch.argmax(probs2, dim=1))
    """
    print("***** debug 1")
    print(len(model1_logits)) # 128
    assert(False)
    """
    fidelity = fidelity_num.item() / len(model1_logits)
    
    return fidelity, fidelity_num

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res

def get_pred(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t().squeeze(0)
        
        return pred

def get_entropy(output):
    
    probs = F.softmax(output, dim=1)
    H = entropy(probs.detach().cpu(), axis=1)
    H = np.mean(H)
    """
    print("***** debug 2")
    print(probs.cpu().size()) # torch.Size([128, 100])
    #print(probs.cpu())
    #print(H.size)
    print(H)
    assert(False)
    """
    return H


def get_mi(model1_logits, model2_logits):
    _, pred1 = model1_logits.topk(1, 1, True, True)
    #pred1 = pred1.t()
    _, pred2 = model2_logits.topk(1, 1, True, True)
    # print(pred1)
    """
    # NOT used
    probs2 = F.softmax(model2_logits, dim=1)
    mutual_info_regression()
    """
    
    mi = mutual_info_classif(pred2.cpu(), np.squeeze(pred1).cpu()) # 1st arg is feature, 2nd arg is target!
    """
    print("***** debug 3")
    print(pred1.size()) # torch.Size([128, 1])
    print(np.squeeze(pred1).size()) # torch.Size([128])
    print(mi)
    assert(False)
    """
    return mi[0]


def get_mis(model1_logits, model2_logits):
    _, pred1 = model1_logits.topk(1, 1, True, True)
    _, pred2 = model2_logits.topk(1, 1, True, True)
    
    mis = mutual_info_score(np.squeeze(pred1).cpu(), np.squeeze(pred2).cpu())
    """
    print("***** debug 4")
    print(mis)
    assert(False)
    """
    return mis


def get_CLIP_image_embeddings(args, model_CLIP, inputs, device):
    #print('---- getting CLIP image embeddings...')
    
    features_list = []
    #labels_list = []
    
    """
    for batch_idx, (imgs_weak, imgs_strong, labels) in enumerate(data_loader):
        #labels = labels.to(device)
        if args.train_add_strong:
            inputs = imgs_strong.to(device)
        if args.train_add_weak:
            inputs = imgs_weak.to(device)
    """
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
    
    return features_all #, labels_all

def get_CLIP_image_embeddings_all(args, model_CLIP, data_loader, device):
    print('---- getting CLIP image embeddings for the whole training set...')
    
    features_list = []
    #labels_list = []
    
    #for batch_idx, (imgs_weak, imgs_strong, labels) in enumerate(data_loader):
    for batch_idx, this_batch in enumerate(data_loader):
        imgs_weak, imgs_strong = this_batch[0], this_batch[1]
        
        #labels = labels.to(device)
        if args.train_add_strong:
            inputs = imgs_strong.to(device)
        if args.train_add_weak:
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
    
    return features_all


def kmeans(X, cluster_num):
    print("Perform K-means clustering...")
    d = X.shape[1]
    X = X.astype(np.float32)
    kmeans = faiss.Kmeans(d, cluster_num, gpu=True, spherical=True, niter=300, nredo=10)
    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)
    I = I.reshape(-1)
    print("K-means clustering done.")
    return I

SIMPLE_IMAGENET_TEMPLATES = (
    lambda c: f"itap of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"art of the {c}.",
    lambda c: f"a photo of the small {c}.",
)
def get_prompt_GT(words, index, device="cuda"):
    prompt = [SIMPLE_IMAGENET_TEMPLATES[index](word.replace("_"," ")) for word in words]
    #print("prompt = " + str(prompt))
    text = clip.tokenize(prompt, truncate=True).to(device)
    return text
def get_prompt_WordNet(words, index, device="cuda"):
    prompt = [SIMPLE_IMAGENET_TEMPLATES[index](word) for word in words]
    text = clip.tokenize(prompt, truncate=True).to(device)
    return text

def get_CLIP_text_embeddings_WordNet(args, model_CLIP, device, train_loader):
    
    ### Option 1: use the WordNet nouns!
    
    #result_dir = './data/' + args.dataset.split('_')[0] + "/WordNet_filtered_nouns_embedding.npy"
    result_dir = './data/' + args.dataset + "/WordNet_filtered_nouns_embedding.npy"
    if os.path.exists(result_dir):
        nouns_embedding_selected = np.load(result_dir)
        nouns_embedding_selected = nouns_embedding_selected / np.linalg.norm(
                            nouns_embedding_selected, axis=1, keepdims=True
                            )
        return nouns_embedding_selected
    
    
    nouns = pd.read_csv("./data/WordNetNouns.csv").values
    nouns_num = nouns.shape[0]
    batch_size = 2048
    
    nouns_embedding = np.zeros((nouns_num, 512))
    for index in range(len(SIMPLE_IMAGENET_TEMPLATES)):
        features_list = []
        print("Inferring text features for index", index)
        for i in range(nouns_num // batch_size + 1):
            start = i * batch_size
            end = start + batch_size
            if end > nouns_num:
                end = nouns_num
            nouns_batch = nouns[start:end]
            with torch.no_grad():
                prompt = get_prompt_WordNet(nouns_batch[:, 0], index)
                #print("prompt.shape = " + str(prompt.shape)) # torch.Size([100, 77])
                this_feature = model_CLIP.encode_text(prompt)
                features_list.append(this_feature.cpu().numpy())
            if i % 50 == 0:
                print(f"[Completed {i * batch_size}/{nouns_num}]")
        features_index = np.concatenate(features_list, axis=0)
        #print("Feature shape:", features_index.shape)
        nouns_embedding += features_index
    
    # Multi Prompts
    nouns_embedding = nouns_embedding / len(SIMPLE_IMAGENET_TEMPLATES)
    nouns_embedding = nouns_embedding / np.linalg.norm(
                        nouns_embedding, axis=1, keepdims=True
                        )
    
    ## filter these nouns nouns_embedding: referenced from filter_nouns.py in TAC
    images_embedding = get_CLIP_image_embeddings_all(args, model_CLIP, train_loader, device)
    
    nouns_embedding = torch.from_numpy(nouns_embedding).cuda().half()
    nouns_num = nouns_embedding.shape[0]

    images_embedding = torch.from_numpy(images_embedding).cuda().half()
    image_num = images_embedding.shape[0]
    
    cluster_num = 150 # 176
    topK = 5
    preds = kmeans(images_embedding.cpu().numpy(), cluster_num)
    
    image_centers = torch.zeros((cluster_num, 512), dtype=torch.float16).cuda()
    for k in range(cluster_num):
        image_centers[k] = images_embedding[preds == k].mean(dim=0)
    image_centers = F.normalize(image_centers, dim=1)

    similarity = torch.matmul(image_centers, nouns_embedding.T)
    softmax_nouns = torch.softmax(similarity, dim=0).cpu().float()
    class_pred = torch.argmax(softmax_nouns, dim=0).long()

    selected_idx = torch.zeros_like(class_pred, dtype=torch.bool)
    for k in range(cluster_num):
        if (class_pred == k).sum() == 0:
            continue
        class_index = torch.where(class_pred == k)[0]
        softmax_class = softmax_nouns[:, class_index]
        
        #print('softmax_nouns.shape = ' + str(softmax_nouns.shape)) # torch.Size([167, 146347])
        #print('softmax_class.shape = ' + str(softmax_class.shape)) # torch.Size([167, 137])
        #assert(False)
        
        confidence = softmax_class.max(dim=0)[0]
        rank = torch.argsort(confidence, descending=True)
        selected_idx[class_index[rank[:topK]]] = True
    selected_idx = selected_idx.cpu().numpy()

    print(selected_idx.sum(), "nouns selected.")
    nouns_embedding_selected = nouns_embedding[selected_idx]
    
    # newly added: for checking the selected wordnet nouns:
    TAC_class_names = nouns[selected_idx, 0]
    print('TAC_class_names.shape = ' + str(TAC_class_names.shape))
    print('************** the selected WordNet nouns are:')
    print(TAC_class_names)
    print('************** END **************')

    np.save(
        './data/' + args.dataset + "/WordNet_filtered_nouns_embedding.npy", #'./data/' + args.dataset.split('_')[0] + "/WordNet_filtered_nouns_embedding.npy",
        nouns_embedding_selected.cpu().numpy(),
    )
    
    nouns_embedding_selected = nouns_embedding_selected.cpu().numpy()
    nouns_embedding_selected = nouns_embedding_selected / np.linalg.norm(
                        nouns_embedding_selected, axis=1, keepdims=True
                        )
    
    return nouns_embedding_selected


def get_CLIP_text_embeddings_WNtreeV1(args, model_CLIP, device, train_loader):
    
    ### Option 1: use the WordNetTreeV1 nouns!: CLIP img cat wordnet text embedding,
    # note that these wordnet nouns are retrieved from the Tree subset using k-NN
    
    #result_dir = './data/' + args.dataset.split('_')[0] + "/WordNet_filtered_nouns_embedding.npy"
    result_dir = './data/' + args.dataset + "/WNtreeV1_filtered_nouns_embedding.npy"
    if os.path.exists(result_dir):
        nouns_embedding_selected = np.load(result_dir)
        nouns_embedding_selected = nouns_embedding_selected / np.linalg.norm(
                            nouns_embedding_selected, axis=1, keepdims=True
                            )
        return nouns_embedding_selected
    
    
    #nouns = pd.read_csv("./data/WordNetNouns.csv").values
    # 读取用户提供的CSV文件
    input_file = './WordNetTree_maxDepth1/' + args.dataset +'_wordnet_relationships.csv'  # 将此替换为你的文件路径
    assert(os.path.exists(input_file))
    df = pd.read_csv(input_file)
    
    # 提取 Label 和 Related Word 中的所有词
    #words = set(df['CIFAR100 Label']).union(df['Related Word'])
    words_gt = set(df['CIFAR100 Label'])
    words_related = set(df['Related Word'])
    
    #print(words_related)
    #print('len(words_related) = ' + str(len(words_related))) # 5349
    
    nouns = np.array(list(words_related))
    nouns_num = nouns.shape[0] # 5349
    batch_size = 2048
    
    #print('nouns = ' + str(nouns))
    #print('nouns_num = ' + str(nouns_num))
    
    nouns_embedding = np.zeros((nouns_num, 512))
    for index in range(len(SIMPLE_IMAGENET_TEMPLATES)):
        features_list = []
        print("Inferring text features for index", index)
        for i in range(nouns_num // batch_size + 1):
            start = i * batch_size
            end = start + batch_size
            if end > nouns_num:
                end = nouns_num
            nouns_batch = nouns[start:end]
            with torch.no_grad():
                prompt = get_prompt_WordNet(nouns_batch, index)
                #print("prompt.shape = " + str(prompt.shape)) # torch.Size([100, 77])
                this_feature = model_CLIP.encode_text(prompt)
                features_list.append(this_feature.cpu().numpy())
            if i % 50 == 0:
                print(f"[Completed {i * batch_size}/{nouns_num}]")
        features_index = np.concatenate(features_list, axis=0)
        #print("Feature shape:", features_index.shape)
        nouns_embedding += features_index
    
    # Multi Prompts
    nouns_embedding = nouns_embedding / len(SIMPLE_IMAGENET_TEMPLATES)
    nouns_embedding = nouns_embedding / np.linalg.norm(
                        nouns_embedding, axis=1, keepdims=True
                        )
    
    #print('nouns_embedding.shape = ' + str(nouns_embedding.shape)) # (5349, 512)
    
    ## filter these nouns nouns_embedding: referenced from filter_nouns.py in TAC
    images_embedding = get_CLIP_image_embeddings_all(args, model_CLIP, train_loader, device)
    
    nouns_embedding = torch.from_numpy(nouns_embedding).cuda().half()
    nouns_num = nouns_embedding.shape[0]

    images_embedding = torch.from_numpy(images_embedding).cuda().half()
    image_num = images_embedding.shape[0]
    
    cluster_num = 150 # 176
    topK = 5
    preds = kmeans(images_embedding.cpu().numpy(), cluster_num)
    
    image_centers = torch.zeros((cluster_num, 512), dtype=torch.float16).cuda()
    for k in range(cluster_num):
        image_centers[k] = images_embedding[preds == k].mean(dim=0)
    image_centers = F.normalize(image_centers, dim=1)

    similarity = torch.matmul(image_centers, nouns_embedding.T)
    softmax_nouns = torch.softmax(similarity, dim=0).cpu().float()
    class_pred = torch.argmax(softmax_nouns, dim=0).long()

    selected_idx = torch.zeros_like(class_pred, dtype=torch.bool)
    for k in range(cluster_num):
        if (class_pred == k).sum() == 0:
            continue
        class_index = torch.where(class_pred == k)[0]
        softmax_class = softmax_nouns[:, class_index]
        
        #print('softmax_nouns.shape = ' + str(softmax_nouns.shape)) # torch.Size([167, 146347])
        #print('softmax_class.shape = ' + str(softmax_class.shape)) # torch.Size([167, 137])
        #assert(False)
        
        confidence = softmax_class.max(dim=0)[0]
        rank = torch.argsort(confidence, descending=True)
        selected_idx[class_index[rank[:topK]]] = True
    selected_idx = selected_idx.cpu().numpy()

    print(selected_idx.sum(), "nouns selected.")
    nouns_embedding_selected = nouns_embedding[selected_idx]
    
    # newly added: for checking the selected wordnet nouns:
    TAC_class_names = nouns[selected_idx]
    print('TAC_class_names.shape = ' + str(TAC_class_names.shape))
    print('************** the selected WordNet nouns are:')
    print(TAC_class_names)
    print('************** END **************')

    np.save(
        './data/' + args.dataset + "/WNtreeV1_filtered_nouns_embedding.npy", #'./data/' + args.dataset.split('_')[0] + "/WordNet_filtered_nouns_embedding.npy",
        nouns_embedding_selected.cpu().numpy(),
    )
    
    nouns_embedding_selected = nouns_embedding_selected.cpu().numpy()
    nouns_embedding_selected = nouns_embedding_selected / np.linalg.norm(
                        nouns_embedding_selected, axis=1, keepdims=True
                        )
    
    
    
    return nouns_embedding_selected



# Function to process each group and generate the desired output
def generate_related_terms(group):
    # Extract the top-most related terms for each relation
    relation_dict = {
        '上级词': '',
        '下级词': '',
        '部分-整体关系（整体词）': '',
        '部分-整体关系（部分词）': ''
    }
    
    # Iterate over the rows in the group and assign the first encountered term for each relation
    for _, row in group.iterrows():
        if row['Relation'] == '上级词' and not relation_dict['上级词']:
            relation_dict['上级词'] = row['Related Word']
        elif row['Relation'] == '下级词' and not relation_dict['下级词']:
            relation_dict['下级词'] = row['Related Word']
        elif row['Relation'] == '部分-整体关系（整体词）' and not relation_dict['部分-整体关系（整体词）']:
            relation_dict['部分-整体关系（整体词）'] = row['Related Word']
        elif row['Relation'] == '部分-整体关系（部分词）' and not relation_dict['部分-整体关系（部分词）']:
            relation_dict['部分-整体关系（部分词）'] = row['Related Word']
    
    #print(relation_dict)
    
    # Only include terms that are not empty, join them with '_'
    related_terms = []
    
    if relation_dict['上级词']:
        related_terms.append(relation_dict['上级词'])
    if relation_dict['下级词']:
        related_terms.append(relation_dict['下级词'])
    if relation_dict['部分-整体关系（整体词）']:
        related_terms.append(relation_dict['部分-整体关系（整体词）'])
    if relation_dict['部分-整体关系（部分词）']:
        related_terms.append(relation_dict['部分-整体关系（部分词）'])
    
    if not related_terms:
        #print(group['CIFAR100 Label'].iloc[0])
        return group['CIFAR100 Label'].iloc[0]  # Return the CIFAR100 Label from the group
    else:
        #print('_'.join(related_terms))
        return ' '.join(related_terms)
    

def get_CLIP_text_embeddings_WNtreeV2(args, model_CLIP, device, train_loader):
    
    ### Option 1: use the WordNetTreeV2 nouns!: CLIP img cat wordnet text embedding,
    # note that these wordnet nouns are retrieved from the Tree subset simply using
    # related words, constructed as "上级词_下级词_部分词_整体词"
    
    #result_dir = './data/' + args.dataset.split('_')[0] + "/WordNet_filtered_nouns_embedding.npy"
    result_dir = './data/' + args.dataset + "/WNtreeV2_filtered_nouns_embedding.npy"
    if os.path.exists(result_dir):
        nouns_embedding_selected = np.load(result_dir)
        nouns_embedding_selected = nouns_embedding_selected / np.linalg.norm(
                            nouns_embedding_selected, axis=1, keepdims=True
                            )
        return nouns_embedding_selected
    
    
    #nouns = pd.read_csv("./data/WordNetNouns.csv").values
    # 读取用户提供的CSV文件
    input_file = './WordNetTree_maxDepth1/' + args.dataset +'_wordnet_relationships.csv'  # 将此替换为你的文件路径
    assert(os.path.exists(input_file))
    df = pd.read_csv(input_file)
    
    # newly modified by Chenqi:
    # Apply the function to generate related terms and reset the index
    result_df = df.groupby('CIFAR100 Label').apply(generate_related_terms).reset_index(name='Related Terms')
    
    # Show the result DataFrame
    #print(result_df)
    
    # Save the result DataFrame to a CSV file
    result_df.to_csv('./data/' + args.dataset + '/related_terms.csv', index=False)
    
    # Drop duplicate rows since we applied the function to the whole group
    #df_unique = df.drop_duplicates(subset=['CIFAR100 Label'])
    
    # 提取 Label 和 Related Word 中的所有词
    #words = set(df['CIFAR100 Label']).union(df['Related Word'])
    words_gt = result_df['CIFAR100 Label']
    words_relatedTerm = result_df['Related Terms']
    
    #print(words_gt)
    #print(words_relatedTerm)
    
    nouns = np.array(list(words_relatedTerm))
    nouns_num = nouns.shape[0] # 100
    batch_size = 2048
    
    #print('nouns = ' + str(nouns))
    #print('nouns_num = ' + str(nouns_num))
    
    nouns_embedding = np.zeros((nouns_num, 512))
    for index in range(len(SIMPLE_IMAGENET_TEMPLATES)):
        features_list = []
        print("Inferring text features for index", index)
        for i in range(nouns_num // batch_size + 1):
            start = i * batch_size
            end = start + batch_size
            if end > nouns_num:
                end = nouns_num
            nouns_batch = nouns[start:end]
            with torch.no_grad():
                prompt = get_prompt_WordNet(nouns_batch, index)
                #print("prompt.shape = " + str(prompt.shape)) # torch.Size([100, 77])
                this_feature = model_CLIP.encode_text(prompt)
                features_list.append(this_feature.cpu().numpy())
            if i % 50 == 0:
                print(f"[Completed {i * batch_size}/{nouns_num}]")
        features_index = np.concatenate(features_list, axis=0)
        #print("Feature shape:", features_index.shape)
        nouns_embedding += features_index
    
    # Multi Prompts
    nouns_embedding = nouns_embedding / len(SIMPLE_IMAGENET_TEMPLATES)
    nouns_embedding = nouns_embedding / np.linalg.norm(
                        nouns_embedding, axis=1, keepdims=True
                        )
    
    #print('nouns_embedding.shape = ' + str(nouns_embedding.shape)) # (100, 512)
    
    ## NO need to filter these nouns nouns_embedding: 
    # will use them all!

    print("ALL nouns selected.")
    nouns_embedding_selected = nouns_embedding #[selected_idx]
    
    # newly added: for checking the selected wordnet nouns:
    TAC_class_names = nouns #[selected_idx]
    print('TAC_class_names.shape = ' + str(TAC_class_names.shape))
    print('************** the selected WordNet nouns are:')
    print(TAC_class_names)
    print('************** END **************')

    np.save(
        './data/' + args.dataset + "/WNtreeV2_filtered_nouns_embedding.npy", #'./data/' + args.dataset.split('_')[0] + "/WordNet_filtered_nouns_embedding.npy",
        nouns_embedding_selected,
    )
    
    nouns_embedding_selected = nouns_embedding_selected / np.linalg.norm(
                        nouns_embedding_selected, axis=1, keepdims=True
                        )
    
    
    return nouns_embedding_selected


def retrieve_text(args, images_embedding, nouns_embedding):
    # referenced from retrieve_text.py (concat_kmeans.py) in TAC
    # Note: here the images_embedding is already just for one args.batch_size!
    tau = 0.005
    
    nouns_embedding = torch.from_numpy(nouns_embedding).cuda().half()
    nouns_num = nouns_embedding.shape[0]
    images_embedding = torch.from_numpy(images_embedding).cuda().half()
    image_num = images_embedding.shape[0]
    
    retrieval_embeddings = []
    batch_size = 8192
    for i in range(image_num // batch_size + 1):
        start = i * batch_size
        end = start + batch_size
        if end > image_num:
            end = image_num
        #images_batch = images_embedding[start:end]
        similarity = torch.matmul(images_embedding[start:end], nouns_embedding.T)
        similarity = torch.softmax(similarity / tau, dim=1)
        retrieval_embedding = (similarity @ nouns_embedding).cpu()
        retrieval_embeddings.append(retrieval_embedding)
        #if i % 50 == 0:
        #    print(f"[Completed {i * batch_size}/{image_num}]")
    retrieval_embedding = torch.cat(retrieval_embeddings, dim=0).cuda().half()
    retrieval_embedding = F.normalize(retrieval_embedding, dim=1).cpu().numpy()
    
    #print('retrieval_embedding.shape = ' + str(retrieval_embedding.shape)) # (args.batch_size, 512)
    #assert(False)
    
    return retrieval_embedding



def get_CLIP_text_embeddings_GT(args, model_CLIP, device):
    
    ### Option 2: use the gt class names!
    
    #"""
    result_dir = './data/' + args.dataset + "/GTnouns_embedding_ensemble.npy"
    if os.path.exists(result_dir):
        embeddings = np.load(result_dir)
        embeddings = embeddings / np.linalg.norm(
                            embeddings, axis=1, keepdims=True
                            )
        return embeddings
    #"""
    
    pkl_file_path = './data/' + args.dataset + '/infoMeta_dict.pkl'
    #print(pkl_file_path)
    if (os.path.exists(pkl_file_path)):
        with open(pkl_file_path, 'rb') as f:
            infoMeta_dict = pickle.load(f)
        meta_dict = infoMeta_dict['meta_dict']
        if "CIFAR100" in args.dataset:
            fine_label_names = meta_dict[b'fine_label_names']
        elif "ImageNet" in args.dataset:
            fine_label_names = meta_dict['fine_label_names']
    elif 'scene' in args.dataset:
        fine_label_names = ['Buildings', 'Forests', 'Glacier', 'Mountains',
                            'Sea', 'Street']
    elif 'UTKFace' in args.dataset:
        fine_label_names = ['White', 'Black', 'Asian', 'Indian',
                            'Others Hispanic Latino Middle Eastern']
    
    # for the CIFAR, we need to convert the byte class names into strings!:
    #nouns = fine_label_names
    nouns = []
    for item_byte in fine_label_names:
        if "CIFAR100" in args.dataset:
            item_string = item_byte.decode('utf-8')
        else: #elif "ImageNet" in args.dataset:
            item_string = item_byte
        nouns.append(item_string)
    
    nouns_num = len(nouns)
    batch_size = 2048
    
    embeddings = np.zeros((nouns_num, 512))
    for index in range(len(SIMPLE_IMAGENET_TEMPLATES)):
        features_list = []
        print("Inferring text features for index", index)
        for i in range(nouns_num // batch_size + 1):
            start = i * batch_size
            end = start + batch_size
            if end > nouns_num:
                end = nouns_num
            nouns_batch = nouns[start:end]
            with torch.no_grad():
                prompt = get_prompt_GT(nouns_batch, index) #nouns_batch[:, 0]
                #print("prompt.shape = " + str(prompt.shape)) # torch.Size([100, 77])
                this_feature = model_CLIP.encode_text(prompt)
                features_list.append(this_feature.cpu().numpy())
            if i % 50 == 0:
                print(f"[Completed {i * batch_size}/{nouns_num}]")
        features_index = np.concatenate(features_list, axis=0)
        #print("Feature shape:", features_index.shape)
        embeddings += features_index
    
    # Multi Prompts
    embeddings = embeddings / len(SIMPLE_IMAGENET_TEMPLATES)
    np.save('./data/' + args.dataset + "/GTnouns_embedding_ensemble.npy", embeddings)
    
    embeddings = embeddings / np.linalg.norm(
                        embeddings, axis=1, keepdims=True
                        )
    
    #print(nouns)
    #assert(False)
    
    return embeddings



# just for debug:
def visualize_batch(dataloader, num_images=6):
    """
    Visualize 'num_images' samples (images and labels) from a given DataLoader.
    """
    
    class_name_list = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    
    # Get the first batch from the DataLoader
    images, _, labels = next(iter(dataloader))
    
    # Limit to 'num_images' samples
    images, labels = images[:num_images], labels[:num_images]

    # Create a figure to display images
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2.5, 3))
    
    for i in range(num_images):
        ax = axes[i] if num_images > 1 else axes
        # Move channels to last dimension if needed: [C, H, W] -> [H, W, C]
        img_np = images[i].permute(1, 2, 0).cpu().numpy()
        
        ax.imshow(img_np)
        ax.set_title(f"Label: {class_name_list[labels[i].item()]}")
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()
def visualize_batch_withPrediction(images, preds, num_images=6):
    """
    Visualize 'num_images' samples (images and labels) from a given DataLoader.
    """
    
    class_name_list = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    
    # Get the first batch from the DataLoader
    #images, _, _ = next(iter(dataloader))
    
    # Limit to 'num_images' samples
    images, preds = images[:num_images,:,:,:], preds[:num_images]

    # Create a figure to display images
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2.5, 3))
    
    for i in range(num_images):
        ax = axes[i] if num_images > 1 else axes
        # Move channels to last dimension if needed: [C, H, W] -> [H, W, C]
        img_np = images[i].permute(1, 2, 0).cpu().numpy()
        
        ax.imshow(img_np)
        ax.set_title(f"Pred: {class_name_list[preds[i].item()]}")
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()
    

def train_teacher_withCLIP(train_loader, model, model_CLIP, optimizer, scheduler, epoch, device,
          temp, alpha, args):
    #print('here0')
    print('=> training...')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))
    
    model.train()
    end = time.time()
    
    text_embedding_train_allCls_gt = get_CLIP_text_embeddings_GT(args, model_CLIP, device)
    
    text_embedding_train_allCls_wordnet = get_CLIP_text_embeddings_WordNet(args, model_CLIP, device, train_loader) # we treat it as the input argument nouns_embedding !
    
    
    # newly added by Chenqi: 
    if 'text' in args.useWhatModal and 'gt' in args.useWhatModal and ('wordnet' in args.useWhatModal or 'WNtree' in args.useWhatModal):
        #print('here3')
        num_batches = len(train_loader)
        print(f"Total number of batches: {num_batches}")
        
        gt_percent = int(args.useWhatModal.split('gt')[0].split('_')[-1])
        print('gt_percent = ' + str(gt_percent))
        #wordnet_percent = int(args.useWhatModal.split('gt')[-1].split('wordnet')[0])
        #print('wordnet_percent = ' + str(wordnet_percent))
        
        num_gt_batches = int(num_batches * 0.01 * gt_percent)
        print('num_gt_batches = ' + str(num_gt_batches))
        #num_wordnet_batches = int(num_batches * 0.01 * wordnet_percent)
        #print('num_wordnet_batches = ' + str(num_wordnet_batches))
        
        gt_batches_idx_list = list(range(num_gt_batches))
        #print('gt_batches_idx_list = ' + str(gt_batches_idx_list))
        wordnet_batches_idx_list = list(range(num_gt_batches, num_batches))
        #print('wordnet_batches_idx_list = ' + str(wordnet_batches_idx_list))
        #assert(False)
    
    for batch_idx, (imgs_weak, imgs_strong, labels) in enumerate(train_loader):
    #for batch_idx, (imgs_weak, imgs_strong, labels_noise, labels_clean) in enumerate(train_loader):
        #print('here4')
        #labels = labels.to(device)
        if args.train_add_strong:
            inputs = imgs_strong.to(device)
        if args.train_add_weak:
            inputs = imgs_weak.to(device)
        
        
        
        #print('labels_noise = ' + str(labels_noise))
        #print('labels_clean = ' + str(labels_clean))
        #assert(False)
        
        # newly added by Chenqi: 
        my_flag = -1
        if 'text' in args.useWhatModal and 'gt' in args.useWhatModal and ('wordnet' in args.useWhatModal or 'WNtree' in args.useWhatModal):
            if batch_idx in gt_batches_idx_list:
                my_flag = 'gt'
            elif batch_idx in wordnet_batches_idx_list:
                my_flag = 'wordnet'
        
        
        image_embedding_train_1batch = get_CLIP_image_embeddings(args, model_CLIP, inputs, device)
        
        
        data_time.update(time.time() - end)
        with torch.cuda.amp.autocast():
            with torch.set_grad_enabled(True):
                
                #outputs = model(inputs) # orig code
                
                # newly modified by Chenqi: for CLIP-KD,
                # we are using either text or the image embeddings
                if args.useWhatModal == 'image_textGT' or \
                    ('image_text' in args.useWhatModal and my_flag == 'gt'):
                    text_clip_embeddings = text_embedding_train_allCls_gt[labels, :]
                    #text_clip_embeddings_noise = text_embedding_train_allCls_gt[labels_noise, :]
                    #text_clip_embeddings_clean = text_embedding_train_allCls_gt[labels_clean, :]
                    
                    #print('text_clip_embeddings_noise = ' + str(text_clip_embeddings_noise))
                    #print('text_clip_embeddings_clean = ' + str(text_clip_embeddings_clean))
                    #assert(False)
                    
                    image_clip_embeddings = image_embedding_train_1batch
                    clip_embeddings = np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                
                elif args.useWhatModal == 'image_textWordNet' or \
                    ('image_text' in args.useWhatModal and my_flag == 'wordnet'): #and 'WNtreeV2' not in args.useWhatModal)
                    
                    text_clip_embeddings = retrieve_text(args, image_embedding_train_1batch, text_embedding_train_allCls_wordnet)
                    image_clip_embeddings = image_embedding_train_1batch
                    clip_embeddings = np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                
                
                # for debug:
                #print('clip_embeddings.shape = ' + str(clip_embeddings.shape)) # (batch_size, 512) or (batch_size, 1024)
                #print(torch.from_numpy(clip_embeddings).dtype) # torch.float64
                
                
                clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32).to(device)
                #clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                
                labels = labels.to(device)
                outputs = model(clip_embeddings)
                
                criterion = nn.CrossEntropyLoss()

                loss = criterion(outputs,labels)
                
                acc1, _ = accuracy(outputs, labels, topk=(1, 5))
                
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_time.update(time.time() - end)
                end = time.time()
                
                if batch_idx % 10 == 0:
                    progress.display(batch_idx)
        #results_dict['batch_count'] += 1
    return (top1.avg, losses.avg)


def my_train_withCLIP(train_loader, model, model_CLIP, teacher, key_strings, optimizer, scheduler, epoch, device,
          temp, alpha, args):
    
    print('=> training...')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))
    
    model.train()
    end = time.time()
    
    # newly added by Chenqi:
    text_embedding_train_allCls_gt = get_CLIP_text_embeddings_GT(args, model_CLIP, device)
    text_embedding_train_allCls_wordnet = get_CLIP_text_embeddings_WordNet(args, model_CLIP, device, train_loader)
    
    
    # newly added by Chenqi: 
    #if args.T4_add_strong_imgCAT10gt90wordnet:
    if 'text' in args.useWhatModal and 'gt' in args.useWhatModal and 'wordnet' in args.useWhatModal:
        num_batches = len(train_loader)
        print(f"Total number of batches: {num_batches}")
        gt_percent = int(args.useWhatModal.split('gt')[0].split('_')[-1])
        #wordnet_percent = 100 - gt_percent
        
        num_gt_batches = int(num_batches * 0.01 * gt_percent)
        print('num_gt_batches = ' + str(num_gt_batches))
        #num_wordnet_batches = int(num_batches * 0.01 * wordnet_percent)
        #print('num_wordnet_batches = ' + str(num_wordnet_batches))
        
        gt_batches_idx_list = list(range(num_gt_batches))
        #print('gt_batches_idx_list = ' + str(gt_batches_idx_list))
        wordnet_batches_idx_list = list(range(num_gt_batches, num_batches))
        #print('wordnet_batches_idx_list = ' + str(wordnet_batches_idx_list))
        #assert(False)
    
    
    
    for batch_idx, (imgs_weak, imgs_strong, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        imgs_weak, imgs_strong = imgs_weak.to(device), imgs_strong.to(device)
        
        # newly added by Chenqi: 
        my_flag = -1
        if 'text' in args.useWhatModal and 'gt' in args.useWhatModal and 'wordnet' in args.useWhatModal: #if args.useWhatModal != 'rawImg': #args.T4_add_strong_imgCAT10gt90wordnet:
            if batch_idx in gt_batches_idx_list:
                my_flag = 'gt'
            elif batch_idx in wordnet_batches_idx_list:
                my_flag = 'wordnet'
        
        if args.S_add_strong:
            inputs = imgs_strong.to(device)
        elif args.S_add_weak:
            inputs = imgs_weak.to(device)
        
        image_embedding_train_1batch = get_CLIP_image_embeddings(args, model_CLIP, inputs, device)
        
        teacher_outputs = []
        KD_loss = []
        with torch.no_grad():
            if len(teacher) > 0 :
                
                for teacher_num in range(len(teacher)):
                    teacher[teacher_num] = teacher[teacher_num].to(device)
                    
                    # newly modified by Chenqi:
                    key_str = key_strings[teacher_num]
                    
                    if key_str=='weak' or key_str=='strong': # for T1
                        teacher_outputs.append(teacher[teacher_num](inputs)) 
                    
                    elif key_str=='image_textGT' or \
                        ('image_text' in key_str and my_flag == 'gt'): # for T4
                        text_clip_embeddings = text_embedding_train_allCls_gt[labels, :]
                        image_clip_embeddings = image_embedding_train_1batch
                        clip_embeddings = np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                        clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32).to(device)
                        #clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                        teacher_outputs.append(teacher[teacher_num](clip_embeddings))
                    
                    elif key_str=='image_textWordNet' or \
                        ('image_text' in key_str and my_flag == 'wordnet'):
                        from mainCLIPKD_naiveV1_ViTMLP import teacher_path
                        
                        text_clip_embeddings = retrieve_text(args, image_embedding_train_1batch, text_embedding_train_allCls_wordnet)
                        image_clip_embeddings = image_embedding_train_1batch
                        clip_embeddings = np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                        clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32).to(device)
                        #clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                        teacher_outputs.append(teacher[teacher_num](clip_embeddings))
                    
                    #print(key_str)
                    
                    """
                    ## orig code:
                    if args.S_add_strong: # "The data types of inputs for both teacher and student are consistent."
                        teacher_outputs.append(teacher[teacher_num](imgs_strong)) 
                    elif args.S_add_weak:
                        teacher_outputs.append(teacher[teacher_num](imgs_weak)) 
                    """
                    
        labels = labels.to(device)
        with torch.set_grad_enabled(True):
            if args.S_add_strong:
                outputs = model(imgs_strong) 
            elif args.S_add_weak:
                outputs = model(imgs_weak) 
            
            #print('debug')
            #print('len(teacher) = ' + str(len(teacher)))
            #print('len(teacher_outputs) = ' + str(len(teacher_outputs)))
            #assert(False)
            
            criterion = loss_kd
            for i in range(len(teacher)):
                KD_loss.append(criterion(outputs, teacher_outputs[i], labels, temp, alpha))
            loss = sum(KD_loss) / len(KD_loss)
            
            acc1, _ = accuracy(outputs, labels, topk=(1, 5))
            
            # losses.update(loss.item(), inputs.size(0))
            # top1.update(acc1[0], inputs.size(0))
            losses.update(loss.item(), imgs_weak.size(0))
            top1.update(acc1[0], imgs_weak.size(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if batch_idx % 10 == 0:
                progress.display(batch_idx)
                
    return (top1.avg,losses.avg)


def my_train_withCLIP_distillNoise(train_loader, model, model_CLIP, teacher, key_strings, optimizer, scheduler, epoch, device,
          temp, alpha, args):
    
    print('=> training...')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))
    
    model.train()
    end = time.time()
    
    # newly added by Chenqi:
    if args.T3_add_gt_noise or args.T4_add_weak_imgCATgt or args.T4_add_strong_imgCATgt \
        or args.T3_add_70gt30wordnet or args.T3_add_30gt70wordnet \
            or args.T3_add_80gt20wordnet or args.T3_add_20gt80wordnet \
                or args.T4_add_strong_imgCATgtNoise:
        text_embedding_train_allCls_gt = get_CLIP_text_embeddings_GT(args, model_CLIP, device)
    elif args.T3_add_wordnet or args.T4_add_weak_imgCATwordnet or args.T4_add_strong_imgCATwordnet \
        or args.T3_add_70gt30wordnet or args.T3_add_30gt70wordnet \
            or args.T3_add_80gt20wordnet or args.T3_add_20gt80wordnet:
        text_embedding_train_allCls_wordnet = get_CLIP_text_embeddings_WordNet(args, model_CLIP, device, train_loader)
    
    # newly added by Chenqi: 
    if args.T3_add_70gt30wordnet or args.T3_add_30gt70wordnet \
        or args.T3_add_80gt20wordnet or args.T3_add_20gt80wordnet:
        num_batches = len(train_loader)
        print(f"Total number of batches: {num_batches}")
        
        if args.T3_add_70gt30wordnet:
            gt_percent = 70
            wordnet_percent = 30
        elif args.T3_add_30gt70wordnet:
            gt_percent = 30
            wordnet_percent = 70
        elif args.T3_add_80gt20wordnet:
            gt_percent = 80
            wordnet_percent = 20
        elif args.T3_add_20gt80wordnet:
            gt_percent = 20
            wordnet_percent = 80
        
        num_gt_batches = int(num_batches * 0.01 * gt_percent)
        print('num_gt_batches = ' + str(num_gt_batches))
        num_wordnet_batches = int(num_batches * 0.01 * wordnet_percent)
        print('num_wordnet_batches = ' + str(num_wordnet_batches))
        
        gt_batches_idx_list = list(range(num_gt_batches))
        #print('gt_batches_idx_list = ' + str(gt_batches_idx_list))
        wordnet_batches_idx_list = list(range(num_gt_batches, num_batches))
        #print('wordnet_batches_idx_list = ' + str(wordnet_batches_idx_list))
        #assert(False)
    
    
    for batch_idx, (imgs_weak, imgs_strong, labels_clean, labels_noised) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        imgs_weak, imgs_strong = imgs_weak.to(device), imgs_strong.to(device)
        
        # newly added by Chenqi: 
        my_flag = -1
        if args.T3_add_70gt30wordnet or args.T3_add_30gt70wordnet \
            or args.T3_add_80gt20wordnet or args.T3_add_20gt80wordnet:
            if batch_idx in gt_batches_idx_list:
                my_flag = 'gt'
            elif batch_idx in wordnet_batches_idx_list:
                my_flag = 'wordnet'
        
        if args.S_add_strong:
            inputs = imgs_strong.to(device)
        elif args.S_add_weak:
            inputs = imgs_weak.to(device)
        if args.T2_add_weak_CLIPimg or args.T2_add_strong_CLIPimg or args.T3_add_wordnet or \
            args.T4_add_weak_imgCATgt or args.T4_add_weak_imgCATwordnet or \
                args.T4_add_strong_imgCATgt or args.T4_add_strong_imgCATwordnet or \
                    my_flag == 'wordnet' or\
                        args.T4_add_strong_imgCATgtNoise:
             image_embedding_train_1batch = get_CLIP_image_embeddings(args, model_CLIP, inputs, device)
        
        teacher_outputs = []
        KD_loss = []
        with torch.no_grad():
            if len(teacher) > 0 :
                
                for teacher_num in range(len(teacher)):
                    teacher[teacher_num] = teacher[teacher_num].to(device)
                    
                    # newly modified by Chenqi:
                    key_str = key_strings[teacher_num]
                    
                    if key_str=='weak' or key_str=='strong': # for T1
                        teacher_outputs.append(teacher[teacher_num](inputs)) 
                    elif key_str=='weak_CLIPimg' or key_str=='strong_CLIPimg': #teacher_num==1 and args.T2_add_weak_CLIPimg: # for T2
                        clip_embeddings = image_embedding_train_1batch
                        clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32).to(device)
                        clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                        teacher_outputs.append(teacher[teacher_num](clip_embeddings)) 
                    elif key_str=='gt_CLIPtext' or \
                        (key_str=='70gt30wordnet' and my_flag == 'gt') or \
                            (key_str=='30gt70wordnet' and my_flag == 'gt') or \
                                (key_str=='80gt20wordnet' and my_flag == 'gt') or \
                                    (key_str=='20gt80wordnet' and my_flag == 'gt'): #teacher_num==2 and args.T3_add_gt: # for T3
                        clip_embeddings = text_embedding_train_allCls_gt[labels_noised, :]
                        clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32).to(device)
                        clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                        teacher_outputs.append(teacher[teacher_num](clip_embeddings))
                    elif key_str=='wordnet_CLIPtext' or \
                        (key_str=='70gt30wordnet' and my_flag == 'wordnet') or \
                            (key_str=='30gt70wordnet' and my_flag == 'wordnet') or \
                                (key_str=='80gt20wordnet' and my_flag == 'wordnet') or \
                                    (key_str=='20gt80wordnet' and my_flag == 'wordnet'): #teacher_num==2 and args.T3_add_wordnet: # still for T3
                        clip_embeddings = retrieve_text(args, image_embedding_train_1batch, text_embedding_train_allCls_wordnet)
                        clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32).to(device)
                        clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                        teacher_outputs.append(teacher[teacher_num](clip_embeddings))
                    elif key_str=='image_textGT' or key_str=='image_text_gtNoise': # for T4
                        text_clip_embeddings = text_embedding_train_allCls_gt[labels_noised, :]
                        image_clip_embeddings = image_embedding_train_1batch
                        clip_embeddings = np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                        clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32).to(device)
                        clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                        teacher_outputs.append(teacher[teacher_num](clip_embeddings))
                    elif key_str=='image_textWordNet': # still for T4
                        text_clip_embeddings = retrieve_text(args, image_embedding_train_1batch, text_embedding_train_allCls_wordnet)
                        image_clip_embeddings = image_embedding_train_1batch
                        clip_embeddings = np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                        clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32).to(device)
                        clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                        teacher_outputs.append(teacher[teacher_num](clip_embeddings))
                    
                    """
                    ## orig code:
                    if args.S_add_strong: # "The data types of inputs for both teacher and student are consistent."
                        teacher_outputs.append(teacher[teacher_num](imgs_strong)) 
                    elif args.S_add_weak:
                        teacher_outputs.append(teacher[teacher_num](imgs_weak)) 
                    """
                    
        labels = labels_clean.to(device)
        with torch.set_grad_enabled(True):
            if args.S_add_strong:
                outputs = model(imgs_strong) 
            elif args.S_add_weak:
                outputs = model(imgs_weak) 
            
            criterion = loss_kd
            for i in range(len(teacher)):
                KD_loss.append(criterion(outputs, teacher_outputs[i], labels, temp, alpha))
            loss = sum(KD_loss) / len(KD_loss)
            
            acc1, _ = accuracy(outputs, labels, topk=(1, 5))
            
            # losses.update(loss.item(), inputs.size(0))
            # top1.update(acc1[0], inputs.size(0))
            losses.update(loss.item(), imgs_weak.size(0))
            top1.update(acc1[0], imgs_weak.size(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if batch_idx % 10 == 0:
                progress.display(batch_idx)
                
    return (top1.avg,losses.avg)


def my_validate_withCLIP(args, val_loader, model, teacher, model_CLIP, device):
    #print('here0')
    print('=> validating...')
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')
    
    # newly modified by Chenqi: for CLIP-KD.
    # referenced from image_embedding and text_embedding in TAC:
    text_embedding_train_allCls_gt = get_CLIP_text_embeddings_GT(args, model_CLIP, device)
    
    text_embedding_train_allCls_wordnet = get_CLIP_text_embeddings_WordNet(args, model_CLIP, device, val_loader)
    
    
    # newly added by Chenqi: 
    if 'text' in args.useWhatModal and 'gt' in args.useWhatModal and ('wordnet' in args.useWhatModal or 'WNtree' in args.useWhatModal):
        #print('here3')
        num_batches = len(val_loader)
        print(f"Total number of batches: {num_batches}")
        
        gt_percent = int(args.useWhatModal.split('gt')[0].split('_')[-1])
        print('gt_percent = ' + str(gt_percent))
        #wordnet_percent = int(args.useWhatModal.split('gt')[-1].split('wordnet')[0])
        #print('wordnet_percent = ' + str(wordnet_percent))
        
        num_gt_batches = int(num_batches * 0.01 * gt_percent)
        print('num_gt_batches = ' + str(num_gt_batches))
        #num_wordnet_batches = int(num_batches * 0.01 * wordnet_percent)
        #print('num_wordnet_batches = ' + str(num_wordnet_batches))
        
        gt_batches_idx_list = list(range(num_gt_batches))
        #print('gt_batches_idx_list = ' + str(gt_batches_idx_list))
        wordnet_batches_idx_list = list(range(num_gt_batches, num_batches))
        #print('wordnet_batches_idx_list = ' + str(wordnet_batches_idx_list))
        #assert(False)
    
    model.eval()
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            end = time.time()
            
            for batch_idx, (imgs_weak,_, labels) in enumerate(val_loader):
                inputs = imgs_weak.to(device)
                #print('here4')
                # newly added by Chenqi: 
                my_flag = -1
                if 'text' in args.useWhatModal and 'gt' in args.useWhatModal and ('wordnet' in args.useWhatModal or 'WNtree' in args.useWhatModal):
                    if batch_idx in gt_batches_idx_list:
                        my_flag = 'gt'
                    elif batch_idx in wordnet_batches_idx_list:
                        my_flag = 'wordnet'
                
                image_embedding_val_1batch = get_CLIP_image_embeddings(args, model_CLIP, inputs, device)
                
                #outputs = model(inputs) # orig code
                
                # newly modified by Chenqi: for CLIP-KD,
                # we are using either text or the image embeddings
                if args.useWhatModal == 'image_textGT' or \
                    ('image_text' in args.useWhatModal and my_flag == 'gt'):
                    text_clip_embeddings = text_embedding_train_allCls_gt[labels, :]
                    image_clip_embeddings = image_embedding_val_1batch
                    clip_embeddings = np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                
                elif args.useWhatModal == 'image_textWordNet' or \
                    ('image_text' in args.useWhatModal and my_flag == 'wordnet'):
                    text_clip_embeddings = retrieve_text(args, image_embedding_val_1batch, text_embedding_train_allCls_wordnet)
                    image_clip_embeddings = image_embedding_val_1batch
                    clip_embeddings = np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                
                
                
                # for debug:
                #print('clip_embeddings.shape = ' + str(clip_embeddings.shape)) # (batch_size, 512)
                #assert(False)
                
                clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32).to(device)
                #clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                
                labels = labels.to(device)
                
                outputs = model(clip_embeddings)
                  
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)
                
                acc1, _ = accuracy(outputs, labels, topk=(1, 5))
                
                """
                # for debug:
                this_acc1_value = acc1.cpu().detach().item()
                print('********* acc1 = ' + str(this_acc1_value)) 
                if this_acc1_value == 100:
                    this_pred = get_pred(outputs, labels, topk=(1, ))
                    #print('@@@@@@@ this_pred = ' + str(this_pred))
                    #print('this_pred.shape = ' + str(this_pred.shape))
                    #print('labels.shape = ' + str(labels.shape))
                    #print('imgs_weak.shape = ' + str(imgs_weak.shape))
                    visualize_batch_withPrediction(imgs_weak, this_pred)
                    assert(False)
                #visualize_batch_withPrediction
                """
                
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % 10 == 0:
                    progress.display(batch_idx)
            
            print(' * Acc@1 {top1.avg:.3f} '
                .format(top1=top1))
            
    return top1.avg, losses.avg


def my_validate(val_loader, model, teacher, device):
    
    print('=> validating...')
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')
    
    model.eval()
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            end = time.time()
            
            #for batch_idx, (imgs_weak,_, labels) in enumerate(val_loader):
            for batch_idx, this_batch in enumerate(val_loader):
                imgs_weak, labels = this_batch[0], this_batch[2]
                
                inputs, labels = imgs_weak.to(device), labels.to(device)

                outputs = model(inputs) 
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)
                
                acc1, _ = accuracy(outputs, labels, topk=(1, 5))
                
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % 10 == 0:
                    progress.display(batch_idx)
            
            print(' * Acc@1 {top1.avg:.3f} '
                .format(top1=top1))
            
    return top1.avg, losses.avg





def train_model_withCLIP(model, 
                dataloaders,
                optimizer, 
                scheduler, 
                tensorboard_writer,
                device,
                temp,
                alpha,
                log_file,
                args,
                key_strings,
                start_epoch=0,
                teacher=None,
                num_epochs=10):
    since = time.time()

    best_epoch = 0
    best_acc1 = 0.0
    
    #stop_calculate_information = False

    history = []
    
    # newly added by Chenqi: for the CLIP-KD!:
    model_CLIP = CLIPModel(model_name="ViT-B/32").to(device)
    model_CLIP.eval()
    
    
    for epoch in range(num_epochs):
        if epoch < start_epoch:
            continue  
        
        scheduler.step()
        
        #if epoch >= args.lr_steps[0] - 1:
        #    stop_calculate_information = True 

        if len(teacher) > 0 :
            
            # newly modified by Chenqi:
            if args.noise_percent ==0:
                acc1_train, loss_train = my_train_withCLIP(dataloaders['train'], model, model_CLIP, teacher, key_strings, optimizer, scheduler, epoch, device, temp, alpha, args)
            else:
                acc1_train, loss_train = my_train_withCLIP_distillNoise(dataloaders['train'], model, model_CLIP, teacher, key_strings, optimizer, scheduler, epoch, device, temp, alpha, args)
            acc1_val, loss_val = my_validate(dataloaders['val'], model, teacher, device)
            #acc1_val_aug, loss_val_aug, results_dict_val = my_validate_aug(dataloaders['val_aug'], model, teacher, device, args, stop_calculate_information)
        else:
            
            # just for debug:
            #visualize_batch(dataloaders['train'], num_images=6)
            #visualize_batch(dataloaders['val'], num_images=6)
            #assert(False)
            
            # below results_dicts are dummy dicts!
            acc1_train, loss_train = train_teacher_withCLIP(dataloaders['train'], model, model_CLIP, optimizer, scheduler, epoch, device, temp, alpha, args)
            acc1_val, loss_val, = my_validate_withCLIP(args, dataloaders['val'], model, teacher, model_CLIP, device)
            # acc1_val_aug, loss_val_aug, results_dict_val = my_validate_aug(dataloaders['val_aug'], model, teacher, device, args)
        
        tensorboard_writer.add_scalar("train/acc",acc1_train,epoch)
        tensorboard_writer.add_scalar("train/loss",loss_train,epoch) 
        tensorboard_writer.add_scalar("val/acc",acc1_val,epoch)
        tensorboard_writer.add_scalar("val/loss",loss_val,epoch)

        #if len(teacher) > 0 :
        #    tensorboard_writer.add_scalar("val/acc_val_aug",acc1_val_aug,epoch)
        #    tensorboard_writer.add_scalar("val/loss_val_aug",loss_val_aug,epoch)

        print_str = [f'{epoch+1}/{num_epochs} Acc@1: {acc1_val}']
        print_write(print_str, log_file)

        this_result = {'epoch': epoch + 1,
                       'acc1_train': acc1_train,
                       'acc1_val': acc1_val,
                       #'results_dict': results_dict
                       }
        #if len(teacher) > 0 :
        #    this_result['results_dict_train'] = results_dict_train
        #    this_result['results_dict_val'] = results_dict_val
            
        history.append(this_result)

        # remember best acc@1 and save checkpoint
        is_best = acc1_val > best_acc1
        best_acc1 = max(acc1_val, best_acc1)
        if is_best:
            best_epoch = epoch + 1
            best_state = {'epoch': epoch + 1,
                          #'arch': args.arch,
                          'state_dict': model.state_dict(),
                          'acc1_train': acc1_train,
                          'best_acc1_val': best_acc1,
                          #'corresponding_acc5': acc5,
                          'optimizer' : optimizer.state_dict(),
                          'lr_scheduler':scheduler.state_dict(),
                        #   'results_dict_train': results_dict_train,
                        #   'results_dict_val': results_dict_val
                          }  
            #if len(teacher) > 0:
            #    best_state['results_dict_train'] = results_dict_train
            #    best_state['results_dict_val'] = results_dict_val
            torch.save(best_state, args.result+'/checkpoint_bestAcc1.pth.tar')

            train_val_acc_gap = acc1_train - best_acc1

        
        if args.lr_steps[0] - 1 == epoch: 
            checkpoint_state = {
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'lr_scheduler':scheduler.state_dict()
                          }  
            torch.save(checkpoint_state, args.result+'/checkpoint_step1.pth.tar')


        checkpoint_state = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'lr_scheduler':scheduler.state_dict()
                        }  
        torch.save(checkpoint_state, args.result+'/checkpoint_each-epoch.pth.tar')

    # save training history to pk:
    f_pkl = open(args.result+'/history.pkl', 'wb')
    pickle.dump(history,f_pkl)
    f_pkl.close()

    time_elapsed = time.time() - since  
    hours = time_elapsed // 3600  
    minutes = (time_elapsed % 3600) // 60  
    seconds = time_elapsed % 60  
    
    print_str = [f'Training complete in {hours:.0f}h {minutes:.0f}m {seconds:.0f}s\n',  
                f"Best validation accuracy is: {best_acc1}\n",
                f"at epoch: {best_epoch}\n",
                f"The accuracy gap between train and val is: {train_val_acc_gap}\n",
                
                ]
    print_write(print_str, log_file)

    return
    
