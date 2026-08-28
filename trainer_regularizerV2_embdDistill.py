#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:16:50 2024

@author: ps
"""


# V2: employ the WordNet-relaxed nouns CLIP text embeddings as regularizer.


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



# Regularization terms:
lambda_hierarchical = 0.2 #0.1 #0.2 #0.05  # Weight for hierarchical loss (you can tune this)
lambda_cosine_reg = 0.005 #0.01 #0.005 #0.02  # Regularization strength (you can tune this)





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
        if 'ImageNet' in args.dataset:
            inputs = imgs_weak.to(device)
        else:
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

    
    # Below is only for checking if we can use KDE to find a good cluster num value.
    # This is referenced from my_filter_nouns_v2.py from the TAC code.
    """
    ############# Code chunk begin: only run once to get cluster_num #############
    # to eliminate the curse of dimensionality and improve the computation efficiency,
    # let's first use PCA on images_embedding to find the most relevant dims:
    pca_feat_num = 5 #7 #10 #5 #3
    #grid_points = 128 #4 # note: this canNOT be too large! --> NOT use. Let's just make the grid automatically get created!
        
    images_embedding_PCA = pca(images_embedding, pca_feat_num)
    print('images_embedding_PCA.shape = ' + str(images_embedding_PCA.shape)) # (50000, pca_feat_num)
    
    from KDEpy import FFTKDE
    x, y = FFTKDE(kernel="gaussian", bw=0.5).fit(images_embedding_PCA).evaluate() #fit(images_embedding_PCA[:100]).evaluate(grid_points)
    print('x.shape = ' + str(x.shape)) # if use grid_points to specify: the 1st dim=grid_points**pca_feat_num, e.g., (128*128*128, pca_feat_num); ow, (1000, pca_feat_num)
    print('y.shape = ' + str(y.shape)) # if use grid_points to specify:  (grid_points**pca_feat_num,); ow, (1000,)
    #print('x[:10] = ' + str(x[:10]))
    
    print('**** y = ' + str(y))
    #for i in range(1000):
    #    print(y[i])
    
    # Belows are referenced from https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    
    ### option 1: we can find the local maxima of y (simply a 1-dim array!)
    ### and use the number of this maxima as the cluster_num for k-means!
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(y, height=1e-8) #10
    print('**** peaks.shape = ' + str(peaks.shape))
    
    import matplotlib.pyplot as plt
    plt.plot(y)
    plt.plot(peaks, y[peaks], "x")
    plt.show()
    
    cluster_num = peaks.shape[0] # 176
    print('**** cluster_num = ' + str(cluster_num))
    assert(False)
    
    ############# Code chunk end ########################## 
    """
    
    images_embedding = torch.from_numpy(images_embedding).cuda().half()
    image_num = images_embedding.shape[0]
    
    
    cluster_num = 20 #10 #300 #150
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
    print(TAC_class_names.tolist())
    print('************** END **************')

    np.save(
        './data/' + args.dataset + "/WordNet_filtered_nouns_embedding.npy", #'./data/' + args.dataset.split('_')[0] + "/WordNet_filtered_nouns_embedding.npy",
        nouns_embedding_selected.cpu().numpy(),
    )
    
    nouns_embedding_selected = nouns_embedding_selected.cpu().numpy()
    nouns_embedding_selected = nouns_embedding_selected / np.linalg.norm(
                        nouns_embedding_selected, axis=1, keepdims=True
                        )
    assert(False)
    return nouns_embedding_selected


def pca(dataMat, topNfeat):
    meanVals = np.mean(dataMat, axis=0) # compute mean for each column (i.e., feat_dim)!
    meanRemoved = dataMat - meanVals  # 标准化（去均值）
    covMat = np.cov(meanRemoved, rowvar=False)
    
    #print('covMat.shape = ' + str(covMat.shape)) # (512, 512)
    #assert(False)
    
    eigVals, eigVets = np.linalg.eig(np.mat(covMat))  # 计算矩阵的特征值和特征向量
    eigValInd = np.argsort(eigVals)  # 将特征值从小到大排序，返回的是特征值对应的数组里的下标
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # 保留最大的前K个特征值
    redEigVects = eigVets[:, eigValInd]  # 对应的特征向量
    lowDDatMat = meanRemoved * redEigVects  # 将数据转换到低维新空间
    # reconMat = (lowDDatMat * redEigVects.T) + meanVals  # 还原原始数据
    return lowDDatMat


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


def retrieve_text_v2(args, images_embedding, nouns_embedding):
    # referenced from retrieve_text() func in V1.
    # Note: here the images_embedding is already just for one args.batch_size!
    tau = 0.005
    
    #nouns_embedding = torch.from_numpy(nouns_embedding).cuda().half()
    #nouns_num = nouns_embedding.shape[0]
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
    retrieval_embedding = F.normalize(retrieval_embedding, dim=1).cpu() #.numpy()
    
    #print('retrieval_embedding.shape = ' + str(retrieval_embedding.shape)) # (args.batch_size, 512)
    #assert(False)
    
    return retrieval_embedding


# Define the hierarchical loss function
def get_hierarchical_loss(gt_embedding, retrieval_embedding):
    """
    Calculate the cosine similarity between the ground truth class embedding
    and the retrieval (WordNet-relaxed) noun embedding.
    """
    similarity = F.cosine_similarity(gt_embedding, retrieval_embedding)
    return (1 - similarity).mean()  # We want to minimize this


def get_CLIP_text_embeddings_GT(args, model_CLIP, device):
    
    ### Option 2: use the gt class names!
    
    result_dir = './data/' + args.dataset + "/GTnouns_embedding_ensemble.npy"
    if os.path.exists(result_dir):
        embeddings = np.load(result_dir)
        embeddings = embeddings / np.linalg.norm(
                            embeddings, axis=1, keepdims=True
                            )
        return embeddings
    
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
    
    return embeddings



def train_teacher_withCLIP_v2(nouns_embedding, nouns_embedding_pretrained, train_loader, model, model_CLIP, optimizer, scheduler, epoch, device,
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
    nouns_embedding.requires_grad = True
    end = time.time()
    
    text_embedding_train_allCls_gt = get_CLIP_text_embeddings_GT(args, model_CLIP, device)
    
    #text_embedding_train_allCls_wordnet = get_CLIP_text_embeddings_WordNet(args, model_CLIP, device, train_loader) # we treat it as the input argument nouns_embedding !
    """
    if 'WNtreeV1' in args.useWhatModal:
        text_embedding_train_allCls_wordnet = get_CLIP_text_embeddings_WNtreeV1(args, model_CLIP, device, train_loader)
    if 'WNtreeV2' in args.useWhatModal:
        text_embedding_train_allCls_WNtreeV2 = get_CLIP_text_embeddings_WNtreeV2(args, model_CLIP, device, train_loader)
    """
    
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
    
    #image_paths = []
    #optimized_nouns_embeddings = []
    for batch_idx, this_batch_ in enumerate(train_loader):
    #for batch_idx, (imgs_weak, imgs_strong, labels_noise, labels_clean) in enumerate(train_loader):
        #print('here4')
        imgs_weak, imgs_strong, labels = this_batch_[0], this_batch_[1], this_batch_[2]
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
        """
        if 'ImageNet' not in args.dataset:
            image_embedding_train_1batch = get_CLIP_image_embeddings(args, model_CLIP, inputs, device)
        else:
            # just load the pretrained image embeddings!:
            image_embedding_train_1batch = this_batch_[3].numpy()
            #print('image_embedding_train_1batch.shape = ' + str(image_embedding_train_1batch.shape)) #(bs, 512)
        #assert(False)
        """
        #text_clip_retrieval_embedding = retrieve_text_v2(args, image_embedding_train_1batch, nouns_embedding)
        
        data_time.update(time.time() - end)
        with torch.cuda.amp.autocast():
            with torch.set_grad_enabled(True):
                
                #outputs = model(inputs) # orig code
                
                assert('image_text' in args.useWhatModal)
                
                # newly modified by Chenqi: for CLIP-KD,
                # we are using either text or the image embeddings
                if args.useWhatModal == 'image_textGT' or \
                    ('image_text' in args.useWhatModal and my_flag == 'gt'):
                    text_clip_embeddings = text_embedding_train_allCls_gt[labels, :]
                    #text_clip_embeddings_noise = text_embedding_train_allCls_gt[labels_noise, :]
                    #text_clip_embeddings_clean = text_embedding_train_allCls_gt[labels_clean, :]
                    image_clip_embeddings = image_embedding_train_1batch
                    clip_embeddings = np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                    clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32).to(device)
                    
                    hierarchical_loss = 0
                    cosine_regularization_loss = 0
                    
                elif args.useWhatModal == 'image_textWordNet' or \
                    ('image_text' in args.useWhatModal and my_flag == 'wordnet'): #and 'WNtreeV2' not in args.useWhatModal)
                    #  or  args.useWhatModal == 'image_textWNtreeV1':
                    text_clip_retrieval_embedding = retrieve_text_v2(args, image_embedding_train_1batch, nouns_embedding)
                    text_clip_embeddings = text_clip_retrieval_embedding.to(torch.float32).to(device)
                    
                    image_clip_embeddings = image_embedding_train_1batch
                    image_clip_embeddings = torch.from_numpy(image_clip_embeddings).to(torch.float32).to(device)
                    
                    clip_embeddings = torch.cat((image_clip_embeddings, text_clip_embeddings), dim=1) #np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                    
                    # Calculate the hierarchical loss for the current batch:
                    # (Note that gt_embeddings and retrieval_embedding are both normalized)
                    gt_embeddings = text_embedding_train_allCls_gt[labels, :]
                    gt_embeddings = torch.from_numpy(gt_embeddings).to(torch.float32).to(device)
                    
                    hierarchical_loss = get_hierarchical_loss(gt_embeddings, text_clip_embeddings)
                    
                    cosine_regularization_loss = (1 - F.cosine_similarity(nouns_embedding, nouns_embedding_pretrained)).mean()
                    # we can also use mse loss for this constraint?!:
                    #F.mse_loss(nouns_embedding, nouns_embedding_pretrained)
                    
                """
                elif ('image_text' in args.useWhatModal and my_flag == 'wordnet' and 'WNtreeV2' in args.useWhatModal) or \
                    args.useWhatModal == 'image_textWNtreeV2':
                    text_clip_embeddings = text_embedding_train_allCls_WNtreeV2[labels, :]
                    image_clip_embeddings = image_embedding_train_1batch
                    clip_embeddings = np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                """
                
                # for debug:
                #print('clip_embeddings.shape = ' + str(clip_embeddings.shape)) # (batch_size, 512) or (batch_size, 1024)
                #print(clip_embeddings.dtype) # torch.float32
                
                #clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32).to(device)
                clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                
                labels = labels.to(device)
                outputs = model(clip_embeddings)
                
                criterion = nn.CrossEntropyLoss()
                
                # Standard classification loss:
                classification_loss = criterion(outputs, labels)
                """
                # Regularization terms:
                lambda_hierarchical = 0.1  # Weight for hierarchical loss (you can tune this)
                lambda_cosine_reg = 0.01  # Regularization strength (you can tune this)
                """
                
                # Now include this regularization loss in the total loss
                loss = classification_loss + \
                    lambda_hierarchical * hierarchical_loss + \
                    lambda_cosine_reg * cosine_regularization_loss
                
                #print('loss = ' + str(loss))
                
                acc1, _ = accuracy(outputs, labels, topk=(1, 5))
                
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                #print("nouns_embedding.grad:", nouns_embedding.grad)
                
                with torch.no_grad():
                    nouns_embedding[:] = F.normalize(nouns_embedding, dim=1)
                
                """
                # newly added by Chenqi:
                # Store the current nouns_embedding (optimized) for this batch
                # Detach the nouns_embedding to avoid saving gradients
                this_optimized_nouns_embedding = nouns_embedding.detach()  # detach to avoid saving gradients
                # Store the embeddings and corresponding paths:
                optimized_nouns_embeddings.append(this_optimized_nouns_embedding.cpu().numpy())  # move to CPU if using CUDA
                image_paths.extend(paths)  # store image paths for this batch
                """
                
                batch_time.update(time.time() - end)
                end = time.time()
                
                if batch_idx % 10 == 0:
                    progress.display(batch_idx)
        #results_dict['batch_count'] += 1
    
    """
    # Stack the embeddings into one array and save to a file
    optimized_nouns_embeddings = np.vstack(optimized_nouns_embeddings)  # Stack the embeddings into a single numpy array
    torch.save(torch.tensor(optimized_nouns_embeddings), f'{output_dir}/nouns_embeddings.pt')
    
    # Save image paths as a DataFrame (you could also use JSON or CSV depending on your preference)
    df = pd.DataFrame({'image_paths': image_paths})
    df.to_csv(f'{output_dir}/image_paths.csv', index=False)
    """
    
    return (top1.avg, losses.avg)


def train_teacher(train_loader, model, optimizer, scheduler, epoch, device,
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
    
    
    for batch_idx, (imgs_weak, imgs_strong, labels) in enumerate(train_loader):
        labels = labels.to(device)
        if args.train_add_strong:
            inputs = imgs_strong.to(device)
        if args.train_add_weak:
            inputs = imgs_weak.to(device)
            
        data_time.update(time.time() - end)
        with torch.cuda.amp.autocast():
            with torch.set_grad_enabled(True):
                outputs = model(inputs)        
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
        
    return (top1.avg, losses.avg)






# orig code
def my_train_withCLIP_v2(train_loader, projector, model, model_CLIP, teacher, key_strings, optimizer, scheduler, epoch, device,
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
    
    #text_embedding_train_allCls_wordnet = get_CLIP_text_embeddings_WordNet(args, model_CLIP, device, train_loader)
    
    
    #if args.T4_add_strong_imgCATWNtreeV1 or args.T4_add_strong_imgCAT10gt90WNtreeV1 or args.T4_add_strong_imgCAT1gt99WNtreeV1:
    #    text_embedding_train_allCls_wordnet = get_CLIP_text_embeddings_WNtreeV1(args, model_CLIP, device, train_loader)
    #if args.T4_add_strong_imgCATWNtreeV2 or args.T4_add_strong_imgCAT10gt90WNtreeV2:
    #    text_embedding_train_allCls_WNtreeV2 = get_CLIP_text_embeddings_WNtreeV2(args, model_CLIP, device, train_loader)
    
    
    # newly added by Chenqi: 
    if args.T4_add_strong_imgCAT10gt90wordnet:
        num_batches = len(train_loader)
        print(f"Total number of batches: {num_batches}")
        gt_percent = 10
        wordnet_percent = 90
        
        num_gt_batches = int(num_batches * 0.01 * gt_percent)
        print('num_gt_batches = ' + str(num_gt_batches))
        num_wordnet_batches = int(num_batches * 0.01 * wordnet_percent)
        print('num_wordnet_batches = ' + str(num_wordnet_batches))
        
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
        if args.T4_add_strong_imgCAT10gt90wordnet:
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
                        
                        clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                        teacher_outputs.append(teacher[teacher_num](clip_embeddings))
                        
                    elif key_str=='image_textWordNet' or \
                        ('image_text' in key_str and my_flag == 'wordnet'): #and 'WNtreeV2' not in key_str) or \
                        #    key_str=='image_textWNtreeV1': # still for T4
                        # load the pre-trained nouns_embeddings:
                        from mainCLIPKD_regularizerV2 import teacher_path
                        embDir = teacher_path[key_str].split('/checkpoint')[0] + '/nouns_embeddings_bestAcc1.pt'
                        assert(os.path.exists(embDir))
                        best_nouns_embedding = torch.load(embDir)
                        text_clip_retrieval_embedding = retrieve_text_v2(args, image_embedding_train_1batch, best_nouns_embedding.to(device))
                        text_clip_embeddings = text_clip_retrieval_embedding.to(torch.float32).to(device)
                        
                        image_clip_embeddings = image_embedding_train_1batch
                        image_clip_embeddings = torch.from_numpy(image_clip_embeddings).to(torch.float32).to(device)
                        
                        clip_embeddings = torch.cat((image_clip_embeddings, text_clip_embeddings), dim=1) #np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                        
                        clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                        teacher_outputs.append(teacher[teacher_num](clip_embeddings))
                    
        labels = labels.to(device)
        with torch.set_grad_enabled(True):
            if args.S_add_strong:
                #outputs = model(imgs_strong) 
                stud_feat, logits = model(imgs_strong, return_logits=True)
            elif args.S_add_weak:
                #outputs = model(imgs_weak) 
                stud_feat, logits = model(imgs_weak, return_logits=True)
            
            
            #print('debug')
            #print('len(teacher) = ' + str(len(teacher)))
            #print('len(teacher_outputs) = ' + str(len(teacher_outputs)))
            #assert(False)
            
            #criterion = loss_kd
            for i in range(len(teacher)):
                #KD_loss.append(criterion(outputs, teacher_outputs[i], labels, temp, alpha)) # orig code
                
                teacher_proj = projector(teacher_outputs[i]) # (B, 512)
                
                embKD_loss = F.mse_loss(stud_feat, teacher_proj)
                ce = nn.CrossEntropyLoss()(logits, labels)
                this_loss = (1.0-alpha) * embKD_loss + alpha * ce
                
                KD_loss.append(this_loss)
                
            
            loss = sum(KD_loss) / len(KD_loss) # orig code: average!
            
            
            ###### For rebuttal: other teacher integration strategy rather than averaging
            
            # 加权平均: 指定与你实验设想对应的权重 --> too simple, NOT used
            #weights = torch.tensor([0.5, 0.5, 2.0], device=KD_loss[0].device)
            # （可选）把权重正规化：0.5+0.5+2 = 3，与平均分母一致
            #weights = weights / weights.sum()         # -> tensor([0.1667, 0.1667, 0.6667])
            
            # 将列表拆成张量后做加权求和
            #stacked_loss = torch.stack(KD_loss)       # shape [3]
            #loss = (weights * stacked_loss).sum()     # 单个标量
            
            
            acc1, _ = accuracy(logits, labels, topk=(1, 5))
            
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




# new ver for rebuttal: 规则‑驱动：按历次验证表现自适应
def my_train_withCLIP_v3(train_loader, weights, model, model_CLIP, teacher, key_strings, optimizer, scheduler, epoch, device,
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
    
    #text_embedding_train_allCls_wordnet = get_CLIP_text_embeddings_WordNet(args, model_CLIP, device, train_loader)
    
    
    #if args.T4_add_strong_imgCATWNtreeV1 or args.T4_add_strong_imgCAT10gt90WNtreeV1 or args.T4_add_strong_imgCAT1gt99WNtreeV1:
    #    text_embedding_train_allCls_wordnet = get_CLIP_text_embeddings_WNtreeV1(args, model_CLIP, device, train_loader)
    #if args.T4_add_strong_imgCATWNtreeV2 or args.T4_add_strong_imgCAT10gt90WNtreeV2:
    #    text_embedding_train_allCls_WNtreeV2 = get_CLIP_text_embeddings_WNtreeV2(args, model_CLIP, device, train_loader)
    
    
    # newly added by Chenqi: 
    if args.T4_add_strong_imgCAT10gt90wordnet:
        num_batches = len(train_loader)
        print(f"Total number of batches: {num_batches}")
        gt_percent = 10
        wordnet_percent = 90
        
        num_gt_batches = int(num_batches * 0.01 * gt_percent)
        print('num_gt_batches = ' + str(num_gt_batches))
        num_wordnet_batches = int(num_batches * 0.01 * wordnet_percent)
        print('num_wordnet_batches = ' + str(num_wordnet_batches))
        
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
        if args.T4_add_strong_imgCAT10gt90wordnet:
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
                        
                        clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                        teacher_outputs.append(teacher[teacher_num](clip_embeddings))
                        
                    elif key_str=='image_textWordNet' or \
                        ('image_text' in key_str and my_flag == 'wordnet'): #and 'WNtreeV2' not in key_str) or \
                        #    key_str=='image_textWNtreeV1': # still for T4
                        # load the pre-trained nouns_embeddings:
                        from mainCLIPKD_regularizerV2 import teacher_path
                        embDir = teacher_path[key_str].split('/checkpoint')[0] + '/nouns_embeddings_bestAcc1.pt'
                        assert(os.path.exists(embDir))
                        best_nouns_embedding = torch.load(embDir)
                        text_clip_retrieval_embedding = retrieve_text_v2(args, image_embedding_train_1batch, best_nouns_embedding.to(device))
                        text_clip_embeddings = text_clip_retrieval_embedding.to(torch.float32).to(device)
                        
                        image_clip_embeddings = image_embedding_train_1batch
                        image_clip_embeddings = torch.from_numpy(image_clip_embeddings).to(torch.float32).to(device)
                        
                        clip_embeddings = torch.cat((image_clip_embeddings, text_clip_embeddings), dim=1) #np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                        
                        clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                        teacher_outputs.append(teacher[teacher_num](clip_embeddings))
                    
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
            
            #loss = sum(KD_loss) / len(KD_loss) # orig code: average!
            
            
            ###### For rebuttal: other teacher integration strategy rather than averaging
            # 规则‑驱动：按历次验证表现自适应
            loss = (weights * torch.stack(KD_loss)).sum()
            
            
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


def my_train_withCLIP_v3_helper(val_loader, model, model_CLIP, teacher, key_strings, optimizer, scheduler, epoch, device, temp, alpha, args):
    print('=> training: helper...')
    
    val_losses = torch.zeros(len(teacher), device=device)
    
    text_embedding_train_allCls_gt = get_CLIP_text_embeddings_GT(args, model_CLIP, device)
    
    with torch.no_grad():
        for batch_idx, this_batch in enumerate(val_loader):
            imgs_weak, labels = this_batch[0], this_batch[2]
            inputs, labels = imgs_weak.to(device), labels.to(device)
            
            out_s = model(inputs)
            
            image_embedding_train_1batch = get_CLIP_image_embeddings(args, model_CLIP, inputs, device)
            
            
            val_teacher_outputs = []
            with torch.no_grad():
                assert(len(teacher) > 0)
                for teacher_num in range(len(teacher)):
                    teacher[teacher_num] = teacher[teacher_num].to(device)
                    
                    # newly modified by Chenqi:
                    key_str = key_strings[teacher_num]
                    
                    if key_str=='weak' or key_str=='strong': # for T1
                        val_teacher_outputs.append(teacher[teacher_num](inputs)) 
                    
                    elif key_str=='image_textGT':
                        #('image_text' in key_str and my_flag == 'gt'): # for T4
                        text_clip_embeddings = text_embedding_train_allCls_gt[labels, :]
                        image_clip_embeddings = image_embedding_train_1batch
                        clip_embeddings = np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                        
                        clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32).to(device)
                        
                        clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                        val_teacher_outputs.append(teacher[teacher_num](clip_embeddings))
                        
                    elif key_str=='image_textWordNet': #or \
                        #('image_text' in key_str and my_flag == 'wordnet'): #and 'WNtreeV2' not in key_str) or \
                        #    key_str=='image_textWNtreeV1': # still for T4
                        # load the pre-trained nouns_embeddings:
                        from mainCLIPKD_regularizerV2 import teacher_path
                        embDir = teacher_path[key_str].split('/checkpoint')[0] + '/nouns_embeddings_bestAcc1.pt'
                        assert(os.path.exists(embDir))
                        best_nouns_embedding = torch.load(embDir)
                        text_clip_retrieval_embedding = retrieve_text_v2(args, image_embedding_train_1batch, best_nouns_embedding.to(device))
                        text_clip_embeddings = text_clip_retrieval_embedding.to(torch.float32).to(device)
                        
                        image_clip_embeddings = image_embedding_train_1batch
                        image_clip_embeddings = torch.from_numpy(image_clip_embeddings).to(torch.float32).to(device)
                        
                        clip_embeddings = torch.cat((image_clip_embeddings, text_clip_embeddings), dim=1) #np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                        
                        clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                        val_teacher_outputs.append(teacher[teacher_num](clip_embeddings))
            
            
            criterion = loss_kd
            for i, t_out in enumerate(val_teacher_outputs):
                val_losses[i] += criterion(out_s, t_out, labels, temp, alpha)
    
    
    
    return val_losses






# new ver for rebuttal: Online 面向‑损失自适应（AdaKD/Inverse‑Loss）
def my_train_withCLIP_v4(train_loader, model, model_CLIP, teacher, key_strings, optimizer, scheduler, epoch, device,
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
    
    #text_embedding_train_allCls_wordnet = get_CLIP_text_embeddings_WordNet(args, model_CLIP, device, train_loader)
    
    
    #if args.T4_add_strong_imgCATWNtreeV1 or args.T4_add_strong_imgCAT10gt90WNtreeV1 or args.T4_add_strong_imgCAT1gt99WNtreeV1:
    #    text_embedding_train_allCls_wordnet = get_CLIP_text_embeddings_WNtreeV1(args, model_CLIP, device, train_loader)
    #if args.T4_add_strong_imgCATWNtreeV2 or args.T4_add_strong_imgCAT10gt90WNtreeV2:
    #    text_embedding_train_allCls_WNtreeV2 = get_CLIP_text_embeddings_WNtreeV2(args, model_CLIP, device, train_loader)
    
    
    # newly added by Chenqi: 
    if args.T4_add_strong_imgCAT10gt90wordnet:
        num_batches = len(train_loader)
        print(f"Total number of batches: {num_batches}")
        gt_percent = 10
        wordnet_percent = 90
        
        num_gt_batches = int(num_batches * 0.01 * gt_percent)
        print('num_gt_batches = ' + str(num_gt_batches))
        num_wordnet_batches = int(num_batches * 0.01 * wordnet_percent)
        print('num_wordnet_batches = ' + str(num_wordnet_batches))
        
        gt_batches_idx_list = list(range(num_gt_batches))
        #print('gt_batches_idx_list = ' + str(gt_batches_idx_list))
        wordnet_batches_idx_list = list(range(num_gt_batches, num_batches))
        #print('wordnet_batches_idx_list = ' + str(wordnet_batches_idx_list))
        #assert(False)
    
    
    ema_loss = torch.zeros(len(teacher), device=device)  # 初始化
    
    for batch_idx, (imgs_weak, imgs_strong, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        imgs_weak, imgs_strong = imgs_weak.to(device), imgs_strong.to(device)
        
        # newly added by Chenqi: 
        my_flag = -1
        if args.T4_add_strong_imgCAT10gt90wordnet:
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
                        
                        clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                        teacher_outputs.append(teacher[teacher_num](clip_embeddings))
                        
                    elif key_str=='image_textWordNet' or \
                        ('image_text' in key_str and my_flag == 'wordnet'): #and 'WNtreeV2' not in key_str) or \
                        #    key_str=='image_textWNtreeV1': # still for T4
                        # load the pre-trained nouns_embeddings:
                        from mainCLIPKD_regularizerV2 import teacher_path
                        embDir = teacher_path[key_str].split('/checkpoint')[0] + '/nouns_embeddings_bestAcc1.pt'
                        assert(os.path.exists(embDir))
                        best_nouns_embedding = torch.load(embDir)
                        text_clip_retrieval_embedding = retrieve_text_v2(args, image_embedding_train_1batch, best_nouns_embedding.to(device))
                        text_clip_embeddings = text_clip_retrieval_embedding.to(torch.float32).to(device)
                        
                        image_clip_embeddings = image_embedding_train_1batch
                        image_clip_embeddings = torch.from_numpy(image_clip_embeddings).to(torch.float32).to(device)
                        
                        clip_embeddings = torch.cat((image_clip_embeddings, text_clip_embeddings), dim=1) #np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                        
                        clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                        teacher_outputs.append(teacher[teacher_num](clip_embeddings))
                    
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
            
            #loss = sum(KD_loss) / len(KD_loss) # orig code: average!
            
            
            ###### For rebuttal: other teacher integration strategy rather than averaging
            
            # Online 面向‑损失自适应（AdaKD/Inverse‑Loss）
            batch_losses = torch.stack(KD_loss) 
            
            # 指数滑动平均更新
            ema_loss = 0.9 * ema_loss + 0.1 * batch_losses.detach()
        
            inv = 1.0 / (ema_loss + 1e-6)     # 损失越低、inv 越大
            weights = inv / inv.sum()         # 归一化
        
            loss = (weights * batch_losses).sum()
            
            
            
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










def my_validate_withCLIP_v2(nouns_embedding, nouns_embedding_pretrained, args, val_loader, model, teacher, model_CLIP, device):
    #print('here0')
    print('=> validating...')
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')
    
    text_embedding_train_allCls_gt = get_CLIP_text_embeddings_GT(args, model_CLIP, device)
    
    #text_embedding_train_allCls_wordnet = get_CLIP_text_embeddings_WordNet(args, model_CLIP, device, train_loader) # we treat it as the input argument nouns_embedding !
    """
    if 'WNtreeV1' in args.useWhatModal:
        text_embedding_train_allCls_wordnet = get_CLIP_text_embeddings_WNtreeV1(args, model_CLIP, device, train_loader)
    if 'WNtreeV2' in args.useWhatModal:
        text_embedding_train_allCls_WNtreeV2 = get_CLIP_text_embeddings_WNtreeV2(args, model_CLIP, device, train_loader)
    """
    
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
    nouns_embedding.requires_grad = False
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            end = time.time()
            
            for batch_idx, this_batch_ in enumerate(val_loader):
                imgs_weak, labels = this_batch_[0], this_batch_[2]
                
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
                """
                if 'ImageNet' not in args.dataset:
                    image_embedding_val_1batch = get_CLIP_image_embeddings(args, model_CLIP, inputs, device)
                else:
                    # just load the pretrained image embeddings!:
                    image_embedding_val_1batch = this_batch_[3].numpy()
                """
                #outputs = model(inputs) # orig code
                
                assert('image_text' in args.useWhatModal)
                
                # newly modified by Chenqi: for CLIP-KD,
                # we are using either text or the image embeddings
                if args.useWhatModal == 'image_textGT' or \
                    ('image_text' in args.useWhatModal and my_flag == 'gt'):
                    text_clip_embeddings = text_embedding_train_allCls_gt[labels, :]
                    #text_clip_embeddings_noise = text_embedding_train_allCls_gt[labels_noise, :]
                    #text_clip_embeddings_clean = text_embedding_train_allCls_gt[labels_clean, :]
                    image_clip_embeddings = image_embedding_val_1batch
                    clip_embeddings = np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                    clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32).to(device)
                    
                    hierarchical_loss = 0
                    cosine_regularization_loss = 0
                    
                elif args.useWhatModal == 'image_textWordNet' or \
                    ('image_text' in args.useWhatModal and my_flag == 'wordnet'): #and 'WNtreeV2' not in args.useWhatModal)
                    #  or  args.useWhatModal == 'image_textWNtreeV1':
                    text_clip_retrieval_embedding = retrieve_text_v2(args, image_embedding_val_1batch, nouns_embedding)
                    text_clip_embeddings = text_clip_retrieval_embedding.to(torch.float32).to(device)
                    
                    image_clip_embeddings = image_embedding_val_1batch
                    image_clip_embeddings = torch.from_numpy(image_clip_embeddings).to(torch.float32).to(device)
                    
                    clip_embeddings = torch.cat((image_clip_embeddings, text_clip_embeddings), dim=1) #np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                    
                    # Calculate the hierarchical loss for the current batch:
                    # (Note that gt_embeddings and retrieval_embedding are both normalized)
                    gt_embeddings = text_embedding_train_allCls_gt[labels, :]
                    gt_embeddings = torch.from_numpy(gt_embeddings).to(torch.float32).to(device)
                    
                    hierarchical_loss = get_hierarchical_loss(gt_embeddings, text_clip_embeddings)
                    
                    cosine_regularization_loss = (1 - F.cosine_similarity(nouns_embedding, nouns_embedding_pretrained)).mean()
                    # we can also use mse loss for this constraint?!:
                    #F.mse_loss(nouns_embedding, nouns_embedding_pretrained)
                    
                """
                elif ('image_text' in args.useWhatModal and my_flag == 'wordnet' and 'WNtreeV2' in args.useWhatModal) or \
                    args.useWhatModal == 'image_textWNtreeV2':
                    text_clip_embeddings = text_embedding_train_allCls_WNtreeV2[labels, :]
                    image_clip_embeddings = image_embedding_val_1batch
                    clip_embeddings = np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                """
                
                
                # for debug:
                #print('clip_embeddings.shape = ' + str(clip_embeddings.shape)) # (batch_size, 512)
                #assert(False)
                
                #clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32).to(device)
                clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                
                labels = labels.to(device)
                
                outputs = model(clip_embeddings)
                  
                criterion = nn.CrossEntropyLoss()
                
                # Standard classification loss:
                classification_loss = criterion(outputs, labels)
                """
                # Regularization terms:
                lambda_hierarchical = 0.1  # Weight for hierarchical loss (you can tune this)
                lambda_cosine_reg = 0.01  # Regularization strength (you can tune this)
                """
                
                # Now include this regularization loss in the total loss
                loss = classification_loss + \
                    lambda_hierarchical * hierarchical_loss + \
                    lambda_cosine_reg * cosine_regularization_loss
                
                #print('loss = ' + str(loss))
                
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

                #outputs = model(inputs) # orig code
                _, outputs = model(inputs, return_logits=True)
                
                
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
                         projector,
                dataloaders,
                optimizer, 
                nouns_embedding, nouns_embedding_pretrained,
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
    
    # for the CLIP-KD!:
    model_CLIP = CLIPModel(model_name="ViT-B/32").to(device)
    model_CLIP.eval()
    
    """
    #### already move this part to the main file!:
    # newly added by Chenqi: for V2 we updating the CLIP nouns embedding:
    if args.mode == 'train':
        nouns_embedding = get_CLIP_text_embeddings_WordNet(args, model_CLIP, device, dataloaders['train']) # text_embedding_train_allCls_wordnet
        nouns_embedding = torch.from_numpy(nouns_embedding).cuda().half()
        #nouns_embedding.requires_grad = True  # Set the noun embeddings as trainable: so that this is to be optimized!
        
        #nouns_embedding_pretrained = get_CLIP_text_embeddings_WordNet(args, model_CLIP, device, dataloaders['train']) # frozen and just act as the anchor for the constraint
        #nouns_embedding_pretrained = torch.from_numpy(nouns_embedding_pretrained).cuda().half()
        nouns_embedding_pretrained = nouns_embedding.clone().detach()  # Keep a copy of the original pretrained embeddings
    """
    
    
    # ----for rebuttal!: 训练循环外：初始化
    weights = torch.tensor([1/3, 1/3, 1/3], device=device) # for strategy (1)
    
    for epoch in range(num_epochs):
        if epoch < start_epoch:
            continue  
        
        scheduler.step()
        
        #if epoch >= args.lr_steps[0] - 1:
        #    stop_calculate_information = True 

        if len(teacher) > 0 :
            # orig ver:
            acc1_train, loss_train = my_train_withCLIP_v2(dataloaders['train'], projector, model, model_CLIP, teacher, key_strings, optimizer, scheduler, epoch, device, temp, alpha, args)
            
            """
            ### (1) teacher integration strategy1: 规则‑驱动：按历次验证表现自适应
            acc1_train, loss_train = my_train_withCLIP_v3(dataloaders['train'], weights, model, model_CLIP, teacher, key_strings, optimizer, scheduler, epoch, device, temp, alpha, args)
            val_losses_for_weights = my_train_withCLIP_v3_helper(dataloaders['val'], model, model_CLIP, teacher, key_strings, optimizer, scheduler, epoch, device, temp, alpha, args)
            # 使用“表现越好→权重越大”规则（这里示例用 1/val_loss）
            inv = 1.0 / (val_losses_for_weights + 1e-8)
            weights = inv / inv.sum()          # 归一化为概率
            """
            
            """
            ### (2) teacher integration strategy2: 
            acc1_train, loss_train = my_train_withCLIP_v4(dataloaders['train'], model, model_CLIP, teacher, key_strings, optimizer, scheduler, epoch, device, temp, alpha, args)
            """
            
            
            acc1_val, loss_val = my_validate(dataloaders['val'], model, teacher, device)
            
            #acc1_val_aug, loss_val_aug, results_dict_val = my_validate_aug(dataloaders['val_aug'], model, teacher, device, args, stop_calculate_information)
        else:
            # below results_dicts are dummy dicts!
            if args.useWhatModal != 'rawImg':
                acc1_train, loss_train = train_teacher_withCLIP_v2(nouns_embedding, nouns_embedding_pretrained, dataloaders['train'], model, model_CLIP, optimizer, scheduler, epoch, device, temp, alpha, args)
                acc1_val, loss_val, = my_validate_withCLIP_v2(nouns_embedding, nouns_embedding_pretrained, args, dataloaders['val'], model, teacher, model_CLIP, device)
            else:
                acc1_train, loss_train = train_teacher(dataloaders['train'], model, optimizer, scheduler, epoch, device, temp, alpha, args)
                acc1_val, loss_val, = my_validate(dataloaders['val'], model, teacher, device)
            
            # acc1_val_aug, loss_val_aug, results_dict_val = my_validate_aug(dataloaders['val_aug'], model, teacher, device, args)
        
        tensorboard_writer.add_scalar("train/acc",acc1_train,epoch)
        tensorboard_writer.add_scalar("train/loss",loss_train,epoch) 
        tensorboard_writer.add_scalar("val/acc",acc1_val,epoch)
        tensorboard_writer.add_scalar("val/loss",loss_val,epoch)

        #if len(teacher) > 0 :
        #    tensorboard_writer.add_scalar("val/acc_val_aug",acc1_val_aug,epoch)
        #    tensorboard_writer.add_scalar("val/loss_val_aug",loss_val_aug,epoch)

        print_str = [f'{epoch+1}/{num_epochs} Train Acc@1: {acc1_train}, Val Acc@1: {acc1_val}']
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
            
            # newly added by Chenqi: save the nouns embedding only for train mode!
            if args.mode == 'train' and args.useWhatModal != 'rawImg':
                best_nouns_embedding = nouns_embedding.detach().cpu()  # Save the embeddings with the best validation accuracy
                torch.save(best_nouns_embedding, args.result+'/nouns_embeddings_bestAcc1.pt')
                

        
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
        
        # newly added by Chenqi: save the nouns embedding only for train mode!
        if args.mode == 'train' and args.useWhatModal != 'rawImg':
            this_nouns_embedding = nouns_embedding.detach().cpu()  # Save the embeddings with the best validation accuracy
            torch.save(this_nouns_embedding, args.result+'/nouns_embeddings_each-epoch.pt')
        
        
        
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
    




