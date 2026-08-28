#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 09:34:31 2025

@author: ps
"""

# code for my rebuttal:
# get the teacher logits entropy, and KL div between one-hot and teacher logits.

# This is for ImageNet(-mini) dataset


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
from data_loader.DataLoaderImageNet import Load_ImageNet_emb, Load_ImageNet, Load_ImageNet_noised
from data_loader.DataLoaderScene import Load_scene
from data_loader.DataLoaderUTKFace import Load_UTKFace

from TACmodels import CLIPModel
import numpy as np
import pickle
import faiss
import clip
import pandas as pd
import torch.nn.functional as F

from scipy.stats import entropy




data_root = {
        'CIFAR100':  '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img',
        'ImageNet':'/home/ps/scratch/KD_imbalance/LFME/my_data/ILSVRC/Data/CLS-LOC',
        'scene': '/home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/scene/cleaned',
        'UTKFace': '/home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/UTKFace/cls_by_race',
        'ImageNet-mini': '/home/ps/scratch/CLIP_KD/data/ImageNet-mini',
        }
teacher_path = {
    
    ### for CIFAR100:
    ##"weak" : "runs/CIFAR100_train_get_Tw/run-1/checkpoint_bestAcc1.pth.tar",
    ##"strong" : "runs/CIFAR100_train_get_Ts/run-1/checkpoint_bestAcc1.pth.tar",
    
    #"image_textWordNet": "runs/CIFAR100_train_get_Ts_image_textWordNet/run-1-epoch61/checkpoint_bestAcc1.pth.tar", #"runs/CIFAR100_train_get_Tw_image_textWordNet/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    #"image_text_20gt80wordnet": "runs/CIFAR100_train_get_Ts_image_text_20gt80wordnet/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    #"image_text_80gt20wordnet": "runs/CIFAR100_train_get_Ts_image_text_80gt20wordnet/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    ##"image_text_50gt50wordnet": "runs/CIFAR100_train_get_Ts_image_text_50gt50wordnet/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    
    #"image_text_20gt80noise": "runs/CIFAR100_train_get_Ts_image_textGT_80noise/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    #"image_text_80gt20noise": "runs/CIFAR100_train_get_Ts_image_textGT_20noise/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    #"image_text_0gt100noise": "runs/CIFAR100_train_get_Ts_image_textGT_100noise/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    
    #"image_textGT": "runs/CIFAR100_train_get_Ts_image_textGT/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    #"image_textLLM": "runs/CIFAR100_train_get_Ts_image_textLLM_V3/run-4-epoch61/checkpoint_bestAcc1.pth.tar",
    
    ### for CIFAR100_imb100:
    #"image_textGT": "runs/CIFAR100_imb100_train_get_Ts_image_textGT/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    #"image_textWordNet": "runs/CIFAR100_imb100_train_get_Ts_image_textWordNet_V2/run-1-epoch200/checkpoint_bestAcc1.pth.tar", #"runs/CIFAR100_imb100_train_get_Ts_image_textWordNet/run-6-epoch100/checkpoint_bestAcc1.pth.tar",
    #"image_textLLM": "runs/CIFAR100_imb100_train_get_Ts_image_textLLM_V3/run-1-epoch200/checkpoint_bestAcc1.pth.tar",
    
    
    ### for ImageNet-mini:
    ##"weak" : "runs/ImageNet-mini_train_get_Tw_image_ViT/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    ##"strong" : "runs/ImageNet-mini_train_get_Ts_image_ViT/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    #"image_textWordNet": "runs/ImageNet-mini_train_get_Ts_image_textWordNet_V2_MLP/run-2-epoch100/checkpoint_bestAcc1.pth.tar",
    
    #"image_text_20gt80wordnet": "runs/ImageNet-mini_train_get_Ts_image_text_20gt80wordnet_V2_MLP/run-1-epoch100/checkpoint_bestAcc1.pth.tar",
    #"image_text_50gt50wordnet": "runs/ImageNet-mini_train_get_Ts_image_text_50gt50wordnet_V2_MLP/run-1-epoch100/checkpoint_bestAcc1.pth.tar",
    #"image_text_80gt20wordnet": "runs/ImageNet-mini_train_get_Ts_image_text_80gt20wordnet_V2_MLP/run-1-epoch100/checkpoint_bestAcc1.pth.tar",
    
    #"image_textGT": "runs/ImageNet-mini_train_get_Ts_image_textGT_V2_MLP/run-1-epoch100/checkpoint_bestAcc1.pth.tar",
    
    #"image_text_80gt20noise": "runs/ImageNet-mini_train_get_Ts_image_textGT_20noise_V2_MLP/run-1-epoch100/checkpoint_bestAcc1.pth.tar",
    #"image_text_50gt50noise": "runs/ImageNet-mini_train_get_Ts_image_textGT_50noise_V2_MLP/run-1-epoch100/checkpoint_bestAcc1.pth.tar",
    #"image_text_20gt80noise": "runs/ImageNet-mini_train_get_Ts_image_textGT_80noise_V2_MLP/run-1-epoch100/checkpoint_bestAcc1.pth.tar",
    #"image_text_0gt100noise": "runs/ImageNet-mini_train_get_Ts_image_textGT_100noise_V2_MLP/run-1-epoch100/checkpoint_bestAcc1.pth.tar",
    
    ### for ImageNet_LT:
    #"image_textGT": "runs/ImageNet_LT_train_get_Ts_image_textGT/run-1-epoch60/checkpoint_bestAcc1.pth.tar",
    #"image_textWordNet": "runs/ImageNet_LT_train_get_Ts_image_textWordNet_V2/run-2-epoch60/checkpoint_bestAcc1.pth.tar", 
    #"image_textLLM": "runs/ImageNet_LT_train_get_Ts_image_textLLM_V3/run-1-epoch60/checkpoint_bestAcc1.pth.tar",
    
    ### for ImageNet:
    #"image_textLLM": "runs/ImageNet_train_get_Ts_image_textLLM_V3/run-1-epoch35/checkpoint_bestAcc1.pth.tar",
    #"image_textWordNet": "runs/ImageNet_train_get_Ts_image_textWordNet_V2/run-4-epoch35/checkpoint_bestAcc1.pth.tar", 
    #"image_textGT": "runs/ImageNet_train_get_Ts_image_textGT/run-1-epoch35/checkpoint_bestAcc1.pth.tar",
    
    ### for scene
    #"image_textGT": "runs/scene_train_get_Ts_image_textGT/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    #"image_textWordNet": "runs/scene_train_get_Ts_image_textWordNet_V2/run-1-epoch61/checkpoint_bestAcc1.pth.tar", 
    #"image_textLLM": "runs/scene_train_get_Ts_image_textLLM_V3/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    
    ### for UTKFace
    "image_textWordNet": "runs/UTKFace_train_get_Ts_image_textWordNet_V2/run-2-epoch61/checkpoint_bestAcc1.pth.tar", 
    "image_textGT": "runs/UTKFace_train_get_Ts_image_textGT/run-1-epoch61/checkpoint_bestAcc1.pth.tar",
    
    
    }


    
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

    
def get_CLIP_text_embeddings_GT(mydataset, model_CLIP, device):
    
    ### Option 2: use the gt class names!
    
    #"""
    result_dir = './data/' + mydataset + "/GTnouns_embedding_ensemble.npy"
    if os.path.exists(result_dir):
        embeddings = np.load(result_dir)
        embeddings = embeddings / np.linalg.norm(
                            embeddings, axis=1, keepdims=True
                            )
        return embeddings
    #"""
    
    pkl_file_path = './data/' + mydataset + '/infoMeta_dict.pkl'
    #print(pkl_file_path)
    if (os.path.exists(pkl_file_path)):
        with open(pkl_file_path, 'rb') as f:
            infoMeta_dict = pickle.load(f)
        meta_dict = infoMeta_dict['meta_dict']
        if "CIFAR100" in mydataset:
            fine_label_names = meta_dict[b'fine_label_names']
        elif "ImageNet" in mydataset and "mini" not in mydataset:
            fine_label_names = meta_dict['fine_label_names']
    elif mydataset == 'ImageNet-mini':
        fine_label_names = ['house finch, linnet, Carpodacus mexicanus', 'robin, American robin, Turdus migratorius', 'triceratops', 'green mamba', 'harvestman, daddy longlegs, Phalangium opilio', 'toucan', 'goose', 'jellyfish', 'nematode, nematode worm, roundworm', 'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica', 'dugong, Dugong dugon', 'Walker hound, Walker foxhound', 'Ibizan hound, Ibizan Podenco', 'Saluki, gazelle hound', 'golden retriever', 'Gordon setter', 'komondor', 'boxer', 'Tibetan mastiff', 'French bulldog', 'malamute, malemute, Alaskan malamute', 'dalmatian, coach dog, carriage dog', 'Newfoundland, Newfoundland dog', 'miniature poodle', 'white wolf, Arctic wolf, Canis lupus tundrarum', 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus', 'Arctic fox, white fox, Alopex lagopus', 'lion, king of beasts, Panthera leo', 'meerkat, mierkat', 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'rhinoceros beetle', 'ant, emmet, pismire', 'black-footed ferret, ferret, Mustela nigripes', 'three-toed sloth, ai, Bradypus tridactylus', 'rock beauty, Holocanthus tricolor', 'aircraft carrier, carrier, flattop, attack aircraft carrier', 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin', 'barrel, cask', 'beer bottle', 'bookshop, bookstore, bookstall', 'cannon', 'carousel, carrousel, merry-go-round, roundabout, whirligig', 'carton', 'catamaran', 'chime, bell, gong', 'clog, geta, patten, sabot', 'cocktail shaker', 'combination lock', 'crate', 'cuirass', 'dishrag, dishcloth', 'dome', 'electric guitar', 'file, file cabinet, filing cabinet', 'fire screen, fireguard', 'frying pan, frypan, skillet', 'garbage truck, dustcart', 'hair slide', 'holster', 'horizontal bar, high bar', 'hourglass', 'iPod', 'lipstick, lip rouge', 'miniskirt, mini', 'missile', 'mixing bowl', 'oboe, hautboy, hautbois', 'organ, pipe organ', 'parallel bars, bars', 'pencil box, pencil case', 'photocopier', 'poncho', 'prayer rug, prayer mat', 'reel', 'school bus', 'scoreboard', 'slot, one-armed bandit', 'snorkel', 'solar dish, solar collector, solar furnace', "spider web, spider's web", 'stage', 'tank, army tank, armored combat vehicle, armoured combat vehicle', 'theater curtain, theatre curtain', 'tile roof', 'tobacco shop, tobacconist shop, tobacconist', 'unicycle, monocycle', 'upright, upright piano', 'vase', 'wok', 'worm fence, snake fence, snake-rail fence, Virginia fence', 'yawl', 'street sign', 'consomme', 'trifle', 'hotdog, hot dog, red hot', 'orange', 'cliff, drop, drop-off', 'coral reef', 'bolete', 'ear, spike, capitulum']
    
    elif 'scene' in mydataset:
        fine_label_names = ['Buildings', 'Forests', 'Glacier', 'Mountains',
                            'Sea', 'Street']
    elif 'UTKFace' in mydataset:
        fine_label_names = ['White', 'Black', 'Asian', 'Indian',
                            'Others Hispanic Latino Middle Eastern']
    
    # for the CIFAR, we need to convert the byte class names into strings!:
    #nouns = fine_label_names
    nouns = []
    for item_byte in fine_label_names:
        if "CIFAR100" in mydataset:
            item_string = item_byte.decode('utf-8')
        else: #elif "ImageNet" in mydataset:
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
    np.save('./data/' + mydataset + "/GTnouns_embedding_ensemble.npy", embeddings)
    
    embeddings = embeddings / np.linalg.norm(
                        embeddings, axis=1, keepdims=True
                        )
    
    #print(nouns)
    #assert(False)
    
    return embeddings
    



def get_CLIP_text_embeddings_WordNet(mydataset, model_CLIP, device, train_loader):
    
    ### Option 1: use the WordNet nouns!
    
    #result_dir = './data/' + mydataset.split('_')[0] + "/WordNet_filtered_nouns_embedding.npy"
    result_dir = './data/' + mydataset + "/WordNet_filtered_nouns_embedding.npy"
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
    images_embedding = get_CLIP_image_embeddings_all(model_CLIP, train_loader, device)
    
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
        './data/' + mydataset + "/WordNet_filtered_nouns_embedding.npy", #'./data/' + mydataset.split('_')[0] + "/WordNet_filtered_nouns_embedding.npy",
        nouns_embedding_selected.cpu().numpy(),
    )
    
    nouns_embedding_selected = nouns_embedding_selected.cpu().numpy()
    nouns_embedding_selected = nouns_embedding_selected / np.linalg.norm(
                        nouns_embedding_selected, axis=1, keepdims=True
                        )
    
    return nouns_embedding_selected


def get_CLIP_text_embeddings_LLM(args, model_CLIP, device, train_loader):
    """
    LLM Stage B (reuse Algorithm1):
      candidates -> CLIP text embeddings (multi-template avg)
      -> use training images CLIP embeddings -> kmeans alignment -> select topK per cluster
      -> save npy
    """
    result_dir = './data_LLM/' + mydataset + "/LLM_filtered_nouns_embedding.npy"
    if os.path.exists(result_dir):
        nouns_embedding_selected = np.load(result_dir)
        nouns_embedding_selected = nouns_embedding_selected / np.linalg.norm(
                            nouns_embedding_selected, axis=1, keepdims=True
                            )
        return nouns_embedding_selected
    
    #assert(False)
    return None


    
def get_CLIP_image_embeddings_all(model_CLIP, data_loader, device):
    print('---- getting CLIP image embeddings for the whole training set...')
    
    features_list = []
    #labels_list = []
    
    #for batch_idx, (imgs_weak, imgs_strong, labels) in enumerate(data_loader):
    for batch_idx, this_batch in enumerate(data_loader):
        imgs_weak, imgs_strong = this_batch[0], this_batch[1]
        
        #labels = labels.to(device)
        #if args.train_add_strong:
        inputs = imgs_strong.to(device)
        #if args.train_add_weak:
        #    inputs = imgs_weak.to(device)
            
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


def get_CLIP_image_embeddings(model_CLIP, inputs, device):
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


def retrieve_text(images_embedding, nouns_embedding):
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



def stable_entropy_from_logits(logits: torch.Tensor) -> float:
    # logits: [B, C] float32 on GPU
    x = logits - logits.max(dim=1, keepdim=True).values  # stabilize
    logp = x - torch.logsumexp(x, dim=1, keepdim=True)   # log softmax
    p = torch.exp(logp)
    H = -(p * logp).sum(dim=1)  # natural log base
    return float(H.mean().item())



def KL_divergence(model1_logits, model2_logits):
    # 计算的是两个模型输出的概率分布之间的 KL 散度  
    probs2 = F.softmax(model2_logits, dim=1)
    log_probs1 = F.log_softmax(model1_logits, dim=1, dtype=torch.double)
    kl_div = F.kl_div(log_probs1, probs2, reduction='batchmean', log_target=False) * 10 # 太小了,放大一些
    return kl_div.item()


def get_one_hot_logits(labels: torch.Tensor,
                       num_classes: int,
                       logit_scale: float = 8.0) -> torch.Tensor:
    """
    Turn class indices -> one‑hot logits.
    Args
        labels       : (B,)  LongTensor with class indices.
        num_classes  : total number of classes.
        logit_scale  : large positive number; higher = peakier distribution.
    Returns
        (B, num_classes) FloatTensor on same device as `labels`.
    """
    # 0/1 one‑hot
    one_hot = F.one_hot(labels, num_classes).float()

    # convert to logits: +logit_scale for the true class, -logit_scale otherwise
    pos = logit_scale * one_hot
    neg = -logit_scale * (1.0 - one_hot)
    return pos + neg


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



if __name__ == '__main__':
    
    mydataset = 'UTKFace' #'scene' #'CIFAR100_imb100' #'CIFAR100' #'ImageNet' #'ImageNet_LT' #'ImageNet-mini'
    batch_size = 128
    noise_percent = 0 # only useful when prompt_type == 'noise'
    wn_percent = 100 # only useful when prompt_type == 'wn'
    prompt_type = 'wn' #'gt' #'noise' # 'wn' # 'llm'
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if 'ImageNet' in mydataset:
        print("Function: Load_ImageNet.")
        if prompt_type != 'noise': # for gt and wn
            dataloaders = {x: Load_ImageNet(data_root=data_root[mydataset.split("_")[0]], dataset=mydataset, phase=x,
                        batch_size=batch_size, num_workers=4,
                        shuffle=True if x == 'train' else False)
                for x in ['val']}  
            
        elif prompt_type == 'noise':
            dataloaders = {x: Load_ImageNet_noised(data_root=data_root[mydataset.split("_")[0]], dataset=mydataset, phase=x,
                         batch_size=batch_size, num_workers=4,
                         noise_percent=noise_percent,
                         shuffle=True if x == 'train' else False)
                    for x in ['val']}
    elif 'CIFAR100' in mydataset:
        print("Function: load_CIFAR100.")
        # newly modified by Chenqi:
        if prompt_type != 'noise': # for gt and wn
            dataloaders = {x: Load_CIFAR100(data_root=data_root[mydataset.split("_")[0]], dataset=mydataset, phase=x,
                         batch_size=batch_size, num_workers=4,
                         shuffle=True if x == 'train' else False)
                    for x in ['val']} 
        
        elif prompt_type == 'noise':
            dataloaders = {x: Load_CIFAR100_noised(data_root=data_root[mydataset.split("_")[0]], dataset=mydataset, phase=x,
                         batch_size=batch_size, num_workers=4,
                         noise_percent=noise_percent,
                         shuffle=True if x == 'train' else False)
                    for x in ['val']} 
    elif 'scene' in mydataset:
        print("Function: Load_scene.")
        dataloaders = {x: Load_scene(data_root=data_root[mydataset.split("_")[0]], dataset=mydataset, phase=x,
                    batch_size=batch_size, num_workers=4,
                    shuffle=True if x == 'train' else False)
            for x in ['train', 'val']}
    
    elif 'UTKFace' in mydataset:
        print("Function: Load_UTKFace.")
        dataloaders = {x: Load_UTKFace(data_root=data_root[mydataset.split("_")[0]], dataset=mydataset, phase=x,
                    batch_size=batch_size, num_workers=4,
                    shuffle=True if x == 'train' else False)
            for x in ['train', 'val']}
    
    
    if mydataset == "ImageNet-mini":
        class_num = 100
    elif mydataset == "ImageNet" or mydataset == "ImageNet_LT":
        class_num = 1000
    elif "CIFAR100" in mydataset:
        class_num = 100
    elif "scene" in mydataset:
        class_num = 6
    elif "UTKFace" in mydataset:
        class_num = 5
    
    #"""
    teacher_model = models.resnet50(pretrained=False)
    inCh = 1024
    teacher_model.conv1 = nn.Conv2d(in_channels=inCh, out_channels=64, kernel_size=7,
                               stride=2, padding=3, bias=False)
    
    num_ftrs = teacher_model.fc.in_features
    print("class_num--->",class_num)
    teacher_model.fc = nn.Linear(num_ftrs, class_num)
    #"""
    """
    teacher_model = MLPClassifier(num_classes=class_num)
    """
    teacher_model.to(device)
    
    
    if prompt_type == 'gt':
        Tmodel = teacher_path["image_textGT"]
    elif prompt_type == 'noise':
        if noise_percent==80:
            Tmodel = teacher_path['image_text_20gt80noise']
        if noise_percent==50:
            Tmodel = teacher_path['image_text_50gt50noise']
        if noise_percent==20:
            Tmodel = teacher_path['image_text_80gt20noise']
        if noise_percent==100:
            Tmodel = teacher_path['image_text_0gt100noise']
    elif prompt_type == 'wn':
        if wn_percent == 100:
            Tmodel = teacher_path["image_textWordNet"]
        elif wn_percent == 80:
            Tmodel = teacher_path["image_text_20gt80wordnet"]
        elif wn_percent == 50:
            Tmodel = teacher_path["image_text_50gt50wordnet"]
        elif wn_percent == 20:
            Tmodel = teacher_path["image_text_80gt20wordnet"]
    elif prompt_type == 'llm':
        Tmodel = teacher_path["image_textLLM"]
    
    print("The loaded teacher model is:",Tmodel)
    teacher_model.load_state_dict(torch.load(Tmodel)['state_dict'])
    teacher_model.eval()
    
    #print(teacher_model)
    #assert(False)
    
    model_CLIP = CLIPModel(model_name="ViT-B/32").to(device)
    model_CLIP.eval()
    
    val_loader = dataloaders['val']
    
    #if prompt_type == 'gt' or prompt_type == 'noise':
    text_embedding_train_allCls_gt = get_CLIP_text_embeddings_GT(mydataset, model_CLIP, device)
    #elif prompt_type == 'wn':
    text_embedding_train_allCls_wordnet = get_CLIP_text_embeddings_WordNet(mydataset, model_CLIP, device, val_loader)
    #elif prompt_type == 'llm':
    text_embedding_train_allCls_llm = get_CLIP_text_embeddings_LLM(mydataset, model_CLIP, device, val_loader)
    
    
    if prompt_type == 'wn':
        num_batches = len(val_loader)
        print(f"Total number of batches: {num_batches}")
        
        gt_percent = 100 - wn_percent
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
    
    
    Tx_entropy = 0
    Tx_klDiv = 0
    batch_count = 0
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            
            for batch_idx, (imgs_weak,_, labels) in enumerate(val_loader):
                inputs = imgs_weak.to(device)
                
                my_flag = -1
                if prompt_type == 'wn':
                    if batch_idx in gt_batches_idx_list:
                        my_flag = 'gt'
                    elif batch_idx in wordnet_batches_idx_list:
                        my_flag = 'wordnet'
                
                image_embedding_val_1batch = get_CLIP_image_embeddings(model_CLIP, inputs, device)
                
                if prompt_type == 'gt' or prompt_type == 'noise' or my_flag == 'gt':
                    text_clip_embeddings = text_embedding_train_allCls_gt[labels, :]
                    image_clip_embeddings = image_embedding_val_1batch
                    clip_embeddings = np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                elif prompt_type == 'wn' and my_flag == 'wordnet':
                    text_clip_embeddings = retrieve_text(image_embedding_val_1batch, text_embedding_train_allCls_wordnet)
                    image_clip_embeddings = image_embedding_val_1batch
                    clip_embeddings = np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                elif prompt_type == 'llm':
                    text_clip_embeddings = retrieve_text(image_embedding_val_1batch, text_embedding_train_allCls_llm)
                    image_clip_embeddings = image_embedding_val_1batch
                    clip_embeddings = np.concatenate((image_clip_embeddings, text_clip_embeddings), axis=1)
                
                clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32).to(device)
                clip_embeddings = clip_embeddings.unsqueeze(-1).unsqueeze(-1)
                clip_embeddings = clip_embeddings.float()
                
                if torch.isnan(clip_embeddings).any() or torch.isinf(clip_embeddings).any():
                    print(f"[BAD] batch={batch_idx} clip_embeddings has NaN/Inf")
                    continue
                
                labels = labels.to(device)
                outputs = teacher_model(clip_embeddings)
                
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"[BAD] batch={batch_idx} outputs has NaN/Inf")
                    print("outputs min/max:", outputs.min().item(), outputs.max().item())
                    print("clip_embeddings min/max:", clip_embeddings.min().item(), clip_embeddings.max().item())
                    continue
                
                oneHot = get_one_hot_logits(labels, num_classes=outputs.size(1), logit_scale=8.0).to(device)
                
                Tx_entropy += stable_entropy_from_logits(outputs) #get_entropy(outputs)
                
                Tx_klDiv += KL_divergence(outputs, oneHot)
                batch_count += 1
                
                print('@@@')
                print('Tx_entropy = ' + str(Tx_entropy))
                print('batch_count = ' + str(batch_count))
    
    # get the results:
    Tx_entropy = Tx_entropy / batch_count
    Tx_klDiv = Tx_klDiv / batch_count
    
    print('*****************************************')
    print('Avg. teacher entropy is: ' + str(Tx_entropy))
    print('KL(one‑hot‖T_x) is: ' + str(Tx_klDiv))
    
    
                
                
                
                
                
    
    
    
    
    
    



