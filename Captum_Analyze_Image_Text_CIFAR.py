#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 20:46:10 2025

@author: ps
"""

### This is for CIFAR dataset



import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from data_loader.DataLoaderCIFAR import Load_CIFAR100
import clip
from torchvision.models import resnet50
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import random
import csv
import faiss
import pandas as pd
import torch.nn.functional as F


# 配置路径
"""
MODEL_PATH = "runs/CIFAR100_train_get_Ts_image_textGT/run-1-epoch61/checkpoint_bestAcc1.pth.tar"
OUTPUT_DIR = "output_Captum/CIFAR100_train_get_Ts_image_textGT/"  # 输出文件夹
"""

"""
MODEL_PATH = "/home/ps/scratch/0315/runs/CIFAR100_train_get_Ts_image_textGT_20noise/run-1-epoch61/checkpoint_bestAcc1.pth.tar" #"runs/CIFAR100_train_get_Ts_image_textGT_20noise/run-1-epoch61/checkpoint_bestAcc1.pth.tar"
OUTPUT_DIR = "output_Captum/CIFAR100_train_get_Ts_image_textGT_20noise/"  # 输出文件夹
"""
"""
MODEL_PATH = "/home/ps/scratch/0315/runs/CIFAR100_train_get_Ts_image_textGT_50noise/run-1-epoch61/checkpoint_bestAcc1.pth.tar"
OUTPUT_DIR = "output_Captum/CIFAR100_train_get_Ts_image_textGT_50noise/"  # 输出文件夹
"""
"""
MODEL_PATH = "/home/ps/scratch/0315/runs/CIFAR100_train_get_Ts_image_textGT_80noise/run-1-epoch61/checkpoint_bestAcc1.pth.tar" #"/root/autodl-tmp/CLIP_KD/runs/CIFAR100_train_get_Tw_image_textGT/run-2-epoch61/checkpoint_bestAcc1.pth.tar"
OUTPUT_DIR = "output_Captum/CIFAR100_train_get_Ts_image_textGT_80noise/"  # 输出文件夹
"""
"""
MODEL_PATH = "/home/ps/scratch/0315/runs/CIFAR100_train_get_Ts_image_textGT_100noise/run-1-epoch61/checkpoint_bestAcc1.pth.tar"
OUTPUT_DIR = "output_Captum/CIFAR100_train_get_Ts_image_textGT_100noise/"  # 输出文件夹
"""



"""
MODEL_PATH = "runs/CIFAR100_train_get_Ts_image_text_20gt80wordnet/run-1-epoch61/checkpoint_bestAcc1.pth.tar"
OUTPUT_DIR = "output_Captum/CIFAR100_train_get_Ts_image_text_20gt80wordnet/"  # 输出文件夹
"""
"""
MODEL_PATH = "runs/CIFAR100_train_get_Ts_image_text_10gt90wordnet/run-1-epoch61/checkpoint_bestAcc1.pth.tar"
OUTPUT_DIR = "output_Captum/CIFAR100_train_get_Ts_image_text_10gt90wordnet/"  # 输出文件夹
"""
"""
MODEL_PATH = "runs/CIFAR100_train_get_Ts_image_text_1gt99wordnet/run-1-epoch61/checkpoint_bestAcc1.pth.tar"
OUTPUT_DIR = "output_Captum/CIFAR100_train_get_Ts_image_text_1gt99wordnet/"  # 输出文件夹
"""
"""
MODEL_PATH = "runs/CIFAR100_train_get_Ts_image_textWordNet/run-3-epoch400/checkpoint_bestAcc1.pth.tar"
OUTPUT_DIR = "output_Captum/CIFAR100_train_get_Ts_image_textWordNet/"  # 输出文件夹
"""
#"""
MODEL_PATH = "runs/CIFAR100_train_get_Ts_image_text_50gt50wordnet/run-1-epoch61/checkpoint_bestAcc1.pth.tar"
OUTPUT_DIR = "output_Captum/CIFAR100_train_get_Ts_image_text_50gt50wordnet/"  # 输出文件夹
#"""
"""
MODEL_PATH = "runs/CIFAR100_train_get_Ts_image_text_80gt20wordnet/run-1-epoch61/checkpoint_bestAcc1.pth.tar"
OUTPUT_DIR = "output_Captum/CIFAR100_train_get_Ts_image_text_80gt20wordnet/"  # 输出文件夹
"""




flag = 'wn' #noise wn
gt_percent = 0.5


CIFAR_META_PATH = "data/CIFAR100/infoMeta_dict.pkl" #"/root/autodl-tmp/CLIP_KD/data/cifar-100-python/meta"
DATA_ROOT = "/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img" #"/root/autodl-tmp/CLIP_KD/data/cifar-100-python"
OUTPUT_CSV = "sampled_results.csv"  # CSV 文件名
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



SIMPLE_IMAGENET_TEMPLATES = (
    lambda c: f"itap of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"art of the {c}.",
    lambda c: f"a photo of the small {c}.",
)

# 加载 CIFAR-100 类别名

def load_cifar100_classes(meta_path):
    with open(meta_path, 'rb') as f:
        infoMeta_dict = pickle.load(f)
    meta_dict = infoMeta_dict['meta_dict']
    fine_label_names =  meta_dict[b'fine_label_names']
    
    nouns = []
    for item_byte in fine_label_names:
        item_string = item_byte.decode('utf-8')
        nouns.append(item_string)
        
    return nouns




# 加载训练的模型
def load_model(model_path, num_classes=100, in_channels=1024):
    model = resnet50(weights=None)
    model.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model

# 分析图片和文字的贡献百分比
def analyze_contribution_captum(model, model_CLIP, image, text_embedding, target):
    with torch.no_grad():
        #image_embedding = clip_model.encode_image(image).to(torch.float32).to(DEVICE)
        images_embedding = get_CLIP_image_embeddings_all(model_CLIP, _, image, DEVICE)
        image_embedding = torch.from_numpy(images_embedding).cuda().half()
        
        text_embedding = text_embedding.to(torch.float32).to(DEVICE)
        """
        text_embedding = clip_model.encode_text(text).to(torch.float32).to(DEVICE)
        text_embedding = text_embedding.mean(dim=0, keepdim=True)
        if flag == 'wn':
            text_embedding = torch.cat((text_embedding, text_embedding_reg), dim=0)
        """
        print('text_embedding.shape = ' + str(text_embedding.shape)) # torch.Size([128, 512])
        #assert(False)

    combined_features = torch.cat((image_embedding, text_embedding), dim=1)
    print('combined_features.shape = ' + str(combined_features.shape)) # torch.Size([128, 1024])
    #assert(False)
    target = target.to(DEVICE)
    
    def forward_func(features):
        return model(features.unsqueeze(-1).unsqueeze(-1))

    ig = IntegratedGradients(forward_func)
    attributions = ig.attribute(combined_features, target=target, n_steps=50)

    image_contribution = attributions[:, :512].abs().sum()
    text_contribution = attributions[:, 512:].abs().sum()

    total_contribution = image_contribution + text_contribution
    image_contribution_ratio = (image_contribution / total_contribution).item()
    text_contribution_ratio = (text_contribution / total_contribution).item()

    return image_contribution_ratio, text_contribution_ratio



# 抽取样本
def sample_test_data(test_loader, sample_size=100):
    samples = list(test_loader)
    random.seed(42)  # 固定随机种子以确保可重复性
    sampled_data = random.sample(samples, sample_size)
    return sampled_data

# 创建输出文件夹
def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

# 保存结果到 CSV 文件
def save_results_to_csv(results, output_dir, csv_name):
    csv_path = os.path.join(output_dir, csv_name)
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Sample Index", "Class Name", "Image Contribution (%)", "Text Contribution (%)"])
        writer.writerows(results)
    print(f"Results saved to {csv_path}")

# 可视化贡献百分比并保存
def visualize_and_save_results(avg_image_ratio, avg_text_ratio, save_path):
    labels = ["Image Contribution", "Text Contribution"]
    sizes = [avg_image_ratio * 100, avg_text_ratio * 100]  # 转为百分比

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=["skyblue", "lightgreen"])
    plt.title("Contribution Ratios")
    plt.axis('equal')  # 保证饼图为圆形
    plt.savefig(save_path)
    print(f"Contribution ratios saved to {save_path}")
    plt.show()


def get_CLIP_image_embeddings_all(model_CLIP, labels_reg, images_reg, DEVICE):
    print('---- getting CLIP image embeddings for the whole training set...')
    
    features_list = []
    #labels_list = []
    
    inputs = images_reg.to(DEVICE)
    
    print('inputs.shape = ' + str(inputs.shape))
    #assert(False)
    
    with torch.no_grad():
        this_feature = model_CLIP.encode_image(inputs)
    features_list.append(this_feature.cpu().numpy())
    
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

def get_prompt_WordNet(words, index, device="cuda"):
    prompt = [SIMPLE_IMAGENET_TEMPLATES[index](word) for word in words]
    text = clip.tokenize(prompt, truncate=True).to(device)
    return text

# 生成文本模板
#def get_prompt(words, templates, device="cuda"):
#    prompts = [templates[i](word.replace("_", " ")) for i in range(len(templates)) for word in words]
#    text = clip.tokenize(prompts, truncate=True).to(device)
#    return text
def get_prompt_GT(words, index, device="cuda"):
    prompt = [SIMPLE_IMAGENET_TEMPLATES[index](word.replace("_"," ")) for word in words]
    #print("prompt = " + str(prompt))
    text = clip.tokenize(prompt, truncate=True).to(device)
    return text


def get_text_embedding_wn(model_CLIP, labels_reg, images_reg, DEVICE):
    
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
    images_embedding = get_CLIP_image_embeddings_all(model_CLIP, labels_reg, images_reg, DEVICE)
    
    nouns_embedding = torch.from_numpy(nouns_embedding).cuda().half()
    nouns_num = nouns_embedding.shape[0]
    
    images_embedding = torch.from_numpy(images_embedding).cuda().half()
    image_num = images_embedding.shape[0]
    
    
    cluster_num = 20 #50 #20 #10 #300 #150
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
    #print('TAC_class_names.shape = ' + str(TAC_class_names.shape))
    #print('************** the selected WordNet nouns are:')
    #print(TAC_class_names.tolist())
    #print('************** END **************')

    #np.save(
    #    './data/' + args.dataset + "/WordNet_filtered_nouns_embedding.npy", #'./data/' + args.dataset.split('_')[0] + "/WordNet_filtered_nouns_embedding.npy",
    #    nouns_embedding_selected.cpu().numpy(),
    #)
    
    nouns_embedding_selected = nouns_embedding_selected.cpu().numpy()
    nouns_embedding_selected = nouns_embedding_selected / np.linalg.norm(
                        nouns_embedding_selected, axis=1, keepdims=True
                        )
    
    #TAC_class_names_list = TAC_class_names.tolist()
    
    tau = 0.005
    nouns_embedding = torch.from_numpy(nouns_embedding_selected).cuda().half() 
    #nouns_embedding = torch.from_numpy(nouns_embedding).cuda().half()
    #nouns_num = nouns_embedding.shape[0]
    #images_embedding = torch.from_numpy(images_embedding).cuda().half()
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
    
    return retrieval_embedding


def get_text_embedding_gt(model_CLIP, words_gt):
    nouns_num = len(words_gt)
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
            nouns_batch = words_gt[start:end]
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
    #np.save('./data/' + args.dataset + "/GTnouns_embedding_ensemble.npy", embeddings)
    
    embeddings = embeddings / np.linalg.norm(
                        embeddings, axis=1, keepdims=True
                        )
    
    return embeddings



if __name__ == "__main__":
    # 创建输出文件夹
    create_output_dir(OUTPUT_DIR)

    # 加载 CLIP 模型
    #clip_model, _ = clip.load("ViT-B/32", device=DEVICE)

    # 加载 CIFAR-100 类别
    CIFAR100_CLASSES = load_cifar100_classes(CIFAR_META_PATH)

    # 加载训练模型
    trained_model = load_model(MODEL_PATH)

    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    # 加载 CIFAR100 测试数据
    dataset = "CIFAR100"
    phase = "val"
    batch_size = 128
    test_loader = Load_CIFAR100(DATA_ROOT, dataset, phase, batch_size, shuffle=True)

    # 抽取 100 个样本
    #sampled_data = sample_test_data(test_loader, sample_size=100)

    # 计算贡献比例
    total_image_ratio = 0
    total_text_ratio = 0
    count = 0
    results = []

    # 添加 tqdm 进度条
    for idx, (images, _, labels) in enumerate(test_loader):
        
        # modified by Chenqi:
        sample_num_gt = int(len(labels) * gt_percent)
        sample_num_reg = len(labels) - sample_num_gt
        #idx_gt = idx[:sample_num_gt]
        #idx_reg = idx[sample_num_gt:]
        labels_gt = labels[:sample_num_gt]
        labels_reg = labels[sample_num_gt:]
        
        images_gt = images[:sample_num_gt,:,:,:]
        images_reg = images[sample_num_gt:,:,:,:]
        
        """
        print('len(labels) = ' + str(len(labels)))
        print('labels.shape = ' + str(labels.shape))
        print('sample_num_gt = ' + str(sample_num_gt))
        print('sample_num_reg = ' + str(sample_num_reg))
        #print('idx_gt = ' + str(idx_gt))
        #print('idx_reg = ' + str(idx_reg))
        print('labels_gt = ' + str(labels_gt))
        print('labels_reg = ' + str(labels_reg))
        print('images_reg.shape = ' + str(images_reg.shape))
        print('images.shape = ' + str(images.shape))
        assert(False)
        """
        
        
        images = images.to(DEVICE)
        #images_reg = images_reg.to(DEVICE)
        #words = [CIFAR100_CLASSES[label] for label in labels]
        
        from TACmodels import CLIPModel
        model_CLIP = CLIPModel(model_name="ViT-B/32").to(DEVICE)
        model_CLIP.eval()
        
        # modified by Chenqi:
        words_gt = [CIFAR100_CLASSES[label] for label in labels_gt]
        #text_gt = get_prompt(words_gt, SIMPLE_IMAGENET_TEMPLATES)
        text_embedding_gt = get_text_embedding_gt(model_CLIP, words_gt)
        #print('text_embedding_gt.shape = ' + str(text_embedding_gt.shape)) # (25, 512)
        
        if flag == 'noise':
            random.shuffle(labels_reg)
            words_reg = [CIFAR100_CLASSES[label] for label in labels_reg]
            #text_reg = get_prompt(words_reg, SIMPLE_IMAGENET_TEMPLATES)
            #print('text_reg.shape = ' + str(text_reg.shape)) # torch.Size([721, 77])
            text_embedding_reg = get_text_embedding_gt(model_CLIP, words_reg)
            #print('text_embedding_reg.shape = ' + str(text_embedding_reg.shape)) # (103, 512)
            #assert(False)
        elif flag == 'wn':
            #words_orig = [CIFAR100_CLASSES[label] for label in labels_reg]
            text_embedding_reg = get_text_embedding_wn(model_CLIP, labels_reg, images_reg, DEVICE)
            #print('text_embedding_reg.shape = ' + str(text_embedding_reg.shape)) # (103, 512)
            #assert(False)
        
        text_embedding_gt = torch.from_numpy(text_embedding_gt)
        if flag == 'noise':
            text_embedding_reg = torch.from_numpy(text_embedding_reg)
        text_embedding = torch.cat((text_embedding_gt, text_embedding_reg), dim=0)
        print('text_embedding.shape = ' + str(text_embedding.shape)) # torch.Size([128, 512])
        #assert(False)
        
        """
        if flag == 'noise':
            text = torch.cat((text_gt, text_reg), dim=0)
            print('text.shape = ' + str(text.shape))
            assert(False)
            text_embedding_reg = None
        elif flag == 'wn':
            text = text_gt
        """
        
        image_ratio, text_ratio = analyze_contribution_captum(
            trained_model, model_CLIP, images, text_embedding, target=labels #.item()
        )

        total_image_ratio += image_ratio
        total_text_ratio += text_ratio
        count += 1

        # 记录每个样本的结果
        #results.append([idx + 1, CIFAR100_CLASSES[labels.item()], image_ratio * 100, text_ratio * 100])
        
        break
        
    # 平均贡献百分比
    avg_image_ratio = total_image_ratio / count
    avg_text_ratio = total_text_ratio / count

    print(f"Average Image Contribution Ratio (Sampled): {avg_image_ratio:.2f}")
    print(f"Average Text Contribution Ratio (Sampled): {avg_text_ratio:.2f}")

    # 保存每个样本的贡献结果到 CSV
    #save_results_to_csv(results, OUTPUT_DIR, OUTPUT_CSV)

    # 保存可视化结果到输出文件夹
    save_path = os.path.join(OUTPUT_DIR, "sampled_contribution_ratios.png")
    visualize_and_save_results(avg_image_ratio, avg_text_ratio, save_path)




