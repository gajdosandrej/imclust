#!/usr/bin/env -S python3 -u # -*- python -*-
# imclust.py (c) R.Jaksa 2021 
# imclust_kmedoids.py - extended version of imclust.py (by A. Gajdos)   

import sys,os
import time

start_time = time.time()

# -------------------------- parse command-line arguments: dirname and no. of clusters

HELP = f"""
NAME
    imclust_kmedoids.py - image clustering demo

USAGE
    imclust_kmedoids.py [OPTIONS] DIRECTORY...

DESCRIPTION
    Image clusteuring demo imclust_kmedoids_v1.py will cluster images in
    the DIRECTORY, and produce a html visualization of results.

OPTIONS
    -h  help;  
    -m  models of NN to provide a numerical representations of images;  
    accepted inputs: see documentation https://www.tensorflow.org/api_docs/python/tf/keras/applications - section 'functions';  
    -c  requested number of clusters;  
    -pca number of Principal components;  
    -dn dataset name; 
    path to images. 

VERSION
    imclust.py 0.1 (c) R.Jaksa 2021
    imclust_kmedoids_v1.py - extended version of imclust.py (by A. Gajdos) 
"""

import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-h","--help",action="store_true")
parser.add_argument("-c","--clusters",type=str,default="10")
parser.add_argument("-m","--models",type=str,default="ResNet50")
parser.add_argument("-dn","--dataset_name",type=str,default="unnamed_dataset") 
parser.add_argument("-pca","--comp",type=int,default=256) 
parser.add_argument("path",type=str,nargs='*') 
args = parser.parse_args()

if args.help or len(args.path)==0:
    print(HELP)
    exit(0)

# ---------------------------------------------------------- get image names from dirs 
print(f"====================================")
print(f"=Loading names of images from dirs.=")
print(f"====================================")
print(f"...")

from glob import glob
import random

path = []
for dir in args.path:
  path += glob(dir+"/**/*.png",recursive=True)
  path += glob(dir+"/**/*.jpg",recursive=True)
random.shuffle(path)
# if size of subsample of cached images is less than pca_comp 
path_pca = path.copy() 

import numpy as np 
# random.seed(24)
# sample_size = 100     
# path = list(np.random.choice(path,size=sample_size,replace=False)) 

print(f"=========================")
print(f"=Names of images loaded.=")
print(f"=========================")

# ------------------------------------------------------------------------- load model 
print(f"====================")
print(f"=Loading NN models.=")
print(f"====================") 
print(f"...")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

models = args.models 
models = models.split(",") 
models_names = []  
models_list = [] 
prepr_list = [] 

# if args.model == 'DenseNet121':
if 'DenseNet121' in models: 
    # preproc: pixels scaled to 0..1 and each channel is normalized according ImageNet 
    prepr = tf.keras.applications.densenet.preprocess_input 
    model = tf.keras.applications.densenet.DenseNet121(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    # models_dict.update({'DenseNet121': model})
    models_names.append('DenseNet121') 
    prepr_list.append(prepr)
    models_list.append(model)
if 'DenseNet169' in models: 
    prepr = tf.keras.applications.densenet.preprocess_input
    model = tf.keras.applications.densenet.DenseNet169(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('DenseNet169') 
    prepr_list.append(prepr)
    models_list.append(model)
if 'DenseNet201' in models: 
    prepr = tf.keras.applications.densenet.preprocess_input 
    model = tf.keras.applications.densenet.DenseNet201(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('DenseNet201') 
    prepr_list.append(prepr)
    models_list.append(model)
if 'EfficientNetB0' in models: 
    prepr = tf.keras.applications.efficientnet.preprocess_input
    model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('EfficientNetB0') 
    prepr_list.append(prepr)
    models_list.append(model)
if 'EfficientNetB1' in models: 
    prepr = tf.keras.applications.efficientnet.preprocess_input 
    model = tf.keras.applications.efficientnet.EfficientNetB1(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('EfficientNetB1') 
    prepr_list.append(prepr)
    models_list.append(model)
if 'EfficientNetB2' in models: 
    prepr = tf.keras.applications.efficientnet.preprocess_input 
    model = tf.keras.applications.efficientnet.EfficientNetB2(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('EfficientNetB2') 
    prepr_list.append(prepr)
    models_list.append(model)
if 'EfficientNetB3' in models: 
    prepr = tf.keras.applications.efficientnet.preprocess_input 
    model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('EfficientNetB3') 
    prepr_list.append(prepr)
    models_list.append(model)
if 'EfficientNetB4' in models: 
    prepr = tf.keras.applications.efficientnet.preprocess_input 
    model = tf.keras.applications.efficientnet.EfficientNetB4(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('EfficientNetB4') 
    prepr_list.append(prepr)
    models_list.append(model)
if 'EfficientNetB5' in models: 
    prepr = tf.keras.applications.efficientnet.preprocess_input 
    model = tf.keras.applications.efficientnet.EfficientNetB5(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('EfficientNetB5') 
    prepr_list.append(prepr)
    models_list.append(model)
if 'EfficientNetB6' in models: 
    # preproc: void, no preprocessing 
    prepr = tf.keras.applications.efficientnet.preprocess_input 
    model = tf.keras.applications.efficientnet.EfficientNetB6(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('EfficientNetB6') 
    prepr_list.append(prepr)
    models_list.append(model)
if 'EfficientNetB7' in models: 
    prepr = tf.keras.applications.efficientnet.preprocess_input 
    model = tf.keras.applications.efficientnet.EfficientNetB7(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('EfficientNetB7') 
    prepr_list.append(prepr)
    models_list.append(model)
if 'InceptionResNetV2' in models: 
    prepr = tf.keras.applications.inception_resnet_v2.preprocess_input 
    model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('InceptionResNetV2') 
    prepr_list.append(prepr) 
    models_list.append(model)
if 'InceptionV3' in models: 
    # preproc: pixels scaled to -1..1 sample-wise 
    prepr = tf.keras.applications.inception_v3.preprocess_input 
    model = tf.keras.applications.inception_v3.InceptionV3(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('InceptionV3') 
    prepr_list.append(prepr) 
    models_list.append(model)
if 'MobileNet' in models: 
    prepr = tf.keras.applications.mobilenet.preprocess_input 
    model = tf.keras.applications.mobilenet.MobileNet(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('MobileNet') 
    prepr_list.append(prepr)
    models_list.append(model)
if 'MobileNetV2' in models: 
    prepr = tf.keras.applications.mobilenet_v2.preprocess_input 
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('MobileNetV2') 
    prepr_list.append(prepr) 
    models_list.append(model)
if 'MobileNetV3Large' in models: 
    prepr = tf.keras.applications.mobilenet_v3.preprocess_input 
    model = tf.keras.applications.MobileNetV3Large(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('MobileNetV3Large') 
    prepr_list.append(prepr) 
    models_list.append(model)
if 'MobileNetV3Small' in models: 
    prepr = tf.keras.applications.mobilenet_v3.preprocess_input 
    model = tf.keras.applications.MobileNetV3Small(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('MobileNetV3Small') 
    prepr_list.append(prepr)
    models_list.append(model)
# elif args.model == 'NASNetLarge': 
    # model = tf.keras.applications.nasnet.NASNetLarge(include_top=False,weights="imagenet",input_shape=(331,331,3))
if 'NASNetMobile' in models: 
    prepr = tf.keras.applications.nasnet.preprocess_input 
    model = tf.keras.applications.nasnet.NASNetMobile(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('NASNetMobile') 
    prepr_list.append(prepr) 
    models_list.append(model)
if 'ResNet101' in models: 
    prepr = tf.keras.applications.resnet50.preprocess_input 
    model = tf.keras.applications.ResNet101(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('ResNet101') 
    prepr_list.append(prepr)
    models_list.append(model)
if 'ResNet101V2' in models: 
    prepr = tf.keras.applications.resnet_v2.preprocess_input 
    model = tf.keras.applications.ResNet101V2(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('ResNet101V2') 
    prepr_list.append(prepr)
    models_list.append(model)
if 'ResNet152' in models: 
    prepr = tf.keras.applications.resnet50.preprocess_input 
    model = tf.keras.applications.ResNet152(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('ResNet152') 
    prepr_list.append(prepr)
    models_list.append(model)
if 'ResNet152V2' in models: 
    # preproc: pixels scaled to -1..1 sample-wise 
    prepr = tf.keras.applications.resnet_v2.preprocess_input 
    model = tf.keras.applications.ResNet152V2(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('ResNet152V2') 
    prepr_list.append(prepr) 
    models_list.append(model)
if 'ResNet50' in models: 
    # preproc: from RGB to BGR, each channel zero-centered according ImageNet, no scaling 
    prepr = tf.keras.applications.resnet50.preprocess_input
    model = tf.keras.applications.resnet50.ResNet50(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('ResNet50')
    prepr_list.append(prepr)
    models_list.append(model)
if 'ResNet50V2' in models: 
    prepr = tf.keras.applications.resnet_v2.preprocess_input 
    model = tf.keras.applications.ResNet50V2(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('ResNet50V2') 
    prepr_list.append(prepr)
    models_list.append(model)
if 'VGG16' in models: 
    # converted from RGB to BGR, each channel zero-centered according ImageNet, no scaling 
    prepr = tf.keras.applications.vgg16.preprocess_input 
    model = tf.keras.applications.VGG16(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('VGG16') 
    prepr_list.append(prepr)
    models_list.append(model)
if 'VGG19' in models: 
    prepr = tf.keras.applications.vgg19.preprocess_input
    model = tf.keras.applications.VGG19(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('VGG19') 
    prepr_list.append(prepr)
    models_list.append(model)
if 'Xception' in models: 
    # The inputs pixel values are scaled between -1 and 1, sample-wise. 
    prepr = tf.keras.applications.xception.preprocess_input 
    model = tf.keras.applications.xception.Xception(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('Xception') 
    prepr_list.append(prepr)
    models_list.append(model)

print(f"===================")
print(f"=NN models loaded.=")
print(f"===================")

# ------------------------------------------------------------------------ load images 
print(f"=======================================")
print(f"=Loading images and embedding vectors.=")
print(f"=======================================")
print(f"...")

from imageio import imread
from skimage.transform import resize
from sklearn.decomposition import PCA 
np.warnings.filterwarnings("ignore",category=np.VisibleDeprecationWarning) 

images_names = [os.path.basename(path[k]) for k in range(len(path))] 
n_imgs = len(images_names) 

# numpy save float32 raw array
def saveraw(name,data):
  arr = np.array(data,"float32")
  file = open(name,"wb")
  arr.tofile(file)
  file.close()

dataset_name = args.dataset_name 
cached_dataset_path = os.path.join('cache', dataset_name)
if not os.path.exists(f'cache'): 
    os.makedirs(f'cache')  
if not os.path.exists(cached_dataset_path):  
    os.makedirs(cached_dataset_path) 

pca_comp = args.comp 

# if pca_comp>n_imgs:
if pca_comp>len(path_pca): 
    print(f"Error: Number of images smaller than number of Principal components.") 
    exit()
SIZE = (224,224,3) 
pca_list = [None] * len(models) 
vectors = [np.empty([0,pca_comp],dtype=np.float32)]*len(models)
images = np.empty([0,224,224,3],dtype=np.float32)  
paths_html = [] 
for i in range(len(models)): 
    paths_html.append([])

# no cached images exist (for particular data set) 
if len(os.listdir(cached_dataset_path)) == 0: 
    
    for i in range(len(models)): 
        paths_html[i] = path.copy() 
    
    i = 0 
    
    while i < len(path): 
        i2 = i + pca_comp 
        imgs = np.array([imread(str(p)).astype(np.float32) for p in path[i:i2]])
        imgs = np.asarray([resize(image,SIZE,0) for image in imgs])
        images = np.concatenate((images, imgs),0)  

# ------------------------------------------------------------- get embedding vectors 
        for j in range(len(models)): 
            vector = models_list[j].predict(prepr_list[j](imgs))
            vector = vector.reshape(vector.shape[0],-1) 
            if i == 0: 
                pca = PCA(n_components=pca_comp)
                pca.fit(vector)  
                pca_list[j] = pca 
            vector = pca_list[j].transform(vector) 
            for k in range(0,len(vector)): 
                saveraw(os.path.join(cached_dataset_path, models_names[j]+'_pca_256_'+images_names[k+i]+'.npy'), vector[k]) 
            vectors[j] = np.concatenate((vectors[j], vector),0)
        
        i += pca_comp 

else:  

    for i in range(len(models)): 
        path1 = path.copy() 
        images_names1 = images_names.copy()
        model_cached_images_paths = [os.path.join(cached_dataset_path, filename) for filename in os.listdir(cached_dataset_path) if filename.startswith(models_names[i])]
        model_cached_images_names = [os.path.splitext(os.path.basename(p))[0].split("_")[3] for p in model_cached_images_paths]
        
        # cached images for model i are loaded 
        for j in range(len(model_cached_images_names)): 
            if model_cached_images_names[j] in images_names1: 
                vectors[i] = np.concatenate((vectors[i], [np.fromfile(model_cached_images_paths[j],dtype="float32")]),0) 
                idx_del = images_names1.index(model_cached_images_names[j])
                images_names1.remove(model_cached_images_names[j]) 
                paths_html[i].append(path1[idx_del])
                del path1[idx_del]
        
        # special case: fewer (non-cached) images than pca_comp are about to load and cache 
        # tuto cast je mozne skopirovat aj do predoslej vetvy a upravit pre pripad, ked nie su cachovane obrazky ale plati, ze vzorka obrazkov je mensia ako pca_comp 
        if 0<len(path1) and len(path1)<pca_comp: 
            imgs = np.array([imread(str(p)).astype(np.float32) for p in path_pca[0:pca_comp]])
            imgs = np.asarray([resize(image,SIZE,0) for image in imgs]) 
            vector = models_list[i].predict(prepr_list[i](imgs))
            vector = vector.reshape(vector.shape[0],-1)
            pca = PCA(n_components=pca_comp)
            pca.fit(vector) 
            pca_list[i] = pca 
            
            imgs = np.array([imread(str(p)).astype(np.float32) for p in path1[k:k2]])
            imgs = np.asarray([resize(image,SIZE,0) for image in imgs])
            images = np.concatenate((images, imgs),0) 
# ------------------------------------------------------------- get embedding vectors
            vector = models_list[i].predict(prepr_list[i](imgs))
            vector = vector.reshape(vector.shape[0],-1) 
            vector = pca_list[i].transform(vector) 
            
            for j in range(0,len(vector)): 
                saveraw(os.path.join(cached_dataset_path, models_names[i]+'_pca_256_'+images_names1[j+k]+'.npy'), vector[j]) 
            vectors[i] = np.concatenate((vectors[i], vector),0)
            
            paths_html[i] = path1.copy()
        
        # more (non-cached) images than pca_comp are about to load and cache       
        else: 
            
            k = 0 
            while k < len(path1): 
                k2 = k + pca_comp  
                imgs = np.array([imread(str(p)).astype(np.float32) for p in path1[k:k2]])
                imgs = np.asarray([resize(image,SIZE,0) for image in imgs])
                images = np.concatenate((images, imgs),0)  
# ------------------------------------------------------------- get embedding vectors 
                vector = models_list[i].predict(prepr_list[i](imgs))
                vector = vector.reshape(vector.shape[0],-1) 
                if k == 0: 
                    pca = PCA(n_components=pca_comp)
                    pca.fit(vector) 
                    pca_list[i] = pca 
                vector = pca_list[i].transform(vector) 
                for j in range(0,len(vector)): 
                    saveraw(os.path.join(cached_dataset_path, models_names[i]+'_pca_256_'+images_names1[j+k]+'.npy'), vector[j]) 
                vectors[i] = np.concatenate((vectors[i], vector),0)
                
                for kk in range(len(path1[k:k2])): 
                    paths_html[i].append(path1[kk])
                
                k += pca_comp 

print(f"======================================")
print(f"=Images and embedding vectors loaded.=")
print(f"======================================")

# ----------------------------------------------------------------------- cluster them
print(f"=======================")
print(f"=K-medoids clustering.=")
print(f"=======================") 
print(f"...")

from sklearn_extra.cluster import KMedoids

CLUSTERS = args.clusters 
CLUSTERS = CLUSTERS.split(",")
clusterings = []
 
for i in range(len(models)): 
    clusterings.append([])
    for j in range(len(CLUSTERS)): 
        clustering = KMedoids(n_clusters=int(float(CLUSTERS[j])))
        clustering.fit(vectors[i])
        cl = clustering.predict(vectors[i])
        clusterings[i].append(cl)
        # print(f"clusters: {cl}")

print(f"==============================")
print(f"=K-medoids clustering - DONE.=")
print(f"==============================")

# ------------------------------------------------ copy images according their cluster

# import shutil
# for i in range(len(images)):
#   if not os.path.exists(f"output/cluster{cluster[i]}"): os.makedirs(f"output/cluster{cluster[i]}")
#   print(f"cp {path[i]} output/cluster{cluster[i]}")
#   shutil.copy2(f"{path[i]}",f"output/cluster{cluster[i]}")

print(f"================================")
print(f"=Calculating indices (metrics).=")
print(f"================================") 
print(f"...")

# -------------------------------------------------------------------------- mean silhouette coefficient (plot + file)
from sklearn.metrics import silhouette_score
import pandas as pd 
import matplotlib.pyplot as plt 

if not os.path.exists(f'clustering_outputs'): 
    os.makedirs(f'clustering_outputs') 
output_dataset_path = os.path.join('clustering_outputs', dataset_name) 
if not os.path.exists(output_dataset_path):  
    os.makedirs(output_dataset_path)

MSC = []
for i in range(len(models)): 
    MSC.append([])
    for j in range(len(CLUSTERS)): 
        MSC[i].append(silhouette_score(vectors[i],clusterings[i][j]))
    
    frame = pd.DataFrame({'Clusters':CLUSTERS, 'MSC':MSC[i]})
    plt.figure(figsize=(12,6))
    plt.plot(frame['Clusters'], frame['MSC'], marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('MSC')
    plt.title('KMedoids: Mean Silhouette Coefficient (MSC) - ' + models_names[i])
    plt.savefig(output_dataset_path + '\MSC_' + models_names[i] + '_kmedoids.png')

    frame.to_csv(output_dataset_path + '\MSC_' + models_names[i] + '_kmedoids.txt', index=None, sep='\t', mode='a')

# -------------------------------------------------------------------------- Calinski-Harabasz index (plot + file)
from sklearn.metrics import calinski_harabasz_score 

CHS = []
for i in range(len(models)): 
    CHS.append([])
    for j in range(len(CLUSTERS)): 
        CHS[i].append(calinski_harabasz_score(vectors[i],clusterings[i][j]))
    
    frame = pd.DataFrame({'Clusters':CLUSTERS, 'CHS':CHS[i]})
    plt.figure(figsize=(12,6))
    plt.plot(frame['Clusters'], frame['CHS'], marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('CHS')
    plt.title('KMedoids: Calinski-Harabasz Score (CHS) - ' + models_names[i])
    plt.savefig(output_dataset_path + '\CHS_' + models_names[i] + '_kmedoids.png')

    frame.to_csv(output_dataset_path + '\CHS_' + models_names[i] + '_kmedoids.txt', index=None, sep='\t', mode='a')

# -------------------------------------------------------------------------- Davies-Bouldin index (plot + file)
from sklearn.metrics import davies_bouldin_score 

DBS = []
for i in range(len(models)): 
    DBS.append([])
    for j in range(len(CLUSTERS)): 
        DBS[i].append(davies_bouldin_score(vectors[i],clusterings[i][j]))
    
    frame = pd.DataFrame({'Clusters':CLUSTERS, 'DBS':DBS[i]})
    plt.figure(figsize=(12,6))
    plt.plot(frame['Clusters'], frame['DBS'], marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('DBS')
    plt.title('KMedoids: Davies-Bouldin Score (DBS) - ' + models_names[i])
    plt.savefig(output_dataset_path + '\DBS_' + models_names[i] + '_kmedoids.png')

    frame.to_csv(output_dataset_path + '\DBS_' + models_names[i] + '_kmedoids.txt', index=None, sep='\t', mode='a')

# -------------------------------------------------------------------------- The COP index (plot + file) 
from sklearn.metrics import pairwise_distances 
from validclust import cop

COP = [] 
for i in range(len(models)): 
    dist = pairwise_distances(vectors[i]) 
    COP.append([])
    for j in range(len(CLUSTERS)): 
        COP[i].append(cop(vectors[i], dist, clusterings[i][j]))
    
    frame = pd.DataFrame({'Clusters':CLUSTERS, 'COP':COP[i]})
    plt.figure(figsize=(12,6))
    plt.plot(frame['Clusters'], frame['COP'], marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('COP')
    plt.title('KMedoids: The COP index - ' + models_names[i])
    plt.savefig(output_dataset_path + '\COP_' + models_names[i] + '_kmedoids.png')

    frame.to_csv(output_dataset_path + '\COP_' + models_names[i] + '_kmedoids.txt', index=None, sep='\t', mode='a')

# -------------------------------------------------------------------------- The SDbw index (plot + file)
from s_dbw import S_Dbw

SDbw = [] 
for i in range(len(models)): 
    SDbw.append([])
    for j in range(len(CLUSTERS)): 
        SDbw[i].append(S_Dbw(vectors[i], clusterings[i][j], centers_id=None, method='Tong', alg_noise='bind', centr='mean', nearest_centr=True, metric='euclidean'))
    
    frame = pd.DataFrame({'Clusters':CLUSTERS, 'SDbw':SDbw[i]})
    plt.figure(figsize=(12,6))
    plt.plot(frame['Clusters'], frame['SDbw'], marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SDbw')
    plt.title('KMedoids: The SDbw index - ' + models_names[i])
    plt.savefig(output_dataset_path + '\SDbw_' + models_names[i] + '_kmedoids.png')

    frame.to_csv(output_dataset_path + '\SDbw_' + models_names[i] + '_kmedoids.txt', index=None, sep='\t', mode='a')

# -------------------------------------------------------------------------- The TSP (plot + file)
# from python_tsp.exact import solve_tsp_dynamic_programming 

# TSP = [] 
# for i in range(len(models)): 
    # TSP.append([])
    # for j in range(len(CLUSTERS)): 
        # for k in range(int(float(CLUSTERS[j]))): 
            # vectors2 = []
            # tsp_temp = []
            # for l in range(len(vectors[i])):
                # if clusterings[i][j][l] == k:
                    # # vectors2 = np.concatenate((vectors2, vectors[i][l])) 
                    # vectors2.append(vectors[i][l])
            # dist = pairwise_distances(vectors2) 
            # permutation, distance = solve_tsp_dynamic_programming(dist)
            # tsp_temp.append(distance)
        # TSP[i].append(np.mean(tsp_temp))
    
    # frame = pd.DataFrame({'Cluster':CLUSTERS, 'TSP':TSP[i]})
    # plt.figure(figsize=(12,6))
    # plt.plot(frame['Cluster'], frame['TSP'], marker='o')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('TSP')
    # plt.title('KMedoids: The TSP - ' + models_names[i])
    # plt.savefig('TSP_' + models_names[i] + '_kmedoids.png')

    # frame.to_csv(r'TSP_' + models_names[i] + '_kmedoids.txt', index=None, sep='\t', mode='a')

print(f"====================================================")
print(f"=Indices (metrics) calculated and written to files.=")
print(f"====================================================")

# -------------------------------------------------------------------------- make html 
print(f"===================================")
print(f"=Creating html page with clusters.=")
print(f"===================================")
print(f"...")

from web import *

for i in range(len(models)):
    for j in range(len(CLUSTERS)):
        # make html section for every cluster
        section = [""]*int(float(CLUSTERS[j]))
        for k in range(n_imgs):
            section[clusterings[i][j][k]] += addimg(f"{paths_html[i][k]}",f"cluster {clusterings[i][j][k]}",f"{paths_html[i][k]}")

        # build the page
        Nazov = f"<h1>algorithm: K-medoids, model: " + models_names[i] + ", number of clusters:" + str(CLUSTERS[j]) + "<h1>\n"
        BODY = ""
        for k in range(len(section)):
            BODY += f"<h2>cluster {k}<h2>\n"
            BODY += section[k]
            BODY += "\n\n"
        html = HTML.format(Nazov=Nazov,BODY=BODY,CSS=CSS)

        # save html
        # print("write: index_"+ models_names[i] +"_kmedoids"+str(CLUSTERS[j])+".html")
        with open(output_dataset_path + "\index_" + models_names[i] + "_kmedoids"+str(CLUSTERS[j])+".html","w") as fd:
            fd.write(html)

print(f"==================================")
print(f"=Html page with clusters created.=")
print(f"==================================")

print(f"=============================")
print(f"=Code executed successfully.=")
print(f"=============================")

print("--- %s seconds ---" % (time.time() - start_time)) 

# ------------------------------------------------------------------------------------
