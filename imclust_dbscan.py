#!/usr/bin/env -S python3 -u # -*- python -*-
# imclust.py (c) R.Jaksa 2021 
# imclust_dbscan.py - extended version of imclust.py (by A. Gajdos)   

import sys,os

# -------------------------- parse command-line arguments: dirname and no. of clusters

HELP = f"""
NAME
    imclust_dbscan.py - image clustering demo

USAGE
    imclust_dbscan.py [OPTIONS] DIRECTORY...

DESCRIPTION
    Image clusteuring demo imclust_dbscan.py will cluster images in
    the DIRECTORY, and produce a html visualization of results.

OPTIONS
    -h  This help. 
    -m  Models of NN to provide a numerical representations of images. 
    Accepted inputs: see documentation https://www.tensorflow.org/api_docs/python/tf/keras/applications - section 'functions'. 
    -e  The maximum distance between two samples for one to be considered as in the neighborhood of the other.

VERSION
    imclust.py 0.1 (c) R.Jaksa 2021
    imclust_dbscan.py - extended version of imclust.py (by A. Gajdos) 
"""

import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-h","--help",action="store_true")
parser.add_argument("-e","--eps",type=str,default="0.5")
parser.add_argument("-m","--models",type=str,default="ResNet50")
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
# models_dict = {} 
models_names = []  
models_list = []
# model = None  

# if args.model == 'DenseNet121':
if 'DenseNet121' in models: 
    model = tf.keras.applications.densenet.DenseNet121(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    # models_dict.update({'DenseNet121': model})
    models_names.append('DenseNet121')
    models_list.append(model)
# elif args.model == 'DenseNet169': 
if 'DenseNet169' in models: 
    model = tf.keras.applications.densenet.DenseNet169(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    # models_dict.update({'DenseNet169': model}) 
    models_names.append('DenseNet169')
    models_list.append(model)
if 'DenseNet201' in models: 
    model = tf.keras.applications.densenet.DenseNet201(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('DenseNet201')
    models_list.append(model)
if 'EfficientNetB0' in models: 
    model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('EfficientNetB0')
    models_list.append(model)
if 'EfficientNetB1' in models: 
    model = tf.keras.applications.efficientnet.EfficientNetB1(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('EfficientNetB1')
    models_list.append(model)
if 'EfficientNetB2' in models: 
    model = tf.keras.applications.efficientnet.EfficientNetB2(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('EfficientNetB2')
    models_list.append(model)
if 'EfficientNetB3' in models: 
    model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('EfficientNetB3')
    models_list.append(model)
if 'EfficientNetB4' in models: 
    model = tf.keras.applications.efficientnet.EfficientNetB4(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('EfficientNetB4')
    models_list.append(model)
if 'EfficientNetB5' in models: 
    model = tf.keras.applications.efficientnet.EfficientNetB5(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('EfficientNetB5')
    models_list.append(model)
if 'EfficientNetB6' in models: 
    model = tf.keras.applications.efficientnet.EfficientNetB6(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('EfficientNetB6')
    models_list.append(model)
if 'EfficientNetB7' in models: 
    model = tf.keras.applications.efficientnet.EfficientNetB7(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('EfficientNetB7')
    models_list.append(model)
if 'InceptionResNetV2' in models: 
    model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('InceptionResNetV2')
    models_list.append(model)
if 'InceptionV3' in models: 
    model = tf.keras.applications.inception_v3.InceptionV3(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('InceptionV3')
    models_list.append(model)
if 'MobileNet' in models: 
    model = tf.keras.applications.mobilenet.MobileNet(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('MobileNet')
    models_list.append(model)
if 'MobileNetV2' in models: 
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('MobileNetV2')
    models_list.append(model)
if 'MobileNetV3Large' in models: 
    model = tf.keras.applications.MobileNetV3Large(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('MobileNetV3Large')
    models_list.append(model)
if 'MobileNetV3Small' in models: 
    model = tf.keras.applications.MobileNetV3Small(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('MobileNetV3Small')
    models_list.append(model)
# elif args.model == 'NASNetLarge': 
    # model = tf.keras.applications.nasnet.NASNetLarge(include_top=False,weights="imagenet",input_shape=(331,331,3))
if 'NASNetMobile' in models: 
    model = tf.keras.applications.nasnet.NASNetMobile(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('NASNetMobile')
    models_list.append(model)
if 'ResNet101' in models: 
    model = tf.keras.applications.ResNet101(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('ResNet101')
    models_list.append(model)
if 'ResNet101V2' in models: 
    model = tf.keras.applications.ResNet101V2(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('ResNet101V2')
    models_list.append(model)
if 'ResNet152' in models: 
    model = tf.keras.applications.ResNet152(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('ResNet152')
    models_list.append(model)
if 'ResNet152V2' in models: 
    model = tf.keras.applications.ResNet152V2(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('ResNet152V2')
    models_list.append(model)
if 'ResNet50' in models: 
    model = tf.keras.applications.resnet50.ResNet50(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('ResNet50')
    models_list.append(model)
if 'ResNet50V2' in models: 
    model = tf.keras.applications.ResNet50V2(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('ResNet50V2')
    models_list.append(model)
if 'VGG16' in models: 
    model = tf.keras.applications.VGG16(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('VGG16')
    models_list.append(model)
if 'VGG19' in models: 
    model = tf.keras.applications.VGG19(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('VGG19')
    models_list.append(model)
if 'Xception' in models: 
    model = tf.keras.applications.xception.Xception(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    models_names.append('Xception')
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
import numpy as np 
from sklearn.decomposition import PCA
np.warnings.filterwarnings("ignore",category=np.VisibleDeprecationWarning)

SIZE = (224,224,3)
# pca = PCA(n_components=256)
pca_list = []
# vectors = np.empty([0,256],dtype=np.float32) 
vectors = [np.empty([0,256],dtype=np.float32)]*len(models)
images = np.empty([0,224,224,3],dtype=np.float32)
i=0
while i < len(path): 
    i2 = i + 256 
    # images = np.array([imread(str(p)).astype(np.float32) for p in path[i:i2]]) 
    imgs = np.array([imread(str(p)).astype(np.float32) for p in path[i:i2]])
    # print(f"images shape: {images.shape}")
    # images = np.asarray([resize(image,SIZE,0) for image in images]) 
    imgs = np.asarray([resize(image,SIZE,0) for image in imgs])
    # print(f"images: {len(images)}")
    # print(f"imgs length: {len(imgs)}")
    # print(f"single image shape: {images[0].shape}") 
    # print(f"imgs shape: {imgs.shape}") 
    images = np.concatenate((images, imgs),0)

# ------------------------------------------------------------- get embeddings vectors
    
    for j in range(len(models)): 
        # vector = models_dict[j].predict(imgs)
        vector = models_list[j].predict(imgs)
        # print(f"model output shape: {vector[0].shape}")
        vector = vector.reshape(vector.shape[0],-1)
        # print(f"reshaped to 1D: {vector[0].shape}") 
        if i == 0: 
            pca = PCA(n_components=256)
            pca.fit(vector) 
            pca_list.append(pca)
        vector = pca_list[j].transform(vector)
        # print(f"vector transformed by pca: {vector[0].shape}")
        vectors[j] = np.concatenate((vectors[j], vector),0)
        
    i += 256

print(f"======================================")
print(f"=Images and embedding vectors loaded.=")
print(f"======================================")

# ----------------------------------------------------------------------- cluster them
print(f"====================")
print(f"=DBSCAN clustering.=")
print(f"====================") 
print(f"...")

from sklearn.cluster import DBSCAN 

eps = args.eps
eps = eps.split(",")
clusterings = []
 
for i in range(len(models)): 
    clusterings.append([])
    for j in range(len(eps)): 
        clustering = DBSCAN(eps=float(eps[j]))
        # clustering.fit(vectors[i])
        # cl = clustering.predict(vectors[i])
        cl = clustering.fit_predict(vectors[i])
        clusterings[i].append(cl)
        # print(f"clusters: {cl}") 

print(f"===========================")
print(f"=DBSCAN clustering - DONE.=")
print(f"===========================")

# ------------------------------------------------ copy images according their cluster

# import shutil
# for i in range(len(images)):
#   if not os.path.exists(f"output/cluster{cluster[i]}"): os.makedirs(f"output/cluster{cluster[i]}")
#   print(f"cp {path[i]} output/cluster{cluster[i]}")
#   shutil.copy2(f"{path[i]}",f"output/cluster{cluster[i]}")

# -------------------------------------------------------------------------- excluding outliers 
vectors_wo = [] 
clusterings_wo = [] 
for i in range(len(models)): 
    vectors_wo.append([]) 
    clusterings_wo.append([])
    for j in range(len(eps)): 
        vectors_wo[i].append([]) 
        clusterings_wo[i].append([])
        for k in range(len(clusterings[i][j])): 
            if clusterings[i][j][k] != -1: 
                vectors_wo[i][j].append(vectors[i][k])
                clusterings_wo[i][j].append(clusterings[i][j][k])

print(f"================================")
print(f"=Calculating indices (metrics).=")
print(f"================================") 
print(f"...")

# -------------------------------------------------------------------------- mean silhouette coefficient (plot + file)
from sklearn.metrics import silhouette_score
import pandas as pd 
import matplotlib.pyplot as plt 

# models_names = list(slovnik.keys())

MSC = [] 
MSC_wo = []
for i in range(len(models)): 
    MSC.append([]) 
    MSC_wo.append([])
    for j in range(len(eps)): 
        MSC[i].append(silhouette_score(vectors[i],clusterings[i][j])) 
        MSC_wo[i].append(silhouette_score(vectors_wo[i][j],clusterings_wo[i][j]))
    
    frame = pd.DataFrame({'eps':eps, 'MSC':MSC[i]})
    plt.figure(figsize=(12,6))
    plt.plot(frame['eps'], frame['MSC'], marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('MSC')
    plt.title('DBSCAN: Mean Silhouette Coefficient (MSC) - ' + models_names[i])
    plt.savefig('MSC_' + models_names[i] + '_dbscan.png')

    frame.to_csv(r'MSC_' + models_names[i] + '_dbscan.txt', index=None, sep='\t', mode='a') 
    
    frame = pd.DataFrame({'eps':eps, 'MSC_wo':MSC_wo[i]})
    plt.figure(figsize=(12,6))
    plt.plot(frame['eps'], frame['MSC_wo'], marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('MSC_wo')
    plt.title('DBSCAN_wo: Mean Silhouette Coefficient (MSC) - ' + models_names[i])
    plt.savefig('MSC_wo_' + models_names[i] + '_dbscan.png')

    frame.to_csv(r'MSC_wo_' + models_names[i] + '_dbscan.txt', index=None, sep='\t', mode='a')
    
# MSC_wo = []  
# for i in range(len(models)): 
    # MSC_wo.append([])
    # for j in range(len(eps)): 
        # vectors_wo = []
        # clusterings_wo = [] 
        # for k in range(len(vectors[i])): 
            # if clusterings[i][j][k] != -1: 
                # vectors_wo.append(vectors[i][k])
                # clusterings_wo.append(clusterings[i][j][k])
        # MSC_wo[i].append(silhouette_score(vectors_wo,clusterings_wo))
    
    # frame = pd.DataFrame({'eps':eps, 'MSC_wo':MSC_wo[i]})
    # plt.figure(figsize=(12,6))
    # plt.plot(frame['eps'], frame['MSC_wo'], marker='o')
    # plt.xlabel('Epsilon')
    # plt.ylabel('MSC_wo')
    # plt.title('Mean Silhouette Coefficient (MSC) - ' + models_names[i])
    # plt.savefig('MSC_wo_' + models_names[i] + '_dbscan.png')

    # frame.to_csv(r'MSC_wo_' + models_names[i] + '_dbscan.txt', index=None, sep='\t', mode='a')

# -------------------------------------------------------------------------- Calinski-Harabasz index (plot + file)
from sklearn.metrics import calinski_harabasz_score 

CHS = [] 
CHS_wo = []
for i in range(len(models)): 
    CHS.append([]) 
    CHS_wo.append([])
    for j in range(len(eps)): 
        CHS[i].append(calinski_harabasz_score(vectors[i],clusterings[i][j]))
        CHS_wo[i].append(calinski_harabasz_score(vectors_wo[i][j],clusterings_wo[i][j]))
    
    frame = pd.DataFrame({'eps':eps, 'CHS':CHS[i]})
    plt.figure(figsize=(12,6))
    plt.plot(frame['eps'], frame['CHS'], marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('CHS')
    plt.title('DBSCAN: Calinski-Harabasz Score (CHS) - ' + models_names[i])
    plt.savefig('CHS_' + models_names[i] + '_dbscan.png')

    frame.to_csv(r'CHS_' + models_names[i] + '_dbscan.txt', index=None, sep='\t', mode='a') 
    
    frame = pd.DataFrame({'eps':eps, 'CHS_wo':CHS_wo[i]})
    plt.figure(figsize=(12,6))
    plt.plot(frame['eps'], frame['CHS_wo'], marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('CHS_wo')
    plt.title('DBSCAN_wo: Calinski-Harabasz Score (CHS) - ' + models_names[i])
    plt.savefig('CHS_wo_' + models_names[i] + '_dbscan.png')

    frame.to_csv(r'CHS_wo_' + models_names[i] + '_dbscan.txt', index=None, sep='\t', mode='a')
    
# -------------------------------------------------------------------------- Davies-Bouldin index (plot + file)
from sklearn.metrics import davies_bouldin_score 

DBS = [] 
DBS_wo = []
for i in range(len(models)): 
    DBS.append([])
    DBS_wo.append([])
    for j in range(len(eps)): 
        DBS[i].append(davies_bouldin_score(vectors[i],clusterings[i][j])) 
        DBS_wo[i].append(davies_bouldin_score(vectors_wo[i][j],clusterings_wo[i][j])) 
    
    frame = pd.DataFrame({'eps':eps, 'DBS':DBS[i]})
    plt.figure(figsize=(12,6))
    plt.plot(frame['eps'], frame['DBS'], marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('DBS')
    plt.title('DBSCAN: Davies-Bouldin Score (DBS) - ' + models_names[i])
    plt.savefig('DBS_' + models_names[i] + '_dbscan.png')

    frame.to_csv(r'DBS_' + models_names[i] + '_dbscan.txt', index=None, sep='\t', mode='a')
    
    frame = pd.DataFrame({'eps':eps, 'DBS_wo':DBS_wo[i]})
    plt.figure(figsize=(12,6))
    plt.plot(frame['eps'], frame['DBS_wo'], marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('DBS_wo')
    plt.title('DBSCAN_wo: Davies-Bouldin Score (DBS) - ' + models_names[i])
    plt.savefig('DBS_wo_' + models_names[i] + '_dbscan.png')

    frame.to_csv(r'DBS_wo_' + models_names[i] + '_dbscan.txt', index=None, sep='\t', mode='a')

# -------------------------------------------------------------------------- The COP index (plot + file) 
from sklearn.metrics import pairwise_distances 
from validclust import cop

COP = []
COP_wo = [] 
for i in range(len(models)): 
    dist = pairwise_distances(vectors[i]) 
    COP.append([])
    COP_wo.append([])
    for j in range(len(eps)): 
        dist_wo = pairwise_distances(vectors_wo[i][j])
        COP[i].append(cop(vectors[i], dist, clusterings[i][j])) 
        COP_wo[i].append(cop(np.array(vectors_wo[i][j]), dist_wo, np.array(clusterings_wo[i][j])))
    
    frame = pd.DataFrame({'eps':eps, 'COP':COP[i]})
    plt.figure(figsize=(12,6))
    plt.plot(frame['eps'], frame['COP'], marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('COP')
    plt.title('DBSCAN: The COP index - ' + models_names[i])
    plt.savefig('COP_' + models_names[i] + '_dbscan.png')

    frame.to_csv(r'COP_' + models_names[i] + '_dbscan.txt', index=None, sep='\t', mode='a')
    
    frame = pd.DataFrame({'eps':eps, 'COP_wo':COP_wo[i]})
    plt.figure(figsize=(12,6))
    plt.plot(frame['eps'], frame['COP_wo'], marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('COP_wo')
    plt.title('DBSCAN_wo: The COP index - ' + models_names[i])
    plt.savefig('COP_wo_' + models_names[i] + '_dbscan.png')

    frame.to_csv(r'COP_wo_' + models_names[i] + '_dbscan.txt', index=None, sep='\t', mode='a')

# -------------------------------------------------------------------------- The SDbw index (plot + file)
from s_dbw import S_Dbw

SDbw = []
SDbw_wo = [] 
for i in range(len(models)): 
    SDbw.append([])
    SDbw_wo.append([])
    for j in range(len(eps)): 
        SDbw[i].append(S_Dbw(vectors[i], clusterings[i][j], centers_id=None, method='Tong', alg_noise='bind', centr='mean', nearest_centr=True, metric='euclidean'))
        SDbw_wo[i].append(S_Dbw(np.array(vectors_wo[i][j]), np.array(clusterings_wo[i][j]), centers_id=None, method='Tong', alg_noise='bind', centr='mean', nearest_centr=True, metric='euclidean'))
    
    frame = pd.DataFrame({'eps':eps, 'SDbw':SDbw[i]})
    plt.figure(figsize=(12,6))
    plt.plot(frame['eps'], frame['SDbw'], marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('SDbw')
    plt.title('DBSCAN: The SDbw index - ' + models_names[i])
    plt.savefig('SDbw_' + models_names[i] + '_dbscan.png')

    frame.to_csv(r'SDbw_' + models_names[i] + '_dbscan.txt', index=None, sep='\t', mode='a') 
    
    frame = pd.DataFrame({'eps':eps, 'SDbw_wo':SDbw_wo[i]})
    plt.figure(figsize=(12,6))
    plt.plot(frame['eps'], frame['SDbw_wo'], marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('SDbw_wo')
    plt.title('DBSCAN%wo: The SDbw index - ' + models_names[i])
    plt.savefig('SDbw_wo_' + models_names[i] + '_dbscan.png')

    frame.to_csv(r'SDbw_wo_' + models_names[i] + '_dbscan.txt', index=None, sep='\t', mode='a')

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
    for j in range(len(eps)):
        # make html section for every cluster
        # section = [""]*int(float(eps[j]))
        section = [""]*len(np.unique(clusterings[i][j]))
        for k in range(len(images)):
            section[clusterings[i][j][k]] += addimg(f"{path[k]}",f"cluster {clusterings[i][j][k]}",f"{path[k]}")

        # build the page
        Nazov = f"<h1>algorithm: DBSCAN, model: " + models_names[i] + ", epsilon:" + str(eps[j]) + "<h1>\n"
        BODY = ""
        for k in range(len(section)):
            BODY += f"<h2>cluster {k}<h2>\n"
            BODY += section[k]
            BODY += "\n\n"
        html = HTML.format(Nazov=Nazov,BODY=BODY,CSS=CSS)

        # save html
        # print("write: index_"+ models_names[i] +"_dbscan"+str(eps[j]).replace(".","")+".html")
        with open("index_" + models_names[i] + "_dbscan"+str(eps[j]).replace(".","")+".html","w") as fd:
            fd.write(html)

print(f"==================================")
print(f"=Html page with clusters created.=")
print(f"==================================")

# ------------------------------------------------------------------------------------
