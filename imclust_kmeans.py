#!/usr/bin/env -S python3 -u # -*- python -*-
# imclust.py (c) R.Jaksa 2021 
# imclust_kmeans.py - modified version of imclust.py (by A. Gajdos)   

import sys,os

# -------------------------- parse command-line arguments: dirname and no. of clusters

HELP = f"""
NAME
    imclust.py - image clustering demo

USAGE
    imclust.py [OPTIONS] DIRECTORY...

DESCRIPTION
    Image clusteuring demo imclust.py will cluster images in
    the DIRECTORY, and produce a html visualization of results.

OPTIONS
    -h  This help. 
    -m  Model of NN to provide a numerical representation of images. 
    Accepted inputs: see documentation https://www.tensorflow.org/api_docs/python/tf/keras/applications - section 'functions'. 
    -c  Requested number of clusters.

VERSION
    imclust.py 0.1 (c) R.Jaksa 2021
    imclust_kmeans.py - modified version of imclust.py (by A. Gajdos) 
"""

import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-h","--help",action="store_true")
# parser.add_argument("-c","--clusters",type=int,default=10)
parser.add_argument("-c","--clusters",type=str,default="10")
parser.add_argument("-m","--model",type=str,default="ResNet50")
parser.add_argument("path",type=str,nargs='*') 
args = parser.parse_args()

if args.help or len(args.path)==0:
    print(HELP)
    exit(0)

# ---------------------------------------------------------- get image names from dirs
from glob import glob
import random

path = []
for dir in args.path:
  path += glob(dir+"/**/*.png",recursive=True)
  path += glob(dir+"/**/*.jpg",recursive=True)
random.shuffle(path)
# for p in path: print(p)

# ------------------------------------------------------------------------- load model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

model = None  

if args.model == 'DenseNet121': 
    model = tf.keras.applications.densenet.DenseNet121(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
elif args.model == 'DenseNet169': 
    model = tf.keras.applications.densenet.DenseNet169(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
elif args.model == 'DenseNet201': 
    model = tf.keras.applications.densenet.DenseNet201(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'EfficientNetB0': 
    model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'EfficientNetB1': 
    model = tf.keras.applications.efficientnet.EfficientNetB1(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'EfficientNetB2': 
    model = tf.keras.applications.efficientnet.EfficientNetB2(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'EfficientNetB3': 
    model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'EfficientNetB4': 
    model = tf.keras.applications.efficientnet.EfficientNetB4(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'EfficientNetB5': 
    model = tf.keras.applications.efficientnet.EfficientNetB5(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'EfficientNetB6': 
    model = tf.keras.applications.efficientnet.EfficientNetB6(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'EfficientNetB7': 
    model = tf.keras.applications.efficientnet.EfficientNetB7(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'InceptionResNetV2': 
    model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'InceptionV3': 
    model = tf.keras.applications.inception_v3.InceptionV3(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'MobileNet': 
    model = tf.keras.applications.mobilenet.MobileNet(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'MobileNetV2': 
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'MobileNetV3Large': 
    model = tf.keras.applications.MobileNetV3Large(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'MobileNetV3Small': 
    model = tf.keras.applications.MobileNetV3Small(include_top=False,weights="imagenet",input_shape=(224,224,3))
# elif args.model == 'NASNetLarge': 
    # model = tf.keras.applications.nasnet.NASNetLarge(include_top=False,weights="imagenet",input_shape=(331,331,3))
elif args.model == 'NASNetMobile': 
    model = tf.keras.applications.nasnet.NASNetMobile(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'ResNet101': 
    model = tf.keras.applications.ResNet101(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'ResNet101V2': 
    model = tf.keras.applications.ResNet101V2(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'ResNet152': 
    model = tf.keras.applications.ResNet152(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'ResNet152V2': 
    model = tf.keras.applications.ResNet152V2(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'ResNet50': 
    model = tf.keras.applications.resnet50.ResNet50(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'ResNet50V2': 
    model = tf.keras.applications.ResNet50V2(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'VGG16': 
    model = tf.keras.applications.VGG16(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'VGG19': 
    model = tf.keras.applications.VGG19(include_top=False,weights="imagenet",input_shape=(224,224,3))
elif args.model == 'Xception': 
    model = tf.keras.applications.xception.Xception(include_top=False,weights="imagenet",input_shape=(224,224,3))

# ------------------------------------------------------------------------ load images
from imageio import imread
from skimage.transform import resize
import numpy as np 
from sklearn.decomposition import PCA
np.warnings.filterwarnings("ignore",category=np.VisibleDeprecationWarning)

SIZE = (224,224,3)
pca = PCA(n_components=256)
vectors = np.empty([0,256],dtype=np.float32) 
images = np.empty([0,224,224,3],dtype=np.float32)
i=0
while i < len(path): 
    i2 = i + 256 
    # images = np.array([imread(str(p)).astype(np.float32) for p in path[i:i2]]) 
    imgs = np.array([imread(str(p)).astype(np.float32) for p in path[i:i2]])
    print(images.shape)
    # images = np.asarray([resize(image,SIZE,0) for image in images]) 
    imgs = np.asarray([resize(image,SIZE,0) for image in imgs])
    # print(f"images: {len(images)}")
    print(f"images: {len(imgs)}")
    # print(f"single image shape: {images[0].shape}") 
    print(f"imgs shape: {imgs.shape}") 
    images = np.concatenate((images, imgs),0)

# ------------------------------------------------------------- get embeddings vectors

    vector = model.predict(imgs)
    print(f"model output shape: {vector[0].shape}")
    vector = vector.reshape(vector.shape[0],-1)
    print(f"reshaped to 1D: {vector[0].shape}") 
    if i == 0: 
        pca.fit(vector)
    vector = pca.transform(vector)
    print(f"vector transformed by pca: {vector[0].shape}")
    vectors = np.concatenate((vectors, vector),0)
    i += 256

# ----------------------------------------------------------------------- cluster them
from sklearn.cluster import KMeans

CLUSTERS = args.clusters 
CLUSTERS = CLUSTERS.split(",")
clusterings = [] 
for i in range(len(CLUSTERS)): 
    clustering = KMeans(n_clusters=int(float(CLUSTERS[i])))
    clustering.fit(vectors)
    cluster = clustering.predict(vectors)
    clusterings.append(cluster)
    print(f"clusters: {cluster}")

# ------------------------------------------------ copy images according their cluster

# import shutil
# for i in range(len(images)):
#   if not os.path.exists(f"output/cluster{cluster[i]}"): os.makedirs(f"output/cluster{cluster[i]}")
#   print(f"cp {path[i]} output/cluster{cluster[i]}")
#   shutil.copy2(f"{path[i]}",f"output/cluster{cluster[i]}")
  
# -------------------------------------------------------------------------- make html
from web import *

for j in range(len(CLUSTERS)):
    # make html section for every cluster
    section = [""]*int(float(CLUSTERS[j]))
    for k in range(len(images)):
        section[clusterings[j][k]] += addimg(f"{path[k]}",f"cluster {clusterings[j][k]}",f"{path[k]}")

    # build the page
    BODY = ""
    for k in range(len(section)):
        BODY += f"<h2>cluster {k}<h2>\n"
        BODY += section[k]
        BODY += "\n\n"
    html = HTML.format(BODY=BODY,CSS=CSS)

    # save html
    print("write: index_kmeans"+str(CLUSTERS[j])+".html")
    with open("index_kmeans"+str(CLUSTERS[j])+".html","w") as fd:
        fd.write(html)

# ------------------------------------------------------------------------------------
