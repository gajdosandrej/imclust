
imclust.py (c) R.Jaksa 2021 
imclust_kmeans.py - edited version of imclust.py (by A. Gajdos) 

INPUTS: 
-m  Model of NN to provide a numerical representation of images. Accepted inputs: see documentation https://www.tensorflow.org/api_docs/python/tf/keras/applications - section 'functions'; 
-c  Requested number of clusters; 
path to images. 

EXAMPLE: 
python imclust_kmeans.py -c 5,10 -m VGG16 d:\Data\Matsuko_UPJS\alpha82_pc\partial_inpainting_on_sample\