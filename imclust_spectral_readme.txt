
imclust.py (c) R.Jaksa 2021 
imclust_spectral.py - extended version of imclust.py (by A. Gajdos) 

INPUTS: 
-h help; 
-m  models of NN to provide a numerical representations of images. Accepted inputs: see documentation https://www.tensorflow.org/api_docs/python/tf/keras/applications - section 'functions'; 
-c  requested number of clusters; 
-pca number of Principal components; 
- dn dataset name;  
path to images. 

OUTPUTS: 
- html file (containing clusters) for each clustering with given number of clusters; 
- mean silhouette coefficient (plot, exported values in text file); 
- Calinski-Harabasz index (plot, exported values in text file); 
- Davies-Bouldin index (plot, exported values in text file); 
- The COP index (plot, exported values in text file); 
- The SDbw index (plot, exported values in text file).      

EXAMPLES: 
python imclust_spectral.py -c 5,10 -m ResNet50,VGG16 -dn test_celeba d:\Data\Matsuko_UPJS\CLUST-data\test_celeba\ 
python imclust_spectral.py -c 5,10,15 -m DenseNet121,InceptionV3,ResNet50,VGG16 -dn test_celeba d:\Data\Matsuko_UPJS\CLUST-data\test_celeba\ 