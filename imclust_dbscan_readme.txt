
imclust.py (c) R.Jaksa 2021 
imclust_dbscan.py - extended version of imclust.py (by A. Gajdos) 

INPUTS: 
-m  Models of NN to provide a numerical representations of images. Accepted inputs: see documentation https://www.tensorflow.org/api_docs/python/tf/keras/applications - section 'functions'; 
-e  The maximum distance between two samples for one to be considered as in the neighborhood of the other; 
path to images. 

OUTPUTS: 
- html file (containing clusters) for each clustering with given model; 
- mean silhouette coefficient (plot, exported values in text file); 
- Calinski-Harabasz index (plot, exported values in text file); 
- Davies-Bouldin index (plot, exported values in text file); 
- The COP index (plot, exported values in text file); 
- The SDbw index (plot, exported values in text file).      

EXAMPLE: 
python imclust_dbscan.py -e 600,700 -m DenseNet121,DenseNet169 d:\Data\Matsuko_UPJS\alpha82_pc\partial_inpainting_on_sample\ 

