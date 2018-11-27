print(
"""

    Running test for cnn module functionality
    
"""
)

print("Importing ...\n")
import cnn
import numpy as np
import os, sys
from sklearn import datasets

cobj = cnn.CNN()

sample = 'samples/test_sample.npy' 
if not os.path.isfile(sample):
    print("No data found. Preparing data ...\n")  
    noisy_circles, _ = datasets.make_circles(n_samples=5000, factor=.5, noise=.05)
    np.save(sample, noisy_circles)
print("Loading data ...\n")    
cobj.load(sample)
print("Creating 2D scatter plot ...\n")  
cobj.evaluate(show=False, save=True, output='samples/eval1.pdf')
print("Creating distance histogram ...\n")  
cobj.dist_hist(show=False, save=True, output='samples/hist1.pdf')
print(cobj)

print("Reducing data set ...\n")
cobj.cut(points=(None, None, 5))
print("Creating 2D scatter plot ...\n")  
cobj.evaluate(show=False, save=True, output='samples/eval2.pdf')
cobj.dist()
print("Creating distance histogram ...\n") 
cobj.dist_hist(show=False, save=True, output='samples/hist2.pdf')
print(cobj)

print("Clustering ...\n") 
cobj.cluster(radius_cutoff=0.1, cnn_cutoff=2)
print(cobj)

print("Isolating clusters ...\n")
cobj.isolate()

cobj.train_children[1].cluster(radius_cutoff=0.055, cnn_cutoff=2)
cobj.train_children[1].isolate()
cobj.reel()