import unittest
import cnn
import numpy as np
import os, sys
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

class TestBDGP(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_fit_circles(self):
        noisy_circles, _ = datasets.make_circles(
            n_samples=200,
            factor=.5,
            noise=.05,
            random_state=8,
            )

        noisy_circles = StandardScaler().fit_transform(noisy_circles)

        params = {
            'radius_cutoff': 0.5,
            'cnn_cutoff': 0,
            'member_cutoff': 1,
            'max_clusters': None
            }

        cobj = cnn.CNN(train=noisy_circles)

        cobj.fit(
            radius_cutoff=params['radius_cutoff'],
            cnn_cutoff=params['cnn_cutoff'],
            member_cutoff=params['member_cutoff'],
            max_clusters=params['max_clusters']
            )

if __name__ == '__main__':
    unittest.main()

# sample = 'samples/test_sample.npy' 
# if not os.path.isfile(sample):
#     print("No data found. Preparing data ...\n")  
#     noisy_circles, _ = datasets.make_circles(n_samples=5000, factor=.5, noise=.05)
#     np.save(sample, noisy_circles)

# cobj.load(sample)
# print("Creating 2D scatter plot ...\n")  
# cobj.evaluate(show=False, save=True, output='samples/eval1.pdf')
# print("Creating distance histogram ...\n")  
# cobj.dist_hist(show=False, save=True, output='samples/hist1.pdf')
# print(cobj)

# print("Reducing data set ...\n")
# cobj.cut(points=(None, None, 5))
# print("Creating 2D scatter plot ...\n")  
# cobj.evaluate(show=False, save=True, output='samples/eval2.pdf')
# cobj.dist()
# print("Creating distance histogram ...\n") 
# cobj.dist_hist(show=False, save=True, output='samples/hist2.pdf')
# print(cobj)

# print("Clustering ...\n") 
# cobj.fit(radius_cutoff=0.1, cnn_cutoff=2)
# print(cobj)

# print("Isolating clusters ...\n")
# cobj.isolate()

# cobj.train_children[1].fit(radius_cutoff=0.055, cnn_cutoff=2)
# cobj.train_children[1].isolate()
# cobj.reel()