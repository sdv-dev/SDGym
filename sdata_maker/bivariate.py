import itertools
import json
import os
import sys
from math import sqrt

import numpy as np
from sklearn.datasets import make_circles

import utils
 
np.random.seed(0)

def create_distribution(dist_type, num_samples):
    if dist_type == "gaussian_grid":
        return make_gaussian_mixture(dist_type, num_samples)
    elif dist_type == "gaussian_ring":
        return make_gaussian_mixture(dist_type, num_samples, num_components = 8)
    elif dist_type == "two_rings":
        return make_two_rings(num_samples)

def make_gaussian_mixture(dist_type, num_samples, num_components = 25, s = 0.05, n_dim = 2):
    """ Generate from Gaussian mixture models arranged in grid or ring
    """
    sigmas = np.zeros((n_dim,n_dim))
    np.fill_diagonal(sigmas, s)
    samples = np.empty([num_samples,n_dim])
    bsize = int(np.round(num_samples/num_components))

    if dist_type == "gaussian_grid":
        mus = np.array([np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
                                                        range(-4, 5, 2))],dtype=np.float32)
    elif dist_type == "gaussian_ring":
        mus = np.array([[-1,0],[1,0],[0,-1],[0,1],[-sqrt(1/2),-sqrt(1/2)],[sqrt(1/2),sqrt(1/2)],[-sqrt(1/2),sqrt(1/2)],[sqrt(1/2),-sqrt(1/2)]])

    for i in range(num_components):
        if (i+1)*bsize >= num_samples:
            samples[i*bsize:num_samples,:] = np.random.multivariate_normal(mus[i],sigmas,size = num_samples-i*bsize)
        else:
            samples[i*bsize:(i+1)*bsize,:] = np.random.multivariate_normal(mus[i],sigmas,size = bsize)
    return samples

def make_two_rings(num_samples):
    samples, labels = make_circles(num_samples, shuffle=True, noise=None, random_state=None, factor=0.8)
    return samples

if __name__ == "__main__":
    output_dir = "data/real"
    try:
        os.mkdir(output_dir)
    except:
        pass
    if len(sys.argv) != 3:
        raise ValueError("Please specify the name of the distribution and number of samples to generate")

    dist_type = sys.argv[1]
    num_samples = int(sys.argv[2])

    samples = create_distribution(dist_type, num_samples*2)

    # Store Meta Files
    meta = []
    for i in range(2):
        meta.append({
                "name": str(i),
                "type": "continuous",
                "min": np.min(samples[:,i].astype('float')),
                "max": np.max(samples[:,i].astype('float'))
        })
    # Store synthetic data
    with open("{}/{}.json".format(output_dir, dist_type), 'w') as f:
        json.dump(meta, f, sort_keys=True, indent=4, separators=(',', ': '))
    np.savez("{}/{}.npz".format(output_dir, dist_type), train=samples[:len(samples)//2], test=samples[len(samples)//2:])



    
    



