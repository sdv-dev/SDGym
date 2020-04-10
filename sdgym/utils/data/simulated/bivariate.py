import argparse
import itertools
import json
import math
import os
import sys

import numpy as np
from sklearn.datasets import make_circles

from .. import utils

np.random.seed(0)

def create_distribution(dist_type, num_samples):
    if dist_type in ["grid", "gridr"]:
        return make_gaussian_mixture(dist_type, num_samples)
    elif dist_type == "ring":
        return make_gaussian_mixture(dist_type, num_samples, num_components = 8)
    elif dist_type == "2rings":
        return make_two_rings(num_samples)

def make_gaussian_mixture(dist_type, num_samples, num_components = 25, s = 0.05, n_dim = 2):
    """ Generate from Gaussian mixture models arranged in grid or ring
    """
    sigmas = np.zeros((n_dim,n_dim))
    np.fill_diagonal(sigmas, s)
    samples = np.empty([num_samples,n_dim])
    bsize = int(np.round(num_samples/num_components))

    if dist_type == "grid":
        mus = np.array([np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
                                                        range(-4, 5, 2))],dtype=np.float32)
    elif dist_type =="gridr":
        mus = np.array([np.array([i, j]) + (np.random.rand(2) - 0.5)
                        for i, j in itertools.product(range(-4, 5, 2),
                            range(-4, 5, 2))],dtype=np.float32)
    elif dist_type == "ring":
        mus = np.array([[-1,0],[1,0],[0,-1],[0,1],[-math.sqrt(1/2),-math.sqrt(1/2)],[math.sqrt(1/2),math.sqrt(1/2)],[-math.sqrt(1/2),math.sqrt(1/2)],[math.sqrt(1/2),-math.sqrt(1/2)]])

    for i in range(num_components):
        if (i+1)*bsize >= num_samples:
            samples[i*bsize:num_samples,:] = np.random.multivariate_normal(mus[i],sigmas,size = num_samples-i*bsize)
        else:
            samples[i*bsize:(i+1)*bsize,:] = np.random.multivariate_normal(mus[i],sigmas,size = bsize)
    return samples

def make_two_rings(num_samples):
    samples, labels = make_circles(num_samples, shuffle=True, noise=None, random_state=None, factor=0.6)
    return samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate simulated Data for a distribution')
    parser.add_argument('distribution', type = str, help = 'specify type of distributions to sample from')
    parser.add_argument('--sample', type=int, default=10000,
                    help='maximum samples in the simulated data.')
    args = parser.parse_args()
    dist = args.distribution
    num_sample = args.sample
    samples = create_distribution(dist, num_sample*2)
    np.random.shuffle(samples)

    output_dir = "data/simulated"
    try:
        os.mkdir(output_dir)
    except:
        pass



    # Store Meta Files
    meta = []
    for i in range(2):
        meta.append({
                "name": str(i),
                "type": "continuous",
                "min": int(np.min(samples[:,i].astype('float'))) - 1,
                "max": int(np.max(samples[:,i].astype('float'))) + 1
        })
    # Store simulated data
    with open("{}/{}.json".format(output_dir, dist), 'w') as f:
        json.dump(meta, f, sort_keys=True, indent=4, separators=(',', ': '))
    np.savez("{}/{}.npz".format(output_dir, dist), train=samples[:len(samples)//2], test=samples[len(samples)//2:])

    utils.verify("{}/{}.npz".format(output_dir, dist),
        "{}/{}.json".format(output_dir, dist))
