"""
\file Distribution.cpp \brief computes

distribution class implementation

Copyright 2007-2022 IMP Inventors. All rights reserved.
"""

import numpy as np
from numba import jit
import scipy as sp

class RadialDistributionFunction(object):
    def __init__(self, bin_size=0.5):
        self.bin_size = bin_size
        self.one_over_bin_size = 1/self.bin_size
        self.distribution = np.array([0.0])
        self.max_distance_ = 50.0

    def scale(self, c):
        self.distribution *= c # [value * c for value in self.distribution]

    def add(self, other_rd):
        for i in range(min(self.distribution.shape[0], other_rd.distribution.shape[0])):
            self.distribution[i] += other_rd.distribution[i]

    def normalize(self):
        # Calculate area
        area = np.sum(self.distribution)

        # Normalize
        self.distribution /= area # [value / area for value in self.distribution]

    def add_to_distribution(self, r, value):
        index = get_index_from_distance(self.bin_size, r)
        if index >= self.distribution.shape[0]:
            ext = np.zeros(index-self.size()+1) # [0.0] * (index-self.size()+1)
            self.distribution = np.concatenate((self.distribution, ext))
            self.max_distance_ = get_distance_from_index(self.bin_size, index + 1)
        self.distribution[index] += value

    def add_to_distribution_many(self, dists, values):
        if len(dists) > 0:
            # dist_bins = np.round(dists*self.one_over_bin_size)

            # if dist_bins.max() > self.distribution.shape[0]:
            #     ext = np.zeros(int(dist_bins.max())-self.distribution.shape[0]+1)
            #     self.distribution = np.concatenate((self.distribution, ext))

            # new_bins = np.arange(0, self.distribution.shape[0]+1)
            # res = sp.stats.binned_statistic(dist_bins, values, 'sum', bins=new_bins)
            # self.distribution += res.statistic

            # self.max_distance = self.bin_size*(self.distribution.shape[0]+1)

            self.distribution =  inner_add_to_distribution_many(dists,
                values, self.distribution, self.one_over_bin_size)

            self.max_distance = self.bin_size*(self.distribution.shape[0]+1)

    def size(self):
        return self.distribution.shape[0]

    def add_distribution(self, distribution):
        self.distribution = distribution
        self.max_distance = self.bin_size*(self.distribution.shape[0]+1)


# @jit(nopython=True)
def get_index_from_distance(bin_size, dist):
        return round(dist * 1/bin_size)

# @jit(nopython=True)
def get_distance_from_index(bin_size, index):
    return index * bin_size

@jit(nopython=True, cache=True)
def inner_add_to_distribution_many(dists, values, distribution, one_over_bin_size):
    max_val = int(dists.max()*one_over_bin_size+0.5)
    if max_val > distribution.shape[0]:
        ext = np.zeros(max_val-distribution.shape[0]+1)
        distribution = np.concatenate((distribution, ext))

    for i in range(len((dists))):
        distribution[int(dists[i]*one_over_bin_size+0.5)] += values[i]

    return distribution
