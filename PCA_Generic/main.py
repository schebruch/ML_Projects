# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:12:50 2019

@author: scheb
"""
from pca import *
from sklearn import datasets

#we will obtain the variance, old data's shape, and new data's shape from the digits dataset
if __name__ == "__main__":
    X = (datasets.load_digits().data).T
    pca = PCA(X)
    
    print("Variance explained: {}%".format(round(100*pca.variance_explained, 2)))
    
    print("Original shape: {}".format(X.shape))
    
    projected_data = pca.projected_data
    print("New data shape: {}. This was reduced by {} dimensions".format(projected_data.shape, X.shape[0] - projected_data.shape[0]))
