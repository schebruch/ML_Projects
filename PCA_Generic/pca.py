# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:04:14 2019

@author: scheb
"""

import numpy as np
import matplotlib.pyplot as plt

#initialize a PCA object
class PCA:
    def __init__(self, data, max_percent_explained = .8):
        self.data = data #original data
        self.cov = get_cov(data) #covariance matrix
        self. top_values, self.top_components = get_top_components(self.cov, max_percent_explained) #top eigenvalues (normalized) and top eigenvectors (normalized)
        self.projected_data = project(self.top_components, data) #new data
        self.variance_explained = variance_explained(self.top_values) #total variance explained by the data = sum(used eigenvalues)/sum(total eigenvalues)



#return the covariance matrix of X
def get_cov(X):
    return np.cov(X)

#return the normalized top percent explained eigenvalues and the top percent_explained eigenvectors of Cov
def get_top_components(Cov, percent_explained = .8):
    values, vectors = np.linalg.eig(Cov) #the vectors are already normalized 
    idx = values.argsort() [::-1]  
    values = values[idx]
    vectors = vectors[:,idx]
    
    total_sum = np.sum(values)
    current_sum = 0
    i = 0
    while current_sum/total_sum < percent_explained and i < len(values):
        current_sum += values[i]
        i += 1
    
    return values[0:i]/total_sum, vectors[:, :i]

#return the components.T * X.      
def project(components, X):
    return (components.T).dot(X)

#return total variance explained by the data 
def variance_explained(top_values):
    return np.sum(top_values)
   





