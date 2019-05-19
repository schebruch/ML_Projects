# -*- coding: utf-8 -*-
"""
Created on Fri May 17 23:33:36 2019

@author: scheb
"""

import numpy as np

#Evaluate <w,X> + b
def linear(w, b, X):
    return (w.T).dot( X ) + b

#evaluate the mean squared error loss
def loss(Predictions, Actual):
    """
    Predictions and actual are both m x 1 matrices
    """
    numExamples = Predictions.shape[1]
    return (np.linalg.norm(Predictions - Actual) )** 2 / numExamples

#evaluate the gradient with respect to w for a vector of training examples
#Z = <w, X> + b
def dW(Z, X, Y):
 #   return (Z - Y).dot(X.T)   
    return X.dot((Z - Y).T)/X.shape[1]


#evalue the gradient with respect to b for a vector of training examples 
#Z = <w,X> + b
def db(Z, Y):
    return np.mean(Z-Y)


