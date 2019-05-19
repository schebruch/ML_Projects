# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:15:36 2019

@author: scheb
"""

from vectorized_functions import *
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

#code to load boston data, taken from: https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155
def load_data():
    boston_dataset = datasets.load_boston()
    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    boston['target'] = boston_dataset.target

    X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
    Y = boston['target']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

    X_train = X_train.values
    X_test = X_test.values
    Y_train = Y_train.values
    Y_test = Y_test.values
    
    
    return X_train, X_test, Y_train, Y_test

#X is d x m, where d is the number of features
#Y is 1 x m
def gradient_descent(X_train, X_test, Y_train, Y_test, num_iters = 10000, eta = .00002475, record_every = 1000):
    
    #initialize params to 0
    b = 0.0
    w = np.zeros([X_train.shape[0],1]) 

    #initialize a loss log
    log = []
    for i in range(num_iters):
        #Z is 1 x m predictions
        Ztrain = linear(w, b, X_train)
        w = np.subtract(w, eta*(dW(Ztrain, X_train, Y_train)))
        b -= eta*db(Ztrain, Y_train)
        
        
        if (i+1) % record_every == 0:  
            Ztest = linear(w, b, X_test)
            trLoss = loss(Ztrain, Y_train)
            testLoss = loss(Ztest, Y_test)
            log.append((trLoss, testLoss))
    
    return w, b, log
    
    
    
    
if __name__ == "__main__":
   X_tr, X_te, y_tr, y_te = load_data()
   w, b, log = gradient_descent(X_tr.T, X_te.T, y_tr.T, y_te.T)
    
   with open('data/results.pkl', 'wb') as f:
       pickle.dump((w, b, log), f)

   print(log) #training loss on left, testing loss on the right. We can see that the loss converges
   
   print("w: ", w)
   print("b: ", b)

        
    