# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:55:19 2019

@author: scheb
"""

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import scipy.io as sio
import pandas as pd
from PIL import Image
def read_image():
    img = cv2.imread('eye.png')  
    image_array = Image.fromarray(img , 'RGB')
    resize_img = image_array.resize((64 , 64))
    np.array(resize_img).shape

    img.shape