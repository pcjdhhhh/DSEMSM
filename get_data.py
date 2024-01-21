# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 08:46:33 2023

@author: Haowen Zhang
"""
from scipy.io import loadmat
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt

def get_time_series(file_name):
    train_file_path = '../' + 'UCR dataset/' + file_name + '/' + 'train.mat'
    test_file_path = '../' + 'UCR dataset/' + file_name + '/' + 'test.mat'
    train = loadmat(train_file_path)
    test = loadmat(test_file_path)
    train = train['train']
    test = test['test']
    train_label = train[:,0]   #train data label
    test_label = test[:,0]     #test data label
    if file_name=='ChinaTown':             #ChinaTown  needs to be converted to the data type
        train_data = train[:,1:].astype('float')   
        test_data = test[:,1:].astype('float')     
    else:
        train_data = train[:,1:]   
        test_data = test[:,1:]     
    [num,dim] = train_data.shape
    for i in range(num):
        temp = stats.zscore(train_data[i],ddof=1)
        train_data[i] = temp
    
    [num,dim] = test_data.shape
    for i in range(num):
        temp = stats.zscore(test_data[i],ddof=1)
        test_data[i] = temp
    res = np.vstack((train_data,test_data))
    return train_data,train_label,test_data,test_label







    
    
            


