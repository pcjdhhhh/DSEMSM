# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 15:10:14 2023

@author: Haowen Zhang
"""
import numpy as np
from MSM import *
from get_data import *
from tools import *
import random

def search_with_MSM_from_first_K(train_data,test_data,k,c,first_K):

    n_test = test_data.shape[0]
    len_first_K = first_K.shape[1]
    res = np.zeros((n_test,k))   
    
    for i in range(n_test):
        #print(i)
        query = test_data[i,:]
        min_k = np.ones(k) * np.inf    
        for j in range(len_first_K):
            temp_MSM = compute_MSM(query,train_data[int(first_K[i,j]),:],c)
            min_ = max(min_k)
            if temp_MSM<min_:
                location = np.where(min_k==min_)[0][0]   
                res[i,location] = int(first_K[i,j])   
                min_k[location] = temp_MSM
    return res


def LS(train_data,test_data,k,c):
   
    n_test = test_data.shape[0]
    n_train = train_data.shape[0]
    res = np.zeros((n_test,k))   
    
    for i in range(n_test):
        #print(i)
        query = test_data[i,:]
        min_k = np.ones(k) * np.inf   
        for j in range(n_train):
            temp_MSM = compute_MSM(query,train_data[j,:],c)
            min_ = max(min_k)
            if temp_MSM<min_:
                location = np.where(min_k==min_)[0][0]   
                res[i,location] = j   #
                min_k[location] = temp_MSM
    return res




def GLB(train_data,test_data,k,c):
    
    n_test = test_data.shape[0]
    n_train = train_data.shape[0]
    res = np.zeros((n_test,k))   
    
    for i in range(n_test):
        query = test_data[i,:]
        lower_bound_save = np.zeros(n_train)
        for j in range(n_train):
            lower_bound_save[j] = glb_msm(query,train_data[j,:],c)   #lower bound computation
        lower_bound_sort = np.argsort(lower_bound_save)
        min_k = np.ones(k) * np.inf    
        for j in range(n_train):
            min_ = max(min_k)
            visit = lower_bound_sort[j]
            lower_bound = lower_bound_save[visit]
            if lower_bound<min_:
                temp_MSM = compute_MSM(query,train_data[visit,:],c)
                if temp_MSM<min_:
                    location = np.where(min_k==min_)[0][0]   
                    res[i,location] = visit   #
                    min_k[location] = temp_MSM
    return res




def FV(train_data,test_data,k,c,candidate_num,index,train_vectors):
   
    n_test = test_data.shape[0]
    n_train = train_data.shape[0]
    res = np.zeros((n_test,k))   
    temp_res = np.zeros((n_test,candidate_num))
    reduced_dim = len(index)
    for i in range(n_test):
        #query to vectors
        query = test_data[i,:]
        vector_query = np.array([compute_MSM(query, train_data[index[j],:],c) for j in range(reduced_dim)])
        vector_dis = np.array([compute_ED(vector_query,train_vectors[j,:]) for j in range(n_train)])
        candidate_index = np.argsort(vector_dis)[0:candidate_num]
        min_k = np.ones(k) * np.inf    
        for j in range(candidate_num):
            temp_MSM = compute_MSM(query,train_data[candidate_index[j],:],c)
            min_ = max(min_k)
            if temp_MSM<min_:
                location = np.where(min_k==min_)[0][0]   
                res[i,location] = candidate_index[j]   
                min_k[location] = temp_MSM
    return res


def RF(train_data,test_data,k,c,candidate_num):
    
    n_test = test_data.shape[0]
    n_train = train_data.shape[0]
    res = np.zeros((n_test,k))   
    for i in range(n_test):
        query = test_data[i,:]
        candidate_index = random.sample(range(0,n_train),candidate_num)
        min_k = np.ones(k) * np.inf    
        for j in range(candidate_num):
            temp_MSM = compute_MSM(query,train_data[candidate_index[j],:],c)
            min_ = max(min_k)
            if temp_MSM<min_:
                location = np.where(min_k==min_)[0][0]   
                res[i,location] = candidate_index[j]   
                min_k[location] = temp_MSM
    return res
        
        
        





def DSEMSM(train_data,test_data,k,c,candidate_num,index,train_vectors):
    
    n_test = test_data.shape[0]
    n_train = train_data.shape[0]
    res = np.zeros((n_test,k))   
    temp_res = np.zeros((n_test,candidate_num))
    reduced_dim = len(index)
    
    for i in range(n_test):
        
        query = test_data[i,:]
        vector_query = np.array([compute_MSM(query, train_data[index[j],:],c) for j in range(reduced_dim)])   #lower bound
        vector_dis = np.array([compute_ED(vector_query,train_vectors[j,:]) for j in range(n_train)])
        candidate_index = np.argsort(vector_dis)[0:candidate_num]
        lower_bound_save = np.zeros(candidate_num)
        for j in range(candidate_num):
            lower_bound_save[j] = glb_msm(query,train_data[candidate_index[j],:],c)
        
        lower_bound_sort = np.argsort(lower_bound_save)
        visit_index = candidate_index[lower_bound_sort]   
        
        min_k = np.ones(k) * np.inf    
        
        for j in range(candidate_num):
            min_ = max(min_k)
            visit = visit_index[j]
            lower_bound = lower_bound_save[lower_bound_sort[j]]   #sort
            if lower_bound<min_:
                
                temp_MSM = compute_MSM(query,train_data[visit,:],c)
                if temp_MSM<min_:
                    location = np.where(min_k==min_)[0][0]   
                    res[i,location] = visit   
                    min_k[location] = temp_MSM
    return res
        
        


        
        
        
        
        







                
        
    