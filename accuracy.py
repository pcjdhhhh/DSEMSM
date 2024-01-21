# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:23:27 2023

@author: Haowen Zhang
"""

import numpy as np
from MSM import *
from get_data import *
from tools import *
import random
import math
from search_function import *
from preprocessing import *
import time
import os



file_name_save = ['CBF']

for file_name in file_name_save:

    
    print('file_name: ',file_name)
    train_data,train_label,test_data,test_label = get_time_series(file_name)
    all_data = np.vstack((train_data,test_data))
    query_num = math.ceil(all_data.shape[0] * 0.01)
    test_data = all_data[0:query_num,:]
    train_data = all_data[query_num:,:]
    print('train_data: ',train_data.shape)
    print('test_data: ',test_data.shape)
    
    
    
    
    #parameters settings according to experimental settings
    candidate_num = 0.2 
    candidate_num = math.ceil(train_data.shape[0]*candidate_num)
    print('candidate_num: ',candidate_num)
    c = 0.5
    reduced_dim = 8
    first_index = 1
    
    
    
    #furthest index generated, if saved beforehand, can be directly read
    furthest_index_save_path = 'furthest_index_save/' + file_name
    furthest_train_vectors_save_path = 'furthest_train_vectors_save/' + file_name
    if os.path.exists(furthest_index_save_path):
        #print('furthest_index exists')
        furthest_index = np.loadtxt(furthest_index_save_path)
        furthest_index = furthest_index.astype('int')
        furthest_train_vectors = np.loadtxt(furthest_train_vectors_save_path)
    else:
        #furthest generated
        #print('furthest_index no exists')
        furthest_index = generate_with_furthest(train_data,reduced_dim,first_index,c)
        furthest_train_vectors = vector_representation(train_data,furthest_index,c)
        np.savetxt(furthest_index_save_path,furthest_index)
        np.savetxt(furthest_train_vectors_save_path,furthest_train_vectors)
    
    
    
    
    
    #The generation of random indexes, if saved beforehand, can be directly read
    random_embedding_save_path = 'random_embedding_save/' + '_' + file_name
    if os.path.exists(random_embedding_save_path):
        random_index = np.loadtxt(random_embedding_save_path).astype(int)
        random_train_vectors = vector_representation(train_data,random_index,c)
    else:
        random_index = np.array(random.sample(range(0,train_data.shape[0]),reduced_dim))
        random_train_vectors = vector_representation(train_data,random_index,c)
        np.savetxt(random_embedding_save_path,random_index)
    
    #First, calculate 50-NN to avoid unnecessary computations
    k=50
    KNN_file_path = 'KNN_index_results/' + file_name + '_' + str(50)
    if os.path.exists(KNN_file_path):
        print('50-NN exists')
        #brute_force_res = np.array([int(np.loadtxt(KNN_file_path))])
        brute_force_res_50 = np.loadtxt(KNN_file_path)
        brute_force_res_50 = brute_force_res_50.reshape((len(brute_force_res_50),-1))
        brute_force_res_50.astype('int')
    else:
        print('50-NN no exist')
        brute_force_res_50 = LS(train_data,test_data,k,c)
        brute_force_res_50.astype('int')
        np.savetxt(KNN_file_path,brute_force_res_50)
    
    
    
    kk=[1,5,10,20,50]
    for k in kk:
        print(k)
        #check if the inspection results of KNN have been saved
        KNN_file_path = 'KNN_index_results/' + file_name + '_' + str(k)
        if os.path.exists(KNN_file_path):
            print(str(k) + '_exists')
            brute_force_res = np.loadtxt(KNN_file_path)
            brute_force_res = brute_force_res.reshape((len(brute_force_res),-1))
        else:
            print(str(k) + '_no exist')
            brute_force_res = search_with_MSM_from_first_K(train_data,test_data,k,c,brute_force_res_50)
            print('brute-force time: ',e-s)
            np.savetxt(KNN_file_path,brute_force_res)
            
            
        
        
        f_and_r_res = FV(train_data,test_data,k,c,candidate_num,furthest_index,furthest_train_vectors)
        print('DSEMSM accuracy: ',compute_overlapping(brute_force_res,f_and_r_res))
        
        
        random_index_res = FV(train_data,test_data,k,c,candidate_num,random_index,random_train_vectors)
        print('RI accuracy: ',compute_overlapping(brute_force_res,random_index_res))
        
        
        
        
        random_filter_res = RF(train_data,test_data,k,c,candidate_num)
        print('RF accuracy: ',compute_overlapping(brute_force_res,random_filter_res))
        
        
        print('-------------------------------------------------')