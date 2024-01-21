# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 08:22:18 2023

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



file_name_save = ['Wafer']

for file_name in file_name_save:

    # read datasets
    
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
    
    
    kk=[1,5,10,20,50]
    
    
    #furthest index generated, if saved beforehand, can be directly read
    furthest_index_save_path = 'furthest_index_save/' + file_name
    furthest_train_vectors_save_path = 'furthest_train_vectors_save/' + file_name
    if os.path.exists(furthest_index_save_path):
        #print('furthest_index exists')
        furthest_index = np.loadtxt(furthest_index_save_path)
        furthest_index = furthest_index.astype('int')
        furthest_train_vectors = np.loadtxt(furthest_train_vectors_save_path)
    else:
        #print('furthest_index no exists')
        furthest_index = generate_with_furthest(train_data,reduced_dim,first_index,c)
        furthest_train_vectors = vector_representation(train_data,furthest_index,c)
        np.savetxt(furthest_index_save_path,furthest_index)
        np.savetxt(furthest_train_vectors_save_path,furthest_train_vectors)
    
    for k in kk:
        print(k)
        #First, check if the inspection results of KNN have been saved
        KNN_file_path = 'KNN_index_results/' + file_name + '_' + str(k)
        if os.path.exists(KNN_file_path):
            print('exists')
            #brute_force_res = np.array([int(np.loadtxt(KNN_file_path))])
            brute_force_res = np.loadtxt(KNN_file_path)
            brute_force_res = brute_force_res.reshape((len(brute_force_res),-1))
        else:
            print('no exist')
            s = time.time()
            brute_force_res = LS(train_data,test_data,k,c)
            e = time.time()
            print('LS: ',e-s)
            np.savetxt(KNN_file_path,brute_force_res)
            
            
        
        
        
        
        
        
        s = time.time()
        brute_force_res = LS(train_data,test_data,k,c)
        e = time.time()
        print('LS time: ',e-s)
        
        
        
        s = time.time()
        lower_bound_res = GLB(train_data,test_data,k,c)
        e = time.time()
        print('GLB time: ',e-s)
        
        '''
        Note that the performance of the DSEMSM algorithm is closely related to that of GLB. Specifically, when GLB performs well 
        (for instance, on datasets ElectricD. and Wafer), DSEMSM is more efficient than FV. However, when GLB is invalid 
        (for example, on datasets ECGFiveDays and S.Control), the performance of DSEMSM is close to FV.
        '''
        
        
        s = time.time()
        f_and_r_res = FV(train_data,test_data,k,c,candidate_num,furthest_index,furthest_train_vectors)
        e = time.time()
        print('FV time: ',e-s)
        
        
        
        s = time.time()
        f_and_r_lower_bound_res = DSEMSM(train_data,test_data,k,c,candidate_num,furthest_index,furthest_train_vectors)
        e = time.time()
        print('DSEMSM time: ',e-s)
        
        print('-------------------------------------------------')
        
        
        

