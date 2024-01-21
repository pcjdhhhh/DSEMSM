# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 16:39:47 2024

"""
#有varification和没varication

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


#time.sleep(30)
'''
file_name_save = ['CBF','Computers','DiatomSizeReduction','Earthquakes','ECG5000','ECGFiveDays','FacesUCR','FiftyWords','Fish','MedicalImages','FaceAll','CricketX','CricketY','CricketZ','InsectWingbeatSound','DistalPhalanxOutlineAgeGroup',
                  'DistalPhalanxOutlineCorrect','DistalPhalanxTW','MiddlePhalanxOutlineAgeGroup','Adiac','LargeKitchenAppliances','MiddlePhalanxOutlineCorrect','MiddlePhalanxTW','MoteStrain','OSULeaf','PhalangesOutlinesCorrect','ProximalPhalanxOutlineAgeGroup','ProximalPhalanxOutlineCorrect','ProximalPhalanxTW','RefrigerationDevices','ScreenType','ShapesAll','SmallKitchenAppliances','SonyAIBORobotSurface1',
                  'SonyAIBORobotSurface2','Strawberry','SwedishLeaf','Symbols','SyntheticControl','ToeSegmentation1','TwoLeadECG','TwoPatterns','Wafer','WordSynonyms','ChlorineConcentration','ElectricDevices','ItalyPowerDemand','Chinatown','Crop','GunPointAgeSpan','GunPointMaleVersusFemale','GunPointOldVersusYoung',
                  'InsectEPGRegularTrain','InsectEPGSmallTrain','SmoothSubspace','PowerCons','FreezerRegularTrain','FreezerSmallTrain']
'''
file_name_save = ['CBF']
k=50
print(k)
for file_name in file_name_save:

    #data sets
    
    print('file_name: ',file_name)
    train_data,train_label,test_data,test_label = get_time_series(file_name)
    all_data = np.vstack((train_data,test_data))
    query_num = math.ceil(all_data.shape[0] * 0.01)
    test_data = all_data[0:query_num,:]
    train_data = all_data[query_num:,:]
    

    candidate_num = k    
    c = 0.5
    reduced_dim = 8
    #k = 1
    first_index = 1
    
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
        s = time.time()
        furthest_index = generate_with_furthest(train_data,reduced_dim,first_index,c)
        furthest_train_vectors = vector_representation(train_data,furthest_index,c)
        e = time.time()
        #print('preprocessing time: ',e-s)
        np.savetxt(furthest_index_save_path,furthest_index)
        np.savetxt(furthest_train_vectors_save_path,furthest_train_vectors)
        
    
        
    
    
    
    
    
    
   
    KNN_file_path = 'KNN_index_results/' + file_name + '_' + str(50)
    if os.path.exists(KNN_file_path):
        #print('50-NN exists')
        #brute_force_res = np.array([int(np.loadtxt(KNN_file_path))])
        brute_force_res_50 = np.loadtxt(KNN_file_path)
        brute_force_res_50 = brute_force_res_50.reshape((len(brute_force_res_50),-1))
        brute_force_res_50.astype('int')
    else:
        #print('50-NN no exist')
        s = time.time()
        brute_force_res_50 = search_with_MSM(train_data,test_data,k,c)
        brute_force_res_50.astype('int')
        e = time.time()
        #print('brute-force time: ',e-s)
        np.savetxt(KNN_file_path,brute_force_res_50)
    
    KNN_file_path = 'KNN_index_results/' + file_name + '_' + str(k)
    if os.path.exists(KNN_file_path):
        #print(str(k) + '_exists')
        #brute_force_res = np.array([int(np.loadtxt(KNN_file_path))])
        brute_force_res = np.loadtxt(KNN_file_path)
        brute_force_res = brute_force_res.reshape((len(brute_force_res),-1))
    else:
        #print(str(k) + '_no exist')
        s = time.time()
        brute_force_res = search_with_MSM_from_first_K(train_data,test_data,k,c,brute_force_res_50)
        e = time.time()
        #print('brute-force time: ',e-s)
        np.savetxt(KNN_file_path,brute_force_res)
        
    s = time.time()
    f_and_r_res = FV(train_data,test_data,k,c,candidate_num,furthest_index,furthest_train_vectors)
    e = time.time()
    #print('filter and refine time: ',e-s)
    print('furthest_index accuracy: ',compute_overlapping(brute_force_res,f_and_r_res))
    
    print('--------------------------------------')
    
    
    
   