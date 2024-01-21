# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 08:51:43 2023

@author: Haowen Zhang
"""
#计算MSM函数
import numpy as np
from get_data import *


def C(x_i,x_i_1,y_i,c):
    if (x_i_1<=x_i and x_i<=y_i) or (x_i_1>=x_i and x_i>=y_i):
        return c
    else:
        return c + min(abs(x_i-x_i_1), abs(x_i-y_i))
    


def delta_y_x(y,x,c):
    UE_y = max(y)
    LE_y = min(y)
    length = len(x)
    res = 0
    for i in range(1,length-1):
        if x[i]>=UE_y:
            res = res + min(abs(x[i]-UE_y),c)
        elif x[i]<=LE_y:
            res = res + min(abs(x[i]-LE_y),c)
        else:
            res = res + 0
    return res

def delta_x_y(x,y,c):
    UE_x = max(x)
    LE_x = min(x)
    length = len(x)
    res = 0
    for i in range(1,length-1):
        if y[i]>=UE_x:
            res = res + min(abs(y[i]-UE_x),c)
        elif y[i]<=LE_x:
            res = res + min(abs(y[i]-LE_x),c)
        else:
            res = res + 0
    return res


def boundary_end(Lx,Lx_1,Ly,Ly_1):
    if Lx_1>=Lx>=Ly or Lx_1<=Lx<=Ly or Ly_1>=Ly>=Lx or Ly_1<=Ly<=Lx:
        tempA = min(abs(Lx-Ly),c)
    else:
        tempA = min(abs(Lx-Ly),c+abs(Lx-Lx_1),c+abs(Ly-Ly_1))
    return tempA
    
def glb_msm(x, y, c):
    leny = len(y)
    lenx = len(x)
    XUE = max(x)
    XLE = min(x)
    YUE = max(y)
    YLE = min(y)

    if y[leny-2]>=y[leny-1]>=x[lenx-1] or y[leny-2]<=y[leny-1]<=x[lenx-1] or x[lenx-2]<=x[lenx-1]<=y[leny-1] or x[lenx-2]>=x[lenx-1]>=y[leny-1]:
        fixed_dist = abs(x[0]-y[0]) + min(abs(x[lenx-1]-y[leny-1]), c)
    else:
        fixed_dist = abs(x[0]-y[0]) + min(
                                        abs(x[lenx-1]-y[leny-1]),
                                        c + abs(y[leny-1] - y[leny-2]),
                                        c + abs(x[lenx-1] - x[lenx-2]))

    y_dist = 0
    for i in range(1, leny-1):

        if y[i] > XUE:
            y_dist += min(abs(y[i]-XUE), c)
        if y[i] < XLE:
            y_dist += min(abs(y[i]-XLE), c)
        
    x_dist = 0
    for i in range(1, lenx-1):

        if x[i] > YUE:
            x_dist += min(abs(x[i]-YUE), c)
        if x[i] < YLE:
            x_dist += min(abs(x[i]-YLE), c)

    lb_dist = fixed_dist + max(y_dist, x_dist)

    return lb_dist


    
    

def compute_ED(x,y):
    
    length = len(x)
    res = 0
    for i in range(length):
        res = res + (x[i]-y[i])**2
    return res

def compute_MSM(x,y,c):
    m = len(x)
    n = len(y)
    cost = np.zeros((m,n))
    cost[0,0] = abs(x[0]-y[0])
    for i in range(1,m):
        cost[i,0] = cost[i-1,0] + C(x[i],x[i-1],y[0],c)
    for j in range(1,n):
        cost[0,j] = cost[0,j-1] + C(y[j],x[0],y[j-1],c)
    for i in range(1,m):
        for j in range(1,n):
            temp_a = cost[i-1,j-1] + abs(x[i]-y[j])
            temp_b = cost[i-1,j] + C(x[i],x[i-1],y[j],c)
            temp_c = cost[i,j-1] + C(y[j],x[i],y[j-1],c)
            cost[i,j] = min(temp_a,temp_b,temp_c)
    return cost[m-1,n-1]





    
    



    
    