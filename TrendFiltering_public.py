#!/usr/bin/env python
# coding: utf-8

import numpy as np
import numpy.random as rd
import pandas as pd
import os
import folium
from scipy.sparse import dia_matrix, identity
from scipy.sparse.linalg import inv, spsolve
import scipy
from scipy.stats import truncnorm,norm
import statistics as stat

def prox(g,x):
    cell1 = x < (-g)
    cell2 = ((-g) <= x) & (x <= g)
    cell3 = g < x
    ret   = g * cell1 + (-g)*cell3 - x * cell2 + x
    return ret

def difference_matrix(k,n):
    if k == 1:
        data = np.array([np.ones(n), -1.0*np.ones(n)])
        offsets = np.array([-1, 0])
        D0_pre = dia_matrix((data, offsets), shape=(n, n))
        D0     = scipy.sparse.csr_matrix((np.delete(D0_pre.todense(),0,0)))
        D       = D0
    elif k ==2:
        data = np.array([np.ones(n), -2.0*np.ones(n), np.ones(n)])
        offsets = np.array([1, 0, -1])
        D1_pre = dia_matrix((data, offsets), shape=(n, n))
        D1     = scipy.sparse.csr_matrix(np.delete(np.delete(D1_pre.todense(),0,0),-1,0))
        D       = D1
    elif k ==3:
        data = np.array([-1.0*np.ones(n),3.0*np.ones(n),-3.0*np.ones(n),np.ones(n)])
        offsets = np.array([-1,0,1,2])
        D2_pre = dia_matrix((data, offsets), shape=(n, n))
        D2     = scipy.sparse.csr_matrix( np.delete(np.delete(np.delete(D2_pre.todense(),0,0),-1,0 ) , -1,0))
        D       = D2
    return D
        

def spline_smoother(X,param_smooth):
# construct matrix D
    n=len(X)
    data = np.array([-1.0*np.ones(n),3.0*np.ones(n),-3.0*np.ones(n),np.ones(n)])
    offsets = np.array([-1,0,1,2])
    D_pre = dia_matrix((data, offsets), shape=(n, n))
    D     = scipy.sparse.csr_matrix( np.delete(np.delete(np.delete(D_pre.todense(),0,0),-1,0 ) , -1,0))
    C     = identity(n) + param_smooth* np.dot(D.T,D)
    ret   = spsolve(C,X)
    return ret

def linear_spline_smoother(X,param_smooth):
# construct matrix D
    n=len(X)
    data = np.array([np.ones(n), -2.0*np.ones(n), np.ones(n)])
    offsets = np.array([1, 0, -1])
    D_pre = dia_matrix((data, offsets), shape=(n, n))
    D     = scipy.sparse.csr_matrix(np.delete(np.delete(D_pre.todense(),0,0),-1,0))
    C     = identity(n) +param_smooth * np.dot(D.T,D)
    ret   = spsolve(C,X)
    return ret



def trend_filtering(X,k,param_regularization,param_Lagrange,e_tolerance=5):
    
    ## Parameters used
    n         = len(X)
    if k == 2:
        data = np.array([np.ones(n), -2.0*np.ones(n), np.ones(n)])
        offsets = np.array([1, 0, -1])
        D_pre = dia_matrix((data, offsets), shape=(n, n))
        D     = scipy.sparse.csr_matrix(np.delete(np.delete(D_pre.todense(),0,0),-1,0))
        C     = identity(n) + param_Lagrange* np.dot(D.T,D)
        
    if k == 3:
        data = np.array([-1.0*np.ones(n),3.0*np.ones(n),-3.0*np.ones(n),np.ones(n)])
        offsets = np.array([-1,0,1,2])
        D_pre = dia_matrix((data, offsets), shape=(n, n))
        D     = scipy.sparse.csr_matrix( np.delete(np.delete(np.delete(D_pre.todense(),0,0),-1,0 ) , -1,0))
        C     = identity(n) +  param_Lagrange* np.dot(D.T,D)
        
    if k == 4:
        data = np.array([np.ones(n), -4.0*np.ones(n), +6.0*np.ones(n), -4.0*np.ones(n), np.ones(n)])
        offsets = np.array([2,1,0,-1,-2])
        D_pre = dia_matrix((data, offsets), shape=(n, n))
        D     = scipy.sparse.csr_matrix(np.delete(np.delete(np.delete(np.delete(D_pre.todense(),0,0),0,0),-1,0),-1,0))
        C     = identity(n) +  param_Lagrange* np.dot(D.T,D)
        
    ## Initial estimates
    theta_hat = [X]
    w         = [np.zeros(n-k)]
    alpha     = [np.zeros(n-k)]
    
    ## ADMM part
    step = 0
    res_primal = 100
    res_dual   = 100
    while ( res_primal > 10**(-e_tolerance)) or ( res_dual > 10**(-e_tolerance)): 
        ret_1 =  X +  (param_Lagrange) *(D.T).dot(w[step] + alpha[step] )
        theta_hat.append( spsolve(C,ret_1.T) )

        ret_2 = D.dot(theta_hat[step+1]) - alpha[step]
        w.append( prox(param_regularization/(param_Lagrange), ret_2) )

        ret_3 = alpha[step] + ( w[step+1] - D.dot(theta_hat[step+1]) )
        alpha.append(ret_3)
        res_primal = np.sqrt(sum((theta_hat[step]-theta_hat[step-1])**2))
        res_dual   = np.sqrt(sum( (alpha[step]-alpha[step-1])**2 ))
        step = step + 1
        
    ## degrees of freedom
    accel_pre   = D.dot(theta_hat[-1])
    accel       = np.round(accel_pre,e_tolerance)
    knot_count = np.count_nonzero(np.abs(accel)>0)
    df         = knot_count + k
    
    return theta_hat[-1], df, accel

def calculate_Z(Master_X,Slave_X,k=2,param_regularization=0.25,param_Lagrange=1.,e_tolerance=5):

    ############################################################
    # trend filtering for master station
    size                                                             = len(Master_X)
    master_theta_hat, master_df, master_accel   = trend_filtering(Master_X,k=k,
                                                                                 param_regularization=param_regularization,
                                                                                 param_Lagrange=param_Lagrange,e_tolerance=e_tolerance)
    master_slope_hat                                         = np.diff(master_theta_hat)
    master_negativekink                                     = np.array([i for i in range(size-2)])[(master_accel)< 0.]

    master_time_positive                                    = np.array([i for i in range(size-1)])[(master_slope_hat > 0.)]
    
    ############################################################
    #master_time_seq
    #左に何個連続しているかを計算 (localを抽出するのに必要)
    change                                                        = (master_accel[1:] == master_accel[:-1])
    
    left                                                              = np.arange(len(change))
    left[change]                                                 = 0
    np.maximum.accumulate(left,out = left)
    
    right                                                            = np.arange(len(change))
    right[change[::-1]]                                        = 0
    np.maximum.accumulate(right,out = right)
    right                                                            = len(change) - right[::-1] -1
    
    master_time_seq                                          =  np.zeros_like(master_accel)
    master_time_seq[:-1]                                    += right
    master_time_seq[1:]                                     -=  left
    master_time_seq[-1]                                     =   0
    ############################################################
    
    ############################################################
    #global slope estimation
    master_global_slope  = np.mean(np.diff(Master_X[master_time_positive]))
    slave_global_slope     = np.mean(np.diff(Slave_X[master_time_positive]))
    
    master_global_sigma  = np.std(np.diff(Master_X[master_time_positive]))/np.sqrt(2) #差分の分散なので正規化が必要
    slave_global_sigma     = np.std(np.diff(Slave_X[master_time_positive]))/np.sqrt(2)
    
    master_global_slope_sd       =  master_global_sigma/ np.sqrt(len(master_time_positive) -1 )
    slave_global_slope_sd          =  slave_global_sigma/ np.sqrt(len(master_time_positive) -1)
    ############################################################
    
    ############################################################
    #local slope estimation
    master_local_slope    = [] 
    slave_local_slope       = []
    master_local_slope_sd   = []
    slave_local_slope_sd      = []
    for k in range(len(master_negativekink)):
        index_k             = master_negativekink[k]
        indices_interest = np.arange(index_k, index_k+ master_time_seq[index_k + 1]+1,dtype=int)
        M_slope = np.array([i+1 for i in indices_interest])
        M_const = np.array([1   for i in range(len(indices_interest))])
        M           = np.vstack((M_slope,M_const)).T
        master_local_slope.append( np.linalg.pinv(M).dot(Master_X[indices_interest])[0])
        slave_local_slope.append( np.linalg.pinv(M).dot(Slave_X[indices_interest])[0])
        
        M_Sigma = np.linalg.pinv(M).dot( np.linalg.pinv(M).T )
        master_local_slope_sd.append( np.sqrt( M_Sigma[0,0]* master_global_sigma**2) )
        slave_local_slope_sd.append( np.sqrt(M_Sigma[0,0]* slave_global_sigma**2) )
        
        
    master_normalized_Z_score =  [10000 if i not in master_negativekink 
                       else (master_local_slope-master_global_slope)[np.where(master_negativekink == i)[0][0]]
                                      / np.sqrt(  master_local_slope_sd[np.where(master_negativekink == i)[0][0]]**2 + master_global_slope_sd**2 )   for i in range(size)]
    slave_normalized_Z_score    =  [10000 if i not in master_negativekink 
                         else (slave_local_slope-slave_global_slope)[np.where(master_negativekink == i)[0][0]]
                                    /  np.sqrt(  slave_local_slope_sd[np.where(master_negativekink == i)[0][0]]**2 + slave_global_slope_sd**2 ) for i in range(size)]

    
    return master_global_slope, slave_global_slope, master_local_slope, slave_local_slope, master_normalized_Z_score, slave_normalized_Z_score, master_negativekink, master_global_sigma, slave_global_sigma



def proposed(Master_seq, Slave_seqs, param_Lagrange,param_regularization,e_tolerance=5):
    ##
    pval = []
    vafter_list = []
    size=len(Master_seq)
    ##
    accel = trend_filtering(Master_seq, k=2, param_Lagrange=param_Lagrange, param_regularization=param_regularization,e_tolerance=e_tolerance)[2]
    Activeset = np.array([i for i in range(size-2)])[np.abs(accel)>0.]
    accel_Activeset = accel[np.abs(accel)>0.]
    ##
    for s in range(len(Slave_seqs)):
        pval_ind = []
        vafter_ind = []
        Y = Slave_seqs[s]
        tf_Y = trend_filtering(Y,k=2,param_Lagrange=param_Lagrange,param_regularization=param_regularization,e_tolerance=e_tolerance)[0]
        for kinkpoint in [Activeset[0]]:
            pval_ind.append(1.)
            vafter_ind.append(-100.)
        for kinkpoint in Activeset[1:-1]:
            if np.sign(accel[kinkpoint]) > 0. :
                pval_ind.append(1.)
                vafter_ind.append(-100.)
            else:
                kinkpointindex = list(Activeset).index(kinkpoint)
                time_pre   = Activeset[kinkpointindex]-Activeset[kinkpointindex-1]
                time_post = Activeset[kinkpointindex+1]-Activeset[kinkpointindex]
                if time_pre < 3:
                    time_pre  = 3
                if time_post < 3.:
                    time_post = 3             
                Mc_1 = np.hstack((np.array([i+1 for i in range(time_pre)]), np.zeros(time_post)))
                Mc_2 = np.hstack(( np.zeros(time_pre),np.array([i+1 for i in range(time_post)])))
                Mc_3 = np.hstack((np.array([1 for i in range(time_pre)]), np.zeros(time_post)))
                Mc_4 = np.hstack(( np.zeros(time_pre),np.array([1 for i in range(time_post)])))
                M_slopetest  =  np.vstack((Mc_1,Mc_2,Mc_3,Mc_4)).T
                test_v = np.matrix([1,-1,0,0]).dot(np.linalg.pinv(M_slopetest))
                test = test_v.dot(Y[kinkpoint-time_pre:kinkpoint+time_post])
                if (20<kinkpoint)&(kinkpoint<size-20):
                    sigma = np.median( (np.abs(Y-tf_Y))[kinkpoint-20:kinkpoint+20] )
                else:
                    sigma = np.median( (np.abs(Y-tf_Y)))
                std   = np.sqrt(sum((np.array(test_v)[0,:])**2))*sigma
                Zscore = np.array(test).flatten()[0] / std
                TTG = norm.cdf(Zscore)
                pval_ind.append(1-TTG)
                
                getslope_vafter = np.matrix([0,1,0,0]).dot(np.linalg.pinv(M_slopetest))
                vafter   = getslope_vafter.dot(Y[kinkpoint-time_pre:kinkpoint+time_post])
                vafter_ind.append(vafter)
        for kinkpoint in [Activeset[-1]]:
            pval_ind.append(1.)
            vafter_ind.append(-100.)
        pval.append(pval_ind)
        vafter_list.append(vafter_ind)
    
    stack_confidence = np.zeros(len(pval[0]))
    for i in range(len(pval[0])):
        if max([pval[s][i] for s in range(len(Slave_seqs))])>0:
            stack_confidence[i] =  1-stat.harmonic_mean(np.array([pval[s][i] for s in range(len(Slave_seqs))]))
        else:
            stack_confidence[i] = 0.
    
    
    return pval, stack_confidence, Activeset, vafter_list


def outlierremove(Observation,threshold=2.):
    time_length=len(Observation)
    spline = spline_smoother(X=Observation,param_smooth=1000.)
    res    = np.abs(spline-Observation)
    estimate_sd = np.sqrt(sum(res**2)/time_length)
    modify_obs = np.zeros(time_length)
    for t in range(time_length):
        if (res[t] > threshold * estimate_sd):
                modify_obs[t] = spline[t]
        else:
                modify_obs[t] = Observation[t]
    return modify_obs


def cal_rho(lon_a,lat_a,lon_b,lat_b):
    ra=6378.140  # equatorial radius (km)
    rb=6356.755  # polar radius (km)
    F=(ra-rb)/ra # flattening of the earth
    rad_lat_a=np.radians(lat_a)
    rad_lon_a=np.radians(lon_a)
    rad_lat_b=np.radians(lat_b)
    rad_lon_b=np.radians(lon_b)
    pa=np.arctan(rb/ra*np.tan(rad_lat_a))
    pb=np.arctan(rb/ra*np.tan(rad_lat_b))
    xx=np.arccos(np.sin(pa)*np.sin(pb)+np.cos(pa)*np.cos(pb)*np.cos(rad_lon_a-rad_lon_b))
    c1=(np.sin(xx)-xx)*(np.sin(pa)+np.sin(pb))**2/np.cos(xx/2)**2
    c2=(np.sin(xx)+xx)*(np.sin(pa)-np.sin(pb))**2/np.sin(xx/2)**2
    dr=F/8*(c1-c2)
    rho=ra*(xx+dr)
    return rho