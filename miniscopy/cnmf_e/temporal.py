# -*- coding: utf-8 -*-
"""A set of routines for estimating the temporal components, given the spatial components and temporal components

"""
import scipy
import numpy as np
import platform
import sys
import scipy.sparse as spr
from scipy.sparse import spdiags, coo_matrix  # ,csgraph
import h5py
#%%
from IPython.core.debugger import Pdb

from .deconvolution import constrained_foopsi
from .utilities import update_order_greedy


def make_G_matrix(T, g):
    """
    create matrix of autoregression to enforce indicator dynamics

    Inputs:
    -----
    T: positive integer
        number of time-bins

    g: nd.array, vector p x 1
        Discrete time constants

    Output:
    ------
    G: sparse diagonal matrix
        Matrix of autoregression
    """
    if type(g) is np.ndarray:
        if len(g) == 1 and g < 0:
            g = 0
        gs = np.matrix(np.hstack((1, -(g[:]).T)))
        ones_ = np.matrix(np.ones((T, 1)))
        G = spdiags((ones_ * gs).T, list(range(0, -len(g) - 1, -1)), T, T)

        return G
    else:
        raise Exception('g must be an array')

def constrained_foopsi_parallel(Ytemp, nT, jj_, kwargs):
    """ necessary for parallel computation of the function  constrained_foopsi

        the most likely discretized spike train underlying a fluorescence trace
    """    
    keys = kwargs.keys()
    kwargs['bl'] = None if 'bl' not in keys else None
    kwargs['c1'] = None if 'c1' not in keys else None
    kwargs['g'] = None if 'g' not in keys else None
    kwargs['sn'] = None if 'sn' not in keys else None
    
    Ytemp= Ytemp.flatten()
    T = np.shape(Ytemp)[0]

    cc_, cb_, c1_, gn_, sn_, sp_, lam_ = constrained_foopsi(Ytemp, **kwargs)
    # cc_ the infered denoised fluorescence signal at each time-bin
    # cb_ Fluorescence baseline value
    # c1_ value of calcium at time 0
    # gn_ Parameters of the AR process that models the fluorescence impulse response
    # sn_ Standard deviation of the noise distribution
    # sp_ Discretized deconvolved neural activity (spikes)
    # lam_ Regularization parameter
    
    gd_ = np.max(np.real(np.roots(np.hstack((1, -gn_.T)))))
    gd_vec = gd_**list(range(T))
    C_ = cc_[:].T + cb_ + np.dot(c1_, gd_vec)
    Sp_ = sp_[:T].T
    Ytemp_ = Ytemp - C_.T

    return C_, Sp_, Ytemp_, cb_, c1_, sn_, gn_, jj_, lam_

def constrained_foopsi_helper(args): return constrained_foopsi_parallel(*args)

def update_iteration(C, YrA, AA, nA, parrllcomp, len_parrllcomp, ITER, **kwargs):

    """
        Update temporal components and background given spatial components using a block coordinate descent approach.
    """        
    duration = YrA.shape[0]    
    bl = np.zeros((YrA.shape[1]))*np.NaN
    c1 = np.zeros((YrA.shape[1]))*np.NaN
    sn = np.zeros((YrA.shape[1]))*np.NaN
    g = np.zeros((YrA.shape[1]))*np.NaN
    lam = np.zeros((YrA.shape[1]))*np.NaN
    if isinstance(C, h5py._hl.dataset.Dataset):
        C_new = np.zeros_like(C)
        chunk_size_col = C.chunks[0]
        for i in range(0, C.shape[0]+chunk_size_col,chunk_size_col):
            C_new[i:i+chunk_size_col,:] = C[i:i+chunk_size_col,:]    
    else:
        C_new = C.copy()

    for IT in range(ITER):
        for count, jo_ in enumerate(parrllcomp):            
            # INITIALIZE THE PARAMS
            jo = np.array(jo_).flatten()            
            Ytemp = YrA[:, jo] + C_new[:,jo]
            Ctemp = np.zeros((duration, np.size(jo)))
            Stemp = np.zeros((duration, np.size(jo)))            
            nT = nA[jo]            
            args_in = zip(Ytemp.T, nT, np.arange(len(jo)), [kwargs]*len(jo))
            # computing the most likely discretized spike train underlying a fluorescence trace
            # if 'multiprocessing' in str(type(dview)):
            #     results = dview.map_async(constrained_foopsi_parallel, args_in).get(4294967)
            # else:
            results = list(map(constrained_foopsi_helper, args_in))
            
            # unparsing and updating the result            
            for chunk in results:
                C_, Sp_, Ytemp_, cb_, c1_, sn_, gn_, jj_, lam_ = chunk
                Ctemp[:,jj_] = C_
                Stemp[:,jj_] = Sp_
                bl[jo[jj_]] = cb_
                c1[jo[jj_]] = c1_
                sn[jo[jj_]] = sn_
                g[jo[jj_]] = gn_.T if kwargs['p'] > 0 else []
                lam[jo[jj_]] = lam_

            
            # Ctemp = spr.csr_matrix(Ctemp)
            YrA -= (Ctemp - C[:,jo]).dot(AA[jo,:])
            C[:,jo] = Ctemp.copy()
            # S_old[:,jo] = Stemp.copy()
            

        # if dview is not None and not('multiprocessing' in str(type(dview))):
        #     dview.results.clear()   
        if isinstance(C, h5py._hl.dataset.Dataset):
            tmp = np.zeros_like(C)
            chunk_size_col = C.chunks[0]
            for i in range(0, C.shape[0]+chunk_size_col,chunk_size_col):
                tmp[i:i+chunk_size_col,:] = C_new[i:i+chunk_size_col,:] - C[i:i+chunk_size_col,:]
            tmp2 = scipy.linalg.norm(C.value, 'fro')
        else:
            tmp = C_new - C
            tmp2 = scipy.linalg.norm(C, 'fro')

        if scipy.linalg.norm(tmp, 'fro') <= 1e-3 * tmp2:
            #stopping: overall temporal component not changing significantly
            break
        else:
            C_new = C

    return C_new, bl, c1, sn, g, lam

def update_temporal_components(Y, A, C, **kwargs):
    """Update temporal components and background given spatial components using a block coordinate descent approach.

    """      
    # Pdb().set_trace()
    from time import time
    start = time()
    duration = Y.shape[0]
    nr = A.shape[0]            
    S = np.zeros_like(C)


    if isinstance(A, h5py._hl.dataset.Dataset) and isinstance(C, h5py._hl.dataset.Dataset):
        chunk_size_row = A.chunks[1]
        chunk_size_col = C.chunks[0]
        nA = np.zeros(nr)
        YdotAT = np.zeros((duration,nr))
        AdotAT = np.zeros((nr,nr))        
        for i in range(0,A.shape[1]+chunk_size_row,chunk_size_row):
            data = A[:,i:i+chunk_size_row]
            nA += np.power(data, 2.0).sum(1)
            AdotAT += data.dot(data.T)
            for j in range(0,duration+chunk_size_col,chunk_size_col):
                data2 = Y[j:j+chunk_size_col,i:i+chunk_size_row]                
                YdotAT[j:j+chunk_size_col,:] += data2.dot(data.T)                
            
        tmp = (1./nA)*np.eye(nr)    
        YA = YdotAT.dot(tmp)
        AA = AdotAT.dot(tmp)
        YrA = YA - C.value.dot(AA.T) # lazy
    else:    
        #Generating residuals    
        nA = np.power(A, 2.0).sum(1).flatten()
        tmp = (1./nA)*np.eye(nr)    
        YA = (Y.dot(A.T)).dot(tmp)

        AA = (A.dot(A.T)).dot(tmp)  # in row | AA is symetrical but not AA.T * tmp.
        YrA = YA - C.dot(AA.T) #here for instance need to transpose AA

    # creating the patch of components to be computed in parrallel
    parrllcomp, len_parrllcomp = update_order_greedy(spr.lil_matrix(AA))

    # entering the deconvolution    
    C, bl, c1, sn, g, lam = update_iteration(C, YrA, AA, nA, parrllcomp, len_parrllcomp, **kwargs)
    
    ff = np.where(np.sum(C, axis=0) == 0)  # remove empty components
    
    if np.size(ff) > 0:  # Eliminating empty temporal components
        ff = ff[0]
        keep = list(range(A.shape[0]))
        for i in ff:
            keep.remove(i)

        A = np.delete(A, list(ff), 0)
        C = np.delete(C, list(ff), 1)
        
        # YrA = np.delete(YrA, list(ff), 1)
        # S = np.delete(S, list(ff), 0)
        # sn = np.delete(sn, list(ff))
        # g = np.delete(g, list(ff))
        # bl = np.delete(bl, list(ff))
        # c1 = np.delete(c1, list(ff))
        # lam = np.delete(lam, list(ff))

        # background_ff = list(filter(lambda i: i > 0, ff - nr))
        # nr = nr - (len(ff) - len(background_ff))
    # print("Time to update temporal components ", time() - start)
    
    return C, A

