"""
Created on Wed Aug 05 20:38:27 2015

# -*- coding: utf-8 -*-
@author: agiovann
"""
import numpy as np
from scipy.ndimage import label, binary_dilation
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure
from scipy.ndimage.morphology import grey_dilation
import scipy.sparse as spr
from sklearn import linear_model
from scipy.ndimage.filters import median_filter
from scipy.ndimage.morphology import binary_closing
from tqdm import tqdm
import h5py

from IPython.core.debugger import Pdb


def circular_constraint(img_original):
    img = img_original.copy()
    nr, nc = img.shape
    [rsub, csub] = img.nonzero()
    if len(rsub) == 0:
        return img

    rmin = np.min(rsub)
    rmax = np.max(rsub) + 1
    cmin = np.min(csub)
    cmax = np.max(csub) + 1

    if (rmax - rmin <= 1) or (cmax - cmin <= 1):
        return img

    if rmin == 0 and rmax == nr and cmin == 0 and cmax == nc:
        ind_max = np.argmax(img)
        y0, x0 = np.unravel_index(ind_max, [nr, nc])
        vmax = img[y0, x0]
        x, y = np.meshgrid(np.arange(nc), np.arange(nr))
        fy, fx = np.gradient(img)
        ind = ((fx * (x0 - x) + fy * (y0 - y) < 0) & (img < vmax / 2))
        img[ind] = 0

        # # remove isolated pixels
        l, _ = label(img)
        ind = binary_dilation(l == l[y0, x0])
        img[~ind] = 0
    else:
        tmp_img = circular_constraint(img[rmin:rmax, cmin:cmax])
        img[rmin:rmax, cmin:cmax] = tmp_img

    return img

def basis_denoising(y, c, boh, sn, id2_, px):
    if np.size(c) > 0:
        _, _, a, _, _ = lars_regression_noise(y, c, 1, sn)
    else:
        return (None, None, None)
    return a, px, id2_

def determine_search_location(A, dims, method='ellipse', min_size=3, max_size=8, dist=3,expandCore=iterate_structure(generate_binary_structure(2, 1), 2).astype(int), **kwargs):
    """
    compute the indices of the distance from the cm to search for the spatial component

    does it by following an ellipse from the cm or doing a step by step dilatation around the cm
    Parameters:
    ----------
    [parsed]
     cm[i]:
        center of mass of each neuron

     A[:, i]: the A of each components

     dims:
        the dimension of each A's ( same usually )

     dist:
        computed distance matrix

     dims: [optional] tuple
                x, y[, z] movie dimensions

    method: [optional] string
            method used to expand the search for pixels 'ellipse' or 'dilate'

    expandCore: [optional]  scipy.ndimage.morphology
            if method is dilate this represents the kernel used for expansion

    min_size: [optional] int

    max_size: [optional] int

    dist: [optional] int

    dims: [optional] tuple
             x, y[, z] movie dimensions

    Returns:
    --------
    dist_indicator: np.ndarray
        distance from the cm to search for the spatial footprint

    Raise:
    -------
    Exception('You cannot pass empty (all zeros) components!')
    """
    # we initialize the values
    d1, d2 = dims    
    nr,d = np.shape(A)    
    dist_indicator = np.zeros((nr, d), dtype=bool)

    if method == 'ellipse':
        print("TODO")
        sys.exit()

        # Coor = dict()
        # # we create a matrix of size A.x of each pixel coordinate in A.y and inverse
        # if len(dims) == 2:
        #     Coor['x'] = np.kron(np.ones(d2), list(range(d1)))
        #     Coor['y'] = np.kron(list(range(d2)), np.ones(d1))
        # elif len(dims) == 3:
        #     Coor['x'] = np.kron(np.ones(d3 * d2), list(range(d1)))
        #     Coor['y'] = np.kron(
        #         np.kron(np.ones(d3), list(range(d2))), np.ones(d1))
        #     Coor['z'] = np.kron(list(range(d3)), np.ones(d2 * d1))
        # if not dist == np.inf:  # determine search area for each neuron
        #     cm = np.zeros((nr, len(dims)))  # vector for center of mass
        #     Vr = []  # cell(nr,1);
        #     dist_indicator = []
        #     pars = []
        #     # for each dim
        #     for i, c in enumerate(['x', 'y', 'z'][:len(dims)]):
        #         # mass center in this dim = (coor*A)/sum(A)
        #         cm[:, i] = old_div(
        #             np.dot(Coor[c], A[:, :nr].todense()), A[:, :nr].sum(axis=0))

        #     # parrallelizing process of the construct ellipse function
        #     for i in range(nr):
        #         pars.append([Coor, cm[i], A[:, i], Vr, dims,
        #                      dist, max_size, min_size, d])
        #     if dview is None:
        #         res = list(map(construct_ellipse_parallel, pars))
        #     else:
        #         if 'multiprocessing' in str(type(dview)):
        #             res = dview.map_async(
        #                 construct_ellipse_parallel, pars).get(4294967)
        #         else:
        #             res = dview.map_sync(construct_ellipse_parallel, pars)
        #     for r in res:
        #         dist_indicator.append(r)

        #     dist_indicator = (np.asarray(dist_indicator)).squeeze().T

        # else:
        #     dist_indicator = True * np.ones((d, nr))

    elif method == 'dilate':
        for i in range(nr):            
            A_temp = A[i].reshape(dims)
            if len(expandCore) > 0:
                A_temp = grey_dilation(A_temp, footprint=expandCore)

            dist_indicator[i] = A_temp.flatten() > 0            
    else:
        dist_indicator = True * np.ones((nr, d))

    return dist_indicator

def computing_indicator(A, C, dims, **kwargs):
    """compute the indices of the distance from the cm to search for the spatial component (calling determine_search_location)

    does it by following an ellipse from the cm or doing a step by step dilatation around the cm
    if it doesn't follow the rules it will throw an exception that is not supposed to be catch by spatial.
           """    
    duration, nr = C.shape  # number of neurons
    nb_pixels = A.shape[1]
    dist_indicator = determine_search_location(A, dims, **kwargs)
    dist_indicator = dist_indicator.T    
    #  bad
    ind2_ = []    
    I, J, V = spr.find(dist_indicator)
    for i in np.sort(I): 
        ind2_.append((i,J[I==i]))
            
    return ind2_

def regression(Y, A, C, sn, cct, idx_C, method_least_square, **kwargs):    
    """update spatial footprints and background through Basis Pursuit Denoising

       for each pixel i solve the problem
           [A(i,:),b(i)] = argmin sum(A(i,:))
       subject to
           || Y(i,:) - A(i,:)*C + b(i)*f || <= sn(i)*sqrt(T);

       for each pixel the search is limited to a few spatial components

       method_least_square:
           method to perform the regression for the basis pursuit denoising.
                'nnls_L0'. Nonnegative least square with L0 penalty
                'lasso_lars' lasso lars function from scikit learn
                'lasso_lars_old' lasso lars from old implementation, will be deprecated



       Returns:
       --------
       px: np.ndarray
            positions o the regression

       idxs_C: np.ndarray
           indices of the Calcium traces for each computed components

       a: learned weight

       Raises:
       -------
       Exception('Least Square Method not found!'
       """

    # /!\ need to import since it is run from within the server    
    # Pixel based    
    
    sn = sn.flatten()    
    duration,nr = C.shape    
    As = np.zeros(A.shape)    
    for px, neurons_in_pixel in idx_C:         
        c_in_pixel = C[:,neurons_in_pixel]
        y = Y[:,px]
        cct_ = cct[neurons_in_pixel]                        
        if method_least_square == 'nnls_L0':  # Nonnegative least square with L0 penalty
            print("TODO")
            import sys
            sys.exit()
            # a = nnls_L0(c.T, y, 1.2 * sn)

        elif method_least_square == 'lasso_lars':  # lasso lars function from scikit learn            
            lambda_lasso = .5 * sn[px] * np.sqrt(np.max(cct_)) / duration
            clf = linear_model.LassoLars(alpha=lambda_lasso, positive=True)
            a_lrs = clf.fit(c_in_pixel, y)
            coefs = a_lrs.coef_ # easier to debug

        else:
            raise Exception(
                'Least Square Method not found!' + method_least_square)        
        As[neurons_in_pixel,px] = coefs

    return As

def threshold_components_parallel(A_i, neuron, dims, medw, thr_method, maxthr, nrgthr, extract_cc, **kwargs):
    """
       Post-processing of spatial components which includes the following steps

       (i) Median filtering
       (ii) Thresholding
       (iii) Morphological closing of spatial support
       (iv) Extraction of largest connected component ( to remove small unconnected pixel )    
       """    
    if 'se' in kwargs.keys():
        se = kwargs['se']
    else:
        se = np.ones((3,) * len(dims), dtype='uint8')    

    ss = np.ones((3,) * len(dims), dtype='uint8')       

    # we reshape this one dimension column of the 2d components into the 2D that
    A_temp = np.reshape(A_i, dims)    
    # we apply a median filter of size medw
    A_temp = median_filter(A_temp, medw)
    
    if thr_method == 'max':        
        BW = (A_temp > maxthr * np.max(A_temp))
    elif thr_method == 'nrg':
        Asor = np.sort(A_temp.flatten())[::-1]
        temp = np.cumsum(Asor ** 2)
        ff = np.squeeze(np.where(temp < nrgthr * temp[-1]))
        if ff.size > 0:
            ind = ff if ff.ndim == 0 else ff[-1]
            A_temp[A_temp < Asor[ind]] = 0
            BW = (A_temp >= Asor[ind])
        else:
            BW = np.zeros_like(A_temp)
    # we want to remove the components that are valued 0 in this now 1d matrix    
    Ath = A_temp.flatten()
    Ath2 = np.zeros_like(Ath)
    # we do that to have a full closed structure even if the values have been trehsolded
    BW = binary_closing(BW.astype(np.int), structure=se)

    # if we have deleted the element
    if BW.max() == 0:
        return Ath2, neuron
    #
    # we want to extract the largest connected component ( to remove small unconnected pixel )
    if extract_cc:
        # we extract each future as independent with the cross structuring elemnt
        labeled_array, num_features = label(BW, structure=ss)
        labeled_array = labeled_array.flatten()        
        nrg = np.zeros((num_features, 1))
        # we extract the energy for each component
        for j in range(num_features):
            nrg[j] = np.sum(Ath[labeled_array == j + 1] ** 2)
        indm = np.argmax(nrg)
        Ath2[labeled_array == indm + 1] = Ath[labeled_array == indm + 1]

    else:
        BW = BW.flatten()
        Ath2[BW] = Ath[BW]

    return Ath2, neuron

def threshold_components_helper(args): return threshold_components_parallel(*args[:-1], **args[-1])

def threshold_components(A, dims, **kwargs):
    """
    Post-processing of spatial components which includes the following steps

    (i) Median filtering
    (ii) Thresholding
    (iii) Morphological closing of spatial support
    (iv) Extraction of largest connected component ( to remove small unconnected pixel )
    """
    se = np.ones((3,) * len(dims), dtype='uint8')    
    ss = np.ones((3,) * len(dims), dtype='uint8')
    # dims and nm of neurones
    nr = A.shape[0]
    # instanciation of A thresh.
    Ath = np.zeros(A.shape)
    pars = zip(A, np.arange(nr), [dims]*nr, [kwargs]*nr)    

    # if dview is not None:
    #     if 'multiprocessing' in str(type(dview)):
    #         res = dview.map_async(
    #             threshold_components_parallel, pars).get(4294967)
    #     else:
    #         res = dview.map_async(threshold_components_parallel, pars)
    # else:    
    res = list(map(threshold_components_helper, pars))

    for r in res:
        At, i = r
        Ath[i] = At
    
    return Ath

def HALS4shape_bckgrnd(patch, Y_resf, iters=5):
    K = patch.b.shape[0]    
    V = self.f.dot(self.f.T)        
    self.b = self.b.tolil()
    for _ in range(iters):
        self.b = self.b + ((Y_resf - V.dot(self.b)).multiply(spr.lil_matrix(1/V.diagonal()).T))
        negat = self.b < 0
        if negat.nnz:
            self.b[negat.nonzero()] = 0.0
        # for m in range(K):  # neurons
        #     ind = self.b[m].nonzero()[1]
        #     self.b[m,ind] = self.b[m,ind] + ((Y_resf[m,ind] - V[m].dot(self.b[:,ind])) / V[m, m])
        #     negat = self.b[m,ind]<0
        #     if negat.nnz:
        #         self.b[m,negat.nonzero()[1]] = 0.0
    return 

def update_spatial_components(Y, A, C, b, sn, normalize, update_background_components, low_rank_background, **kwargs):
    """update spatial footprints and background through Basis Pursuit Denoising 

    for each pixel i solve the problem
        [A(i,:),b(i)] = argmin sum(A(i,:))
    subject to
        || Y(i,:) - A(i,:)*C + b(i)*f || <= sn(i)*sqrt(T);

    for each pixel the search is limited to a few spatial components

    """    
    from time import time    
    
    start = time()
    dims = sn.shape
    duration, nr = C.shape    
    kwargs['expandCore'] = iterate_structure(generate_binary_structure(2, 1), 2).astype(int)
    # we compute the indicator from distance indicator    
    ind2 = computing_indicator(A, C, dims, **kwargs)
            
    if normalize and C is not None:                
        if isinstance(C, h5py._hl.dataset.Dataset):
            Csquare = np.zeros(nr)
            chunk_size_col = C.chunks[0]
            for i in range(0, duration+chunk_size_col,chunk_size_col):
                Csquare += np.power(C[i:i+chunk_size_col], 2.0).sum(0)
            Csquare = np.sqrt(Csquare)
            d_ = Csquare * np.eye(C.shape[1])
            chunk_size_row = A.chunks[1]
            # Ain = np.zeros_like(A)
            
            # for i in range(0, A.shape[1]+chunk_size_row,chunk_size_row): 
            #     Ain[:,i:i+chunk_size_row] = d_.dot(A[:,i:i+chunk_size_row])

            chunk_size_col = C.chunks[1]
            for i in range(0, nr+chunk_size_col, chunk_size_col):
                C[:,i:i+chunk_size_col] = C[:,i:i+chunk_size_col] / Csquare[i:i+chunk_size_col]

        else:    
            Csquare = np.sqrt(np.power(C, 2.0).sum(0))
            d_ = Csquare * np.eye(C.shape[1])
            # Ain = d_.dot(A) # THis is not used in CAIman after so not doing it here
            C = C/Csquare
    else:
        if isinstance(C, h5py._hl.dataset.Dataset):
            C = C.value.copy()
        else:
            C = C.copy()
    
    

    #Updating Spatial Components using lasso lars')
    if isinstance(C, h5py._hl.dataset.Dataset):
        CdotCT = np.zeros((nr,nr))
        chunk_size_row = C.chunks[0]
        for i in range(0, duration+chunk_size_row, chunk_size_row):            
            data = C[i:i+chunk_size_row,:]
            CdotCT += data.T.dot(data)
        CdotCT = CdotCT.diagonal()
    else:
        CdotCT = C.T.dot(C).diagonal()    

    
    Anew = regression(Y, A, C, sn, CdotCT, ind2, **kwargs)

    #thresholding components
    Anew = threshold_components(Anew, dims, **kwargs)
    ff = np.where(Anew.sum(1) == 0)[0] # remove empty components
    if len(ff):
        index = np.ones(Anew.shape[0], dtype = bool)
        index[ff] = False
        # updating A
        if isinstance(A, h5py._hl.dataset.Dataset):
            A.resize((index.sum(),A.shape[1]))
            A[:] = Anew[index]            
        else:
            A = Anew[index]
        # updating C
        if isinstance(C, h5py._hl.dataset.Dataset):
            # bad but should be small dimensions in number of neurons so should be ok
            Ctmp = C.value[:,index].copy()
            C.resize((C.shape[0],index.sum()))
            C[:] = Ctmp[:]
        else:
            C = C[:,index]
        nr = nr - len(ff)
    else:
        if isinstance(A, h5py._hl.dataset.Dataset):
            A[:] = Anew[:]
        else:
            A = Anew

        # if low_rank_background:
        #     background_ff = list(filter(lambda i: i >= nb, ff - nr))
        #     f = np.delete(f, background_ff, 0)
        # else:
        #     background_ff = list(filter(lambda i: i >= 0, ff - nr))
        #     f = np.delete(f, background_ff, 0)
        #     b_in = np.delete(b_in, background_ff, 1)
    
    if update_background_components:        
        if b is not None: #
            # Computing residuals                        
            self.f = self.f.T
            Y_resf = self.f.dot(Y) - (self.f.dot(self.C)).dot(self.A)
            HALS4shape_bckgrnd(patch, Y_resf)
    
    return A, C


