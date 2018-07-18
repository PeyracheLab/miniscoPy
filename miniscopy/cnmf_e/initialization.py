# -*- coding: utf-8 -*-
""" Initialize the component for the CNMF

contain a list of functions to initialize the neurons and the corresponding traces with
different set of methods like ICA PCA, greedy roi


"""
import os
# try:
#     os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
# finally:
#     del os.environ['OPENBLAS_NUM_THREADS']


import cv2
import scipy
from skimage.morphology import disk
import scipy.sparse as spr
import itertools
from tqdm import tqdm
from sklearn.decomposition import NMF
from scipy.ndimage.measurements import center_of_mass
import h5py
from IPython.core.debugger import Pdb

from .spatial import circular_constraint
from .deconvolution import deconvolve_ca
from .temporal import update_temporal_components
from .spatial import update_spatial_components

def get_noise_fft(Y, max_num_samples_fft=3072, noise_range=[0.25,0.5], noise_method='logmexp', **kwargs):
    """Estimate the noise level for each pixel by averaging the power spectral density.

    Inputs:
    -------
    noise_range: np.ndarray [2 x 1] between 0 and 0.5
        Range of frequencies compared to Nyquist rate over which the power spectrum is averaged
        default: [0.25,0.5]

    noise method: string
        method of averaging the noise.
        Choices:
            'mean': Mean
            'median': Median
            'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Output:
    ------
    sn: np.ndarray
        Noise level for each pixel
    """     
    duration    = len(Y)
    dims        = Y.shape[1:]    
    if duration > max_num_samples_fft:
        # take first part, middle and end part
        idx_frames = np.concatenate((np.arange(1,max_num_samples_fft // 3 + 1),
                                    np.arange(duration//2 - max_num_samples_fft //6,duration//2 + max_num_samples_fft //6),
                                    np.arange(duration-max_num_samples_fft//3,duration)))
        duration = max_num_samples_fft

    else:
        idx_frames = np.arange(duration)

    ff          = np.arange(0, 0.5 + 1. / duration, 1. / duration)
    ind1        = ff > noise_range[0]
    ind2        = ff <= noise_range[1]
    ind         = np.logical_and(ind1, ind2)
    

    if type(Y) is h5py._hl.dataset.Dataset:
        psdx        = np.zeros((dims[0],dims[1],ind.sum()))
        for i in tqdm(range(dims[0])): # doing it row by row            
            data = Y[:,i,:]
            sample = data[idx_frames]
            dft = np.fft.fft(sample, axis = 0)[:len(ind)][ind].T
            psdx[i]   = (1./duration) * (dft.real * dft.real)+(dft.imag*dft.imag)
        psdx = psdx.reshape(np.prod(dims),ind.sum())

    elif type(Y) is np.ndarray:
        dft         = np.fft.fft(Y[idx_frames].reshape(duration,np.prod(dims)), axis = 0)[:len(ind)][ind].T        
        psdx        = (1./duration) * (dft.real * dft.real)+(dft.imag*dft.imag)

    if noise_method == 'mean':
        noise = np.sqrt(np.mean(psdx, axis=1))
    elif noise_method == 'median':
        noise = np.sqrt(np.median(psdx, axis=1))
    else:
        noise = np.log(psdx + 1e-10)
        noise = np.mean(noise, axis=1)
        noise = np.exp(noise)
        noise = np.sqrt(noise)

    return noise.reshape(dims)


def compute_W(patch, Y, A, C, CdotA, radius, dims):
    """compute background according to ring model
    solves the problem
        min_{W,b0} ||X-W*X|| with X = Y - A*C - b0*1'
    subject to
        W(i,j) = 0 for each pixel j that is not in ring around pixel i
    Problem parallelizes over pixels i
    Fluctuating background activity is W*X, constant baselines b0.
    Parameters:
    ----------
    Y: np.ndarray (2D or 3D)
        movie, raw data in 2D or 3D (pixels x time).
    A: np.ndarray or sparse matrix
        spatial footprint of each neuron.
    C: np.ndarray
        calcium activity of each neuron.
    dims: tuple
        x, y[, z] movie dimensions
    radius: int
        radius of ring
    data_fits_in_memory: [optional] bool
        If true, use faster but more memory consuming computation

    Returns:
    --------
    W: scipy.sparse.csr_matrix (pixels x pixels)
        estimate of weight matrix for fluctuating background
    b0: np.ndarray (pixels,)
        estimate of constant background baselines
    """
    from time import time   
    start = time() 
    ring = disk(radius + 1, dtype=bool)
    ring[1:-1, 1:-1] = np.bitwise_xor(ring[1:-1, 1:-1], disk(radius, dtype=bool))        
    ringidx = np.vstack(np.where(ring)).T - radius - 1    
    
    if isinstance(A, np.ndarray) and isinstance(C, np.ndarray):
        tmp = C.mean(0).dot(A)
    elif isinstance(A, h5py._hl.dataset.Dataset) and isinstance(C, h5py._hl.dataset.Dataset):
        print("TODO")
        sys.exit()
        pass

    tmp2 = np.zeros_like(tmp)
    chunk_size = Y.chunks[0]
    for i in range(0, Y.shape[0]+chunk_size,chunk_size): tmp2 += Y[i:i+chunk_size,:].sum(0)
    tmp2 = tmp2/float(Y.shape[0])

    # constante fluorescence is mean(Y) - A * mean(C)
    b0 = tmp2 - tmp
    
    if isinstance(CdotA, np.ndarray):
        X = (Y - CdotA - b0).T # (pixels, time)
    elif isinstance(CdotA, h5py._hl.dataset.Dataset):
        print("TODO")
        sys.exit()
        pass

    W = np.zeros((np.prod(dims),np.prod(dims)))

    pixels_pos  = np.array([x for x in itertools.product(range(dims[0]), range(dims[1]))])                 
    xpos_ring   = np.vstack(pixels_pos[:,0])+ringidx[:,0]                                                  
    ypos_ring   = np.vstack(pixels_pos[:,1])+ringidx[:,1]
    xypos_ring  = np.dstack((xpos_ring, ypos_ring))
    inside      = (xpos_ring>=0) * (ypos_ring>=0) * (xpos_ring<dims[0]) * (ypos_ring<dims[1])
    
    for i,(r,c) in enumerate(itertools.product(range(dims[0]), range(dims[1]))):
        # xy = np.array([r,c]) + ringidx                
        # inside = np.prod((xy >= 0) * (xy < dims), 1)        
        # index = xy[inside==1]                
        index = xypos_ring[i, inside[i]==1]
        flatten_index = np.ravel_multi_index(index.T, dims)                
        B = X[flatten_index]                
        tmp0 = B.dot(B.T)        
        tmp0 = tmp0 + 1.e-9*np.eye(index.shape[0])                
        tmp = np.linalg.inv(tmp0)        
        tmp2 = X[i].dot(B.T)
        tmp3 = tmp2.dot(tmp)
        W[i,flatten_index] = tmp3
        end = time()
    
    # print("Time to update weight ", time() - start)
    return W, b0

def extract_ac(data_filtered_box_, data_raw_box, ind_ctr, patch_dims, filter_data_centering=True):
    # parameters
    min_corr_neuron = 0.7
    max_corr_bg = 0.3    
    data_filtered_box = data_filtered_box_.copy()

    # compute the temporal correlation between each pixel and the seed pixel        
    # normalization | new from the original cnmfe
    if filter_data_centering:        
        data_filtered_box -= data_filtered_box.mean(axis=0)  # data centering
        tmp = np.sqrt(np.sum(data_filtered_box ** 2, axis=0))  # data
        tmp[tmp == 0] = 1
        data_filtered_box /= tmp
        y0 = data_filtered_box[:, ind_ctr]  # fluorescence trace at the center
        tmp_corr = np.dot(y0.reshape(1, -1), data_filtered_box)  # corr. coeff. with y0        
    else:
        # here I do a centring of the data 
        # but just to compute the correlation coefficient
        tmp = data_filtered_box - data_filtered_box.mean(axis = 0)
        tmp2 = np.sqrt(np.sum(tmp ** 2, axis=0))  # data
        tmp2[tmp2 == 0] = 1
        tmp /= tmp2
        y0 = tmp[:, ind_ctr]  # fluorescence trace at the center
        tmp_corr = np.dot(y0.reshape(1, -1), tmp)  # corr. coeff. with y0



    # pixels in the central area of neuron
    ind_neuron = (tmp_corr > min_corr_neuron).squeeze()
    # pixels outside of neuron's ROI
    ind_bg = (tmp_corr < max_corr_bg).squeeze()

    # extract temporal activity
    ci = np.mean(data_filtered_box[:, ind_neuron], axis=1).reshape(-1, 1)
    # initialize temporal activity of the neural
    if filter_data_centering:
        ci -= np.median(ci)

    if np.linalg.norm(ci) == 0:  # avoid empty results
        return None, None, False

    # roughly estimate the background fluctuation
    y_bg = np.median(data_raw_box[:, ind_bg], axis=1).reshape(-1, 1)
    # extract spatial components
    # pdb.set_trace()
    X = np.hstack([ci - ci.mean(), y_bg - y_bg.mean(), np.ones(ci.shape)])
    XX = np.dot(X.transpose(), X)
    Xy = np.dot(X.T, data_raw_box)

    ai = scipy.linalg.lstsq(XX, Xy)[0][0]
    ai = ai.reshape(patch_dims)
    ai[ai < 0] = 0

    # post-process neuron shape
    ai = circular_constraint(ai)

    # return results
    return ai, ci.reshape(len(ci)), True

def local_correlations_fft(data, **kwargs):
    """Computes the correlation image for the input dataset Y using a faster FFT based method
    """
    if isinstance(data, np.ndarray):
        data = data.astype('float32')
        data -= data.mean(0)
        datastd = data.std(0)
        datastd[datastd == 0] = np.inf
        data /= datastd
        sz = np.ones((3, 3), dtype='float32')
        sz[1, 1] = 0

        dataconv = np.zeros_like(data)

        for i, frame in enumerate(data):
            dataconv[i] = cv2.filter2D(frame, -1, sz, borderType=0)
        MASK = cv2.filter2D(np.ones(data.shape[1:], dtype = 'float32'), -1, sz, borderType=0)
        return np.mean(dataconv*data, axis = 0) / MASK

    elif isinstance(data, h5py._hl.dataset.Dataset):
        new_data = data.parent.create_dataset('tmp', shape = data.shape, chunks = True)
        dataconv = data.parent.create_dataset('dataconv', shape = data.shape, chunks = True)
        data_mean = np.mean(data, 0)
        data_std = np.std(data, 0)
        data_std[data_std == 0] = np.inf
        chunk_size = data.chunks[0]
        sz = np.ones((3, 3), dtype='float32')
        sz[1, 1] = 0        
        corr = np.zeros(data.shape[1:])        
        for i in tqdm(range(0, data.shape[0], chunk_size)):
            tmp = data[i:i+chunk_size] - data_mean            
            tmp = tmp / data_std
            new_data[i:i+chunk_size,:] = tmp
            for j, frame in enumerate(tmp):
                dataconv[i+j] = cv2.filter2D(frame, -1, sz, borderType =0)
        corr += np.sum(new_data[i:i+chunk_size,:,:] * dataconv[i:i+chunk_size,:,:], axis = 0)

        corr /= float(data.shape[0])

        MASK = cv2.filter2D(np.ones(data.shape[1:], dtype = 'float32'), -1, sz, borderType=0)
        
        del data.parent['tmp']
        del data.parent['dataconv']

        return corr/MASK


    
def init_neurons_corr_pnr(Yc, dims, gSig, gSiz, thresh_init, min_corr, min_pnr, bd, min_pixel, center_psf, filter_data_centering, **kwargs):
    """
    using greedy method to initialize neurons by selecting pixels with large
    local correlation and large peak-to-noise ratio
    
    """    
    from time import time
    start = time()
    deconvolve_options = {'bl': None,
                          'c1': None,
                          'g': None,
                          'sn': None,
                          'p': 1,
                          'approach': 'constrained foopsi',
                          'method': 'oasis',
                          'bas_nonneg': True,
                          'noise_range': [.25, .5],
                          'noise_method': 'logmexp',
                          'lags': 5,
                          'fudge_factor': 1.0,
                          'verbosity': None,
                          'solvers': None,
                          'optimize_g': 1,
                          'penalty': 1}

    # Pdb().set_trace()

    duration = Yc.shape[0]    
    data_filtered = np.zeros((duration, dims[0], dims[1]))
    data_raw = np.zeros((duration, np.prod(dims)))

    # spatially filter data
    ksize = tuple((3*gSig) // 2 * 2 + 1)    
    chunk_size  = Yc.chunks[0]
    
    for i in range(0, duration+chunk_size, chunk_size):
        data = Yc[i:i+chunk_size,:]
        data_raw[i:i+chunk_size,:] = data.copy()
        for j, frame in enumerate(data):        
            if center_psf:
                tmp = cv2.GaussianBlur(frame.reshape(dims), ksize = ksize, sigmaX = gSig[0], sigmaY = gSig[1], borderType=1)
                tmp2 = cv2.boxFilter(frame.reshape(dims), ddepth=-1, ksize = ksize, borderType = 1)
                data_filtered[i+j] = tmp - tmp2
            else:
                tmp = cv2.GaussianBlur(frame.reshape(dims), ksize = ksize, sigmaX = gSig[0], sigmaY = gSig[1], borderType=1)
                data_filtered[i+j] = tmp
    
    # Pdb().set_trace()

    # compute peak-to-noise ratio    
    if filter_data_centering:
        data_filtered -= data_filtered.mean(axis=0)    
    data_max = np.max(data_filtered, axis=0)    
    noise_pixel = get_noise_fft(data_filtered)
    pnr = np.divide(data_max, noise_pixel)

    # remove small values and only keep pixels with large fluorescence signals
    tmp_data = np.copy(data_filtered)
    tmp_data[tmp_data < thresh_init * noise_pixel] = 0

    # compute correlation image
    cn = local_correlations_fft(tmp_data)
    del(tmp_data)
    if np.isnan(cn).sum(): print("nan pixels in cn")
    
    # screen seed pixels as neuron centers
    v_search = cn * pnr
    v_search[(cn < min_corr) | (pnr < min_pnr)] = 0
    ind_search = (v_search <= 0)  # indicate whether the pixel has
    # been searched before. pixels with low correlations or low PNRs are
    # ignored directly. ind_search[i]=0 means the i-th pixel is still under
    # consideration of being a seed pixel

    # pixels near the boundaries are ignored because of artifacts
    ind_bd = np.zeros(dims).astype(np.bool)  # indicate boundary pixels
    if bd > 0:
        ind_bd[:bd, :] = True
        ind_bd[-bd:, :] = True
        ind_bd[:, :bd] = True
        ind_bd[:, -bd:] = True
    ind_search[ind_bd] = True

    # creating variables for storing the results    
    max_number = np.int32(np.ceil(((ind_search.size - ind_search.sum())/3))) # changed from original cnmfe
    if max_number == 0: max_number = 1 # if there is just one pixel, it's worth looking at it
    Ain = np.zeros(shape=(max_number, dims[0], dims[1]),dtype=np.float32)  # neuron shapes / spatial footprint
    Cin = np.zeros(shape=(max_number, duration),dtype=np.float32)  # de-noised traces
    Sin = np.zeros(shape=(max_number, duration),dtype=np.float32)  # spiking # activity
    Cin_raw = np.zeros(shape=(max_number, duration),dtype=np.float32)  # raw traces
    center = np.zeros(shape=(max_number,2), dtype= np.int)  # neuron centers

    num_neurons = 0  # number of initialized neurons
    continue_searching = True
    min_v_search = min_corr * min_pnr
    count = 0
    
    while continue_searching:
        # local maximum, for identifying seed pixels in following steps
        v_search[(cn < min_corr) | (pnr < min_pnr)] = 0
        v_search[ind_search] = 0
        tmp_kernel = np.ones((gSiz // 3) * 2)
        v_max = cv2.dilate(v_search, tmp_kernel)

        # automatically select seed pixels as the local maximums
        v_max[(v_search != v_max) | (v_search < min_v_search)] = 0
        v_max[ind_search] = 0
        [rsub_max, csub_max] = v_max.nonzero()  # subscript of seed pixels
        local_max = v_max[rsub_max, csub_max]
        n_seeds = len(local_max)  # number of candidates
        if n_seeds == 0:
            # no more candidates for seed pixels            
            break
        else:
            # order seed pixels according to their corr * pnr values
            ind_local_max = local_max.argsort()[::-1]
        img_vmax = np.median(local_max)

        # try to initialize neurons given all seed pixels
        for ith_seed, idx in enumerate(ind_local_max):
            r = rsub_max[idx]
            c = csub_max[idx]
            ind_search[r, c] = True  # this pixel won't be searched
            if v_search[r, c] < min_v_search:
                # skip this pixel if it's not sufficient for being a seed pixel
                continue

            # roughly check whether this is a good seed pixel            
            y0 = data_filtered[:, r,c]
            if np.max(y0) < thresh_init * noise_pixel[r, c]:
                v_search[r, c] = 0
                continue

            # does it correlate with another neuron                        
            if Ain[:, r, c].sum() > 0:
                testp = Cin_raw[Ain[:, r, c] > 0]
                rr = [scipy.stats.pearsonr(y0, cc)[0] for cc in testp]
                if np.max(rr) > .7:
                    v_search[r, c] = 0
                    continue

            # new heuristic from Caiman
            y0_diff = np.diff(y0)
            if y0_diff.max() < 3*y0_diff.std():
                v_search[r,c] = 0
                continue


            # crop a small box for estimation of ai and ci
            r_min = max(0, r - gSiz[0])
            r_max = min(dims[0], r + gSiz[0] + 1)
            c_min = max(0, c - gSiz[1])
            c_max = min(dims[1], c + gSiz[1] + 1)
            nr = r_max - r_min
            nc = c_max - c_min
            patch_dims = (nr, nc)  # patch dimension

            index_box = np.zeros(dims)
            index_box[r_min:r_max,c_min:c_max] = True
            index_box_1d = np.where(index_box.flatten())[0]
            data_raw_box = data_raw[:,index_box_1d]
            data_filtered_box = data_filtered[:, r_min:r_max, c_min:c_max].reshape(-1, nr * nc)
            # index of the seed pixel in the cropped box
            ind_ctr = np.ravel_multi_index((r - r_min, c - c_min),dims=(nr, nc))

            # neighbouring pixels to update after initializing one neuron
            r2_min = max(0, r - 2 * gSiz[0])
            r2_max = min(dims[0], r + 2 * gSiz[0] + 1)
            c2_min = max(0, c - 2 * gSiz[1])
            c2_max = min(dims[1], c + 2 * gSiz[1] + 1)
            try:                        
                ai, ci_raw, ind_success = extract_ac(data_filtered_box,data_raw_box, ind_ctr, patch_dims, filter_data_centering)
            except ValueError:
                Pdb().set_trace()

            if (np.sum(ai > 0) < min_pixel) or (not ind_success):
                # bad initialization. discard and continue
                continue
            else:
                # cheers! good initialization.
                center[num_neurons] = [r, c]
                Ain[num_neurons, r_min:r_max, c_min:c_max] = ai
                Cin_raw[num_neurons] = ci_raw.squeeze()
                if deconvolve_options:
                    # deconvolution                          
                    # if count == 1:
                    #     Pdb().set_trace()
                    # count += 1                     
                    ci, si, tmp_options, baseline, c1 = deconvolve_ca(ci_raw, deconvolve_options.copy())                    
                    
                    Cin[num_neurons] = ci
                    Sin[num_neurons] = si
                else:
                    # no deconvolution
                    baseline = np.median(ci_raw)
                    ci_raw -= baseline
                    ci = ci_raw.copy()
                    ci[ci < 0] = 0
                    Cin[num_neurons] = ci.squeeze()

                # remove the spatial-temporal activity of the initialized
                # and update correlation image & PNR image
                # update the raw data                
                ac = ci[:,np.newaxis].dot(ai.flatten()[np.newaxis,:])
                data_raw[:,index_box_1d] -= ac
                
                # spatially filtered the neuron shape
                tmp_img = Ain[num_neurons, r2_min:r2_max, c2_min:c2_max]
                if center_psf:
                    ai_filtered = cv2.GaussianBlur(tmp_img, ksize=ksize,sigmaX=gSig[0], sigmaY=gSig[1],borderType=cv2.BORDER_REFLECT) \
                        - cv2.boxFilter(tmp_img, ddepth=-1,ksize=ksize, borderType=cv2.BORDER_REFLECT)
                else:
                    ai_filtered = cv2.GaussianBlur(tmp_img, ksize=ksize,sigmaX=gSig[0], sigmaY=gSig[1],borderType=cv2.BORDER_REFLECT)
                
                # update the filtered data
                data_filtered[:, r2_min:r2_max, c2_min:c2_max] -= ai_filtered[np.newaxis, ...] * ci[..., np.newaxis, np.newaxis]
                data_filtered_box = data_filtered[:,r2_min:r2_max, c2_min:c2_max].copy()

                # update PNR image
                if filter_data_centering:
                    data_filtered_box -= data_filtered_box.mean(axis=0)
                max_box = np.max(data_filtered_box, axis=0)
                noise_box = noise_pixel[r2_min:r2_max, c2_min:c2_max]
                pnr_box = np.divide(max_box, noise_box)
                pnr[r2_min:r2_max, c2_min:c2_max] = pnr_box
                pnr_box[pnr_box < min_pnr] = 0

                # update correlation image
                data_filtered_box[data_filtered_box < thresh_init * noise_box] = 0
                cn_box = local_correlations_fft(data_filtered_box)
                cn_box[np.isnan(cn_box) | (cn_box < 0)] = 0
                cn[r_min:r_max, c_min:c_max] = cn_box[(r_min - r2_min):(r_max - r2_min), (c_min - c2_min):(c_max - c2_min)]
                cn_box = cn[r2_min:r2_max, c2_min:c2_max]
                cn_box[cn_box < min_corr] = 0

                # update v_search
                v_search[r2_min:r2_max, c2_min:c2_max] = cn_box * pnr_box
                # avoid searching nearby pixels
                v_search[r_min:r_max, c_min:c_max] *= (ai < np.max(ai) / 2.)

                # increase the number of detected neurons
                num_neurons += 1  #
                if num_neurons == max_number:
                    continue_searching = False
                    break


    # print('In total, ', num_neurons, 'neurons were initialized.')
    
    A = Ain[:num_neurons]
    C = Cin[:num_neurons]
    C_raw = Cin_raw[:num_neurons]
    S = Sin[:num_neurons]
    center = center[:num_neurons]

    # print("Time in init_neurons_corr_pnr ", time() -start)
    return A, C, C_raw, S, center

def greedyROI_corr(patch, Yc, tsub, ssub, ring_size_factor, **kwargs):
    """
    initialize neurons based on pixels' local correlations and peak-to-noise ratios.
    """    
    from time import time    
    start = time()
    new_dims = tuple(Yc.attrs['dims'])
    # Pdb().set_trace()
    # Yc is not modified inside this function | it is not either in original cnmfe or it's a bug
    A, C, _, _, center = init_neurons_corr_pnr(Yc, new_dims, **kwargs)        
    
    # to ease dot product between A and C, A is reshaped in (neurons, pixel) and C is (time, neurons)
    duration = Yc.shape[0]
    A = A.reshape(A.shape[0], np.prod(new_dims))
    C = C.transpose()
    S = np.zeros_like(C.shape)
    CdotA = C.dot(A)    

    B = Yc - CdotA
    sn = cv2.resize(patch.sn, new_dims[::-1], interpolation = cv2.INTER_AREA)
    nr = C.shape[1]
    
    if ring_size_factor is not None:        
        # background according to ringmodel                                
        W, b0 = compute_W(patch, Yc, A, C, CdotA, np.round(kwargs['gSiz'][0]*ring_size_factor).astype('int'), new_dims)
                
        tmp = B - b0   # Local background minus constante background
        B = Yc - (b0 + tmp.dot(W.T)) # ring model of local background fluorescence

        if nr:
            # 1st iteration on decimated data
            # update temporal
            C = update_temporal_components(B.copy(), A, C, **patch.parameters['temporal_params'])

            # update spatial     
            A, C = update_spatial_components(B.copy(), A, C, None, sn, **patch.parameters['spatial_params'])
            
            # background according to ringmodel
            W, b0 = compute_W(patch, Yc, A, C, CdotA, np.round(kwargs['gSiz'][0]*ring_size_factor).astype('int'), new_dims)

        
        # 2nd iteration on non-decimated data          
        # original size in time first
        nr = C.shape[1]
        if nr:
            patch.C = patch.patch_group.create_dataset('C', shape = (patch.duration,nr), chunks = (patch.chunks[0],1))
            if tsub > 1:
                index = np.arange(duration).repeat(tsub)
                if len(index) <= patch.duration:
                    patch.C[0:len(index)] = C[index]
                    patch.C[-1] = C[-1] # tripling the last element
                else:
                    patch.C[:] = C[index][0:patch.duration]            
            elif tsub == 1:
                patch.C[:] = C[:]
        else:
            patch.C = patch.patch_group.create_dataset('C', shape = (patch.duration, nr))

        tmp = patch.C.value.dot(A)

        if ssub == 1:
            B = patch.patch_group['Y'] - tmp
        else: # downsample movie in space only
            Ys = np.zeros(tmp.shape)
            B = np.zeros(tmp.shape)
            chunk_size = patch.patch_group['Y'].chunks[0]
            for i in range(0, patch.duration+chunk_size,chunk_size):
                data = patch.patch_group['Y'][i:i+chunk_size]
                for j, frame in enumerate(data):
                    Ys[i+j] = cv2.resize(frame.reshape(patch.dims), new_dims[::-1], interpolation = cv2.INTER_AREA).flatten()                    
                    B[i+j] = Ys[i+j] - tmp[i+j]


        # nr = C.shape[1]        
        # if tsub > 1:              
        #     if nr: # when doubling C matrix, it can causes a bug if odd size
        #         patch.C = patch.patch_group.create_dataset('C', shape = (patch.duration,nr), chunks = (patch.chunks[0],1))
        #         index = np.arange(duration).repeat(tsub)
        #         if len(index) <= patch.duration:
        #             patch.C[0:len(index)] = C[index]
        #             patch.C[-1] = C[-1] # tripling the last element
        #         else:
        #             patch.C[:] = C[index][0:patch.duration]
        #     else:
        #         patch.C = patch.patch_group.create_dataset('C', shape = (patch.duration,nr))
                
        #     tmp = patch.C.value.dot(A)
        #     if ssub == 1:                
        #         B = patch.patch_group['Y'] - tmp
        #     else: # downsample movie in space only                        
        #         Ys = np.zeros(tmp.shape)
        #         B = np.zeros(tmp.shape)
        #         chunk_size = patch.patch_group['Y'].chunks[0]                
        #         for i in range(0, patch.duration+chunk_size,chunk_size):
        #             data = patch.patch_group['Y'][i:i+chunk_size]                    
        #             for j, frame in enumerate(data):
        #                 Ys[i+j] = cv2.resize(frame.reshape(patch.dims), new_dims[::-1], interpolation = cv2.INTER_AREA).flatten()                    
        #                 B[i+j] = Ys[i+j] - tmp[i+j]
        #     # N.B: upsampling B in space is fine, but upsampling in time doesn't work well,
        #     # cause the error in upsampled background can be of similar size as neural signal            
        # else:        
        #     B = patch.patch_group['Y'] - C.dot(A) # TO CHECK HERE
        
        tmp = B - b0  #(Y - AC - b0)
        YplusB = b0 + tmp.dot(W.T)

        # original size in space
        if ssub > 1:                                    
            patch.B = patch.patch_group.create_dataset('B', shape = (patch.duration, np.prod(patch.dims)), chunks = patch.chunks)
            chunk_size = patch.B.chunks[0]
            # resize YplusB in original size
            for i in range(0, patch.duration+chunk_size,chunk_size):
                data = np.zeros((chunk_size,np.prod(patch.dims)))                
                stop = 0
                for j, frame in enumerate(YplusB[i:i+chunk_size]):
                    data[j] = cv2.resize(frame.reshape(new_dims), patch.dims[::-1], interpolation = cv2.INTER_NEAREST).flatten()
                    stop = j+1                                
                patch.B[i:i+stop] = data[0:stop,:]
            if nr:
                patch.A = patch.patch_group.create_dataset('A', shape = (nr, np.prod(patch.dims)), chunks = (1,patch.chunks[1]))            
            else:
                patch.A = patch.patch_group.create_dataset('A', shape = (nr, np.prod(patch.dims)))            
            # resize A in original size                
            for n, frame in enumerate(A):
                patch.A[n] = cv2.resize(frame.reshape(new_dims), patch.dims[::-1], interpolation = cv2.INTER_NEAREST).flatten()[:]
               
        # time to load the full movie           
        # summing by pixels here
        chunk_size = patch.B.chunks[1]
        for i in range(0, np.prod(patch.dims)+chunk_size,chunk_size):            
            patch.B[:,i:i+chunk_size] = patch.patch_group['Y'][:,i:i+chunk_size] - patch.B[:,i:i+chunk_size]
                
        if nr:
            # # Update temporal                    
            patch.C[...] = update_temporal_components(patch.B, patch.A, patch.C, **patch.parameters['temporal_params'])   

            # # Update spatial
            patch.A[...], patch.C[...] = update_spatial_components(patch.B, patch.A, patch.C, None, patch.sn, **patch.parameters['spatial_params'])
            
        nr = patch.A.shape[0]
        # normalize
        nA = np.zeros(nr)
        if nr:
            chunk_size_col = patch.A.chunks[1]
            for i in range(0, patch.A.shape[1]+chunk_size_col, chunk_size_col):
                nA += np.power(patch.A[:,i:i+chunk_size_col], 2.0).sum(1).flatten()
            nA = np.sqrt(nA)
        
            chunk_size_row = patch.A.chunks[0]
            for i in range(0, nr+chunk_size_row, chunk_size_row):
                patch.A[i:i+chunk_size_row,:] = patch.A[i:i+chunk_size_row,:] / nA[i:i+chunk_size_row][:,np.newaxis]
            
            chunk_size_row = patch.C.chunks[0]
            for i in range(0, patch.C.shape[0]+chunk_size_row, chunk_size_row):
                patch.C[i:i+chunk_size_row,:] = patch.C[i:i+chunk_size_row,:] * nA
        

        # not sure why it need to be downscale again in space        
        A = np.zeros((nr, np.prod(new_dims)))
        if nr:
            chunk_size_row = patch.A.chunks[0]        
            for i in range(0, nr+chunk_size_row, chunk_size_row):
                data = patch.A[i:i+chunk_size_row,:]
                for j, frame in enumerate(data):
                    A[i+j] = cv2.resize(frame.reshape(patch.dims), new_dims[::-1], interpolation = cv2.INTER_AREA).flatten()

        # and in time
        if patch.C.shape[1]:
            C = cv2.resize(patch.C.value, (patch.C.shape[1], patch.duration//tsub), fx = 1., fy = 1./tsub, interpolation = cv2.INTER_AREA)
        else:
            C = np.zeros((patch.duration//tsub,0))
        CdotA = C.dot(A)

        # background according to ringmodel        
        W, b0 = compute_W(patch, Yc, A, C, CdotA, np.round(kwargs['gSiz'][0]*ring_size_factor).astype('int'), new_dims)
        
        # final update           
        if nr:
            tmp = np.zeros((patch.duration,np.prod(new_dims)))
            chunk_size_row = patch.C.chunks[0]        
            for i in range(0, patch.duration+chunk_size_row, chunk_size_row):            
                tmp[i:i+chunk_size_row,:] = (Ys[i:i+chunk_size_row,:] - patch.C[i:i+chunk_size_row,:].dot(A)) - b0
        else:
            tmp = Ys - b0

        B =  b0 + tmp.dot(W.T)


    #Estimate low rank Background    
    model = NMF(n_components=patch.parameters['temporal_params']['nb'], init='nndsvdar')
    b_in = model.fit_transform(np.maximum(B.T, 0))    
    f_in = np.linalg.lstsq(b_in, B.T, rcond = None)[0]
    
    # need to resize 
    patch.b = patch.patch_group.create_dataset('b', shape = (b_in.shape[1], np.prod(patch.dims)), chunks=(1,patch.chunks[0]))
    for i in range(patch.b.shape[0]):        
        tmp = b_in[:,i].astype(np.float32).reshape(new_dims)
        patch.b[i,:] = cv2.resize(tmp, patch.dims[::-1], interpolation=cv2.INTER_LINEAR).flatten()[:]    

    patch.f = patch.patch_group.create_dataset('f', data = f_in.astype(np.float32).T, chunks=(patch.chunks[1],1))

    return

def downscale(patch, ssub, tsub):
    """downscaling without zero padding
    """        
    new_dims    = (patch.duration//tsub,patch.dims[0]//ssub, patch.dims[1]//ssub)    
    Yc          = patch.patch_group.create_dataset('Yc', shape = (new_dims[0], np.prod(new_dims[1:])), chunks = True) # The spatial footprint of size (d,K) with d = (h*w) # python style in lines        
    chunk_size  = patch.Y.chunks[0]
    tmp         = np.zeros((patch.duration, np.prod(new_dims[1:])))
    for i in range(0, patch.duration+chunk_size, chunk_size):        
        data = patch.Y[i:i+chunk_size,:]
        for j, frame in enumerate(data):            
            tmp[i+j,:] = cv2.resize(frame.reshape(patch.dims), new_dims[1:][::-1], interpolation = cv2.INTER_AREA).flatten()[:]
    

    Yc[...] = cv2.resize(tmp, (np.prod(new_dims[1:]), new_dims[0]), fx = 1., fy = 1./tsub, interpolation = cv2.INTER_AREA)
    Yc.attrs['dims'] = new_dims[1:]
    
    return Yc

def initialize_components(patch, gSig, gSiz, ssub, tsub, **kwargs):
    """
    Initalize components

    This method uses a greedy approach followed by hierarchical alternative least squares (HALS) NMF.
    Optional use of spatio-temporal downsampling to boost speed.
    """
    from time import time
    start = time()
    gSig = np.array(gSig)
    gSiz = np.array(gSiz)    
    # rescale according to downsampling factor
    d_gSig = np.round(gSig/ssub).astype(np.int)
    d_gSiz = np.round(gSiz/ssub).astype(np.int)

    # no normalization
    # no choice on the method, it's corr_pnr
    # this icrements the performance against ground truth and solves border problems
    Yc = downscale(patch, ssub, tsub)
    
    # ROI extraction
    kwargs_copy = kwargs.copy()
    kwargs_copy['gSig'] = d_gSig
    kwargs_copy['gSiz'] = d_gSiz
    kwargs_copy['tsub'] = tsub
    kwargs_copy['ssub'] = ssub

    greedyROI_corr(patch, Yc, **kwargs_copy)
    
    K = patch.A.shape[0] # number of neurons initialized
        
    if K:
        center = np.zeros((K, 2))
        for k in range(K):
            center[k] = center_of_mass(patch.A[k].reshape(patch.dims))
        patch.center = center
    else:
        patch.center = []
    
    # print("Time for initialization ", time() -start)

    return 



