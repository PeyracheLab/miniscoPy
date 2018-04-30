import numpy as np
from IPython.core.debugger import Pdb
import cv2
import scipy.sparse as spr
import h5py as hd

def generate_data(frate, N, T, dims, framerate=100):
    from scipy.ndimage.filters import gaussian_filter
    difference_of_Gaussians=True
    sig=(4, 4)
    truncate=np.exp(-2)    
    M = int(N * 1.5)
    # the position in the picture
    centers = 4 + (np.random.rand(M, 2) * (np.array(dims) - 2 * 4)).astype('uint16')
    # spatial patch in all the image
    true_A = np.zeros((M,)+dims, dtype='float32')
    for i in range(M):
        true_A[(i,)+tuple(centers[i])] = 1.

    if difference_of_Gaussians:    
        q = 0.75
        for n in range(M):
            s = (.67 + .33 * np.random.rand(2)) * np.array(sig)
            tmp = gaussian_filter(true_A[n], s)
            tmp2 = gaussian_filter(true_A[n], q*s)
            tmp3 = q**2 * (.2 + .6 * np.random.rand())
            true_A[n] = np.maximum(tmp - tmp2 * tmp3, 0)
    else:
        for n in range(M):
            s = [ss * (.75 + .25 * np.random.rand()) for ss in sig]
            true_A[n] = gaussian_filter(true_A[n], s)

    # doing A in row
    true_A = true_A.reshape(M, np.prod(dims))
    true_A *= (true_A >= np.vstack(true_A.max(1)*truncate))
    true_A /= np.vstack(np.linalg.norm(true_A, 2, 1))
    keep = np.ones(M, dtype = bool)
    overlap = (true_A.dot(true_A.T) - np.eye(M)).diagonal()
    keep = np.argsort(overlap)[0:N]
    true_A = true_A[keep]

    centers = centers[keep,:]


    true_S = frate.copy()
    # calcium trace
    tau=0.7
    true_C = true_S.astype('float32')
    for i in range(N):
        gamma = np.exp(-1. / (tau * framerate))
        for t in range(1, T):
            true_C[t, i] += gamma * true_C[t-1, i]


    fluctuating_bkgrd=[50, 80]
    bkgrd = 10  # fluorescence baseline
    if fluctuating_bkgrd:
        K = np.array([[np.exp(-(i - j)**2 / 2. / fluctuating_bkgrd[0]**2)
                       for i in range(T)] for j in range(T)])
        ch = np.linalg.cholesky(K + 1e-10 * np.eye(T))
        # fluctuation of fluorescence in time
        true_f = 1e-2 * ch.dot(np.random.randn(T)).astype('float32') / bkgrd
        true_f -= true_f.mean()
        true_f += 1
        K = np.array([[np.exp(-(i - j)**2 / 2. / fluctuating_bkgrd[1]**2)
                       for i in range(dims[0])] for j in range(dims[0])])
        ch = np.linalg.cholesky(K + 1e-10 * np.eye(dims[0]))
        true_b = 3 * 1e-2 * np.outer(*ch.dot(np.random.randn(dims[0], 2)).T).flatten().astype('float32')            
        true_b -= true_b.mean()
        true_b += 1        
        true_b = cv2.resize(true_b.reshape(dims[0],dims[0]), dims[::-1])
        true_b = true_b.flatten()
    else:
        true_f = np.ones(T, dtype='float32')
        true_b = np.ones(np.prod(dims), dtype='float32')

    # adding some local background fluorescence based on true Neurons
    s = (.67 + .33 * np.random.rand(2)) * np.array(sig) * 6.0
    local_background_A = np.zeros_like(true_A)
    for i, a in enumerate(true_A):      
        local_background_A[i] = gaussian_filter(a.reshape(dims), s).flatten()*2.0
    local_background_S  = np.random.poisson(0.04, (T, N)) + frate
    tau=0.7
    local_background_C = local_background_S.astype('float32')
    for i in range(N):
        gamma = np.exp(-1. / (tau * framerate))
        for t in range(1, T):
            local_background_C[t, i] += gamma * local_background_C[t-1, i]

    # + random local background from out of focus neurons
    R = 30
    centers_R = 4 + (np.random.rand(R, 2) * (np.array(dims) - 2 * 4)).astype('uint16')
    A_R = np.zeros(((R,)+dims))
    s = (.67 + .33 * np.random.rand(2)) * np.array(sig) * 20.0
    for i in range(R):
        A_R[(i,)+tuple(centers_R[i])] = 1.
        A_R[i] = gaussian_filter(A_R[i], s)*20
    A_R = A_R.reshape(R, np.prod(dims))
    S_R = np.random.poisson(0.5, (T, R))
    C_R = S_R.astype('float32')
    tau = 0.1
    for i in range(R):
        gamma = np.exp(-1. / (tau * framerate))
        for t in range(1, T):
            C_R[t, i] += gamma * C_R[t-1, i]

    # total local background 
    local_background_A = np.vstack((local_background_A, A_R))
    local_background_C = np.hstack((local_background_C, C_R))
    local_background = local_background_C.dot(local_background_A)

    true_b *= bkgrd
    noise   =.1

    Yr = true_C.dot(true_A) + local_background + np.outer(true_f, true_b) + noise * np.random.randn(T,np.prod(dims)).astype('float32')

    true_data = {   'A':true_A,
                    'C':true_C,
                    'S':true_S,
                    'b':true_b,
                    'f':true_f,
                    'center':centers}
    return Yr, true_data

def compute_residuals(patch):
    from time import time
    start = time()
    
    Ab = np.vstack((patch.A, patch.b))
    Cf = np.hstack((patch.C, patch.f))
    nA = np.ravel(np.power(Ab, 2.0).sum(axis=1))
    tmp = 1./nA  * np.eye(Ab.shape[0])

    chunk_size_row, chunk_size_col = patch.Y.chunks
    YdotAbT = np.zeros((patch.duration, Ab.shape[0]))
    for i in range(0, patch.duration+chunk_size_row,chunk_size_row):        
        for j in range(0, Ab.shape[1]+chunk_size_col,chunk_size_col):            
            YdotAbT[i:i+chunk_size_row,:] += patch.Y[i:i+chunk_size_row,j:j+chunk_size_col].dot(Ab[:,j:j+chunk_size_col].T)
    
    YA = YdotAbT.dot(tmp)
    # YA = (Y.dot(Ab.T)).dot(tmp)

    AA = (Ab.dot(Ab.T)).dot(tmp)
        
    YrA = (YA - Cf.dot(AA.T))[:,:patch.A.shape[0]]
    if patch.A.shape[0]:
        patch.YrA = patch.patch_group.create_dataset('YrA', data = YrA, chunks = (patch.chunks[0], 1))
    else:
        patch.YrA = patch.patch_group.create_dataset('YrA', data = YrA)
    
    # print("Time to compute_residuals ", time() - start)
    return

def update_order_greedy(A, flag_AA=True):
    """Determines the update order of the temporal components

    this, given the spatial components using a greedy method
    Basically we can update the components that are not overlapping, in parallel

    Input:
     -------
     A:       sparse crc matrix
              matrix of spatial components (d x K)
     OR
              A.T.dot(A) matrix (d x d) if flag_AA = true

     flag_AA: boolean (default true)

     Outputs:
     ---------
     parllcomp:   list of sets
          list of subsets of components. The components of each subset can be updated in parallel

     len_parrllcomp:  list
          length of each subset

    @author: Eftychios A. Pnevmatikakis, Simons Foundation, 2017
    """
    K = np.shape(A)[-1]
    parllcomp = []
    for i in range(K):
        new_list = True
        for ls in parllcomp:
            if flag_AA:
                if A[i, ls].nnz == 0:
                    ls.append(i)
                    new_list = False
                    break
            else:
                if (A[:, i].T.dot(A[:, ls])).nnz == 0:
                    ls.append(i)
                    new_list = False
                    break

        if new_list:
            parllcomp.append([i])
    len_parrllcomp = [len(ls) for ls in parllcomp]
    return parllcomp, len_parrllcomp

def normalize_AC(patch, doYrA = True):
    """ Normalize to unit norm A and b
    Parameters:
    ----------
    A,C,Yr,b,f: 
        outputs of CNMF
    """
    nr = patch.A.shape[0]
    nA = np.zeros(nr)
    if nr:
        chunk_size_col = patch.A.chunks[1]
        for i in range(0, patch.A.shape[1]+chunk_size_col, chunk_size_col):
            nA += np.power(patch.A[:,i:i+chunk_size_col], 2.0).sum(1).flatten()        
        nA = np.sqrt(nA)
        for i in range(nr):
            patch.A[i,:] = patch.A[i,:]/nA[i]
            patch.C[:,i] = patch.C[:,i]*nA[i]
            if doYrA:
                patch.YrA[:,i] = patch.YrA[:,i]*nA[i]

    nb = patch.b.shape[0]
    nB = np.zeros(nb)
    if nb:
        chunk_size_col = patch.b.chunks[1]
        for i in range(0, patch.b.shape[1]+chunk_size_col, chunk_size_col):
            nB += np.power(patch.b[:,i:i+chunk_size_col], 2.0).sum(1).flatten()
        nB = np.sqrt(nB)
        for i in range(nb):
            patch.b[i,:] = patch.b[i,:] / nB[i]
            patch.f[:,i] = patch.f[:,i] * nB[i]


    return

def get_default_parameters(key='all'):
    parameters = {'cnmfe': {'init_params': {'bd': 1,
                                           'center_psf': True,
                                           'gSig': (4, 4),
                                           'gSiz': (10, 10),
                                           'min_corr': 0.8,
                                           'min_pixel': 3,
                                           'min_pnr': 10,
                                           'ring_size_factor': 1.5,
                                           'ssub': 2,
                                           'thresh_init': 2,
                                           'tsub': 2, 
                                           'filter_data_centering':True},
                          'merge_params': {'thr': 0.8},
                          'patch_params': {'nb_patch': (3, 4),
                                           'only_init': True,
                                           'overlaps': 40,
                                           'remove_very_bad_comps': False,
                                           'skip_refinement': False,
                                           'ssub': 1,
                                           'tsub': 1},
                          'preprocess_params': {'check_nan': False,
                                           'max_num_samples_fft': 3072,
                                           'noise_method': 'mean',
                                           'noise_range': (0.25, 0.5)},
                          'spatial_params': {'dist': 3,
                                           'extract_cc': True,
                                           'low_rank_background': True,
                                           'max_size': 8,
                                           'maxthr': 0.1,
                                           'medw': (3, 3),
                                           'method': 'dilate',
                                           'method_least_square': 'lasso_lars',
                                           'min_size': 3,
                                           'normalize': True,
                                           'nrgthr': 0.9999,
                                           'thr_method': 'nrg',
                                           'update_background_components': True},
                          'temporal_params': {'ITER': 2,
                                           'bas_nonneg': False,
                                           'block_size': 5000,
                                           'fudge_factor': 0.96,
                                           'lags': 5,
                                           'memory_efficient': False,
                                           'method': 'oasis',
                                           'nb': 16,
                                           'noise_method': 'mean',
                                           'noise_range': (0.25, 0.5),
                                           'num_blocks_per_run': 5,
                                           'p': 1,
                                           'solvers': ['ECOS', 'SCS']}},
                    'frame_rate': 30,
                    'motion_correction': {'batch_size': 100,
                                          'filter_size': 10,
                                          'filter_size_patch': 5,
                                          'max_deviation_patch': 1,
                                          'max_deviation_rigid': 10,
                                          'nb_round': 5,
                                          'overlaps': (6, 6),
                                          'strides': (96, 94),
                                          'upsample_factor': 2,
                                          'upsample_factor_grid': 1}}
    if key is 'all':
        return parameters
    elif key is 'cnmfe':
        return parameters[key]
    elif key is 'motion_correction':
        return parameters[key]
    else:
        print("unknown key")
    return

def copy_data(filename, source, target, name, index, chunking):
  # should be more general
  print("yo")
  file = hd.File(filename, 'r+')
  print(file[source].shape)
  file[target].create_dataset(name, data = file[source][:,index], chunks = chunking)
  print(file[target+'/'+name].shape)
  file.close()
  del file
  return