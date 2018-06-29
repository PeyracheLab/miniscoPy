# -*- coding: utf-8 -*-
"""


The functions apply_shifts_dft, register_translation, _compute_error, _compute_phasediff, and _upsampled_dft are from
SIMA (https://github.com/losonczylab/sima), licensed under the  GNU GENERAL PUBLIC LICENSE, Version 2, 1991.
These same functions were adapted from sckikit-image, licensed as follows:

Copyright (C) 2011, the scikit-image team
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are
 met:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in
     the documentation and/or other materials provided with the
     distribution.
  3. Neither the name of skimage nor the names of its contributors may be
     used to endorse or promote products derived from this software without
     specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.



"""
import numpy as np
import cv2
import itertools
from . import sima_functions as sima 
import warnings
import pandas as pd
import re
import av
from tqdm import tqdm
import os
import h5py as hd
from IPython.core.debugger import Pdb
from copy import copy
from miniscopy.base.sima_functions import *



def get_vector_field_image (folder_name,shift_appli, parameters):
    '''This function is based on the jupyter notebook main_test_motion_correction.ipynb
    parameters:
    -folder_name : str, the name of the folder where is the inital movie
    -shift_appli : ndarry, the shift applied to the template in oder to get the shifted image
    -parameters : dict'''

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import glob

    files = glob.glob(folder_name+'/*.avi')
    if len(files) == 0:
        import urllib.request
        url = "https://www.dropbox.com/s/mujmgcwkit52xpn/A0608_msCam1.avi?dl=1"
        with urllib.request.urlopen(url) as response, open(folder_name +  '/A0608_msCam1.avi', 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        files= glob.glob(folder_name+'/*.avi')
        if len(files) == 0:
            print("No avi files found, please provide one at least")
            sys.exit(-1)
    video_info, videos, dims = get_video_info(files)
    hdf_mov2 = get_hdf_file(videos, video_info, dims,parameters['save_original'])
    movie = hdf_mov2['movie']
    template1 = movie[0].copy()
    template = template1.reshape(dims)
    max_fluo_template, mf_template = get_max_fluo(template,parameters)
    top,bottom, left, right = max_fluo_template[0]-50,max_fluo_template[0]+50,max_fluo_template[1]-50,max_fluo_template[1]+50
    rect = patches.Rectangle((left,top),100,100,linewidth=1,edgecolor='r',facecolor='none',label = 'shift area')
    image =  template.copy()
    image[top : bottom, left:right] = image[top + shift_appli[0] : bottom + shift_appli[0], left+ shift_appli[1]: right + shift_appli[1]]
    max_fluo_image, mf_image = get_max_fluo(image,parameters)
    patches_index, wdims, pdims    = get_patches_position(dims, **parameters)
    shifts_patch = np.zeros((len(patches_index),2))
    for i,patch_pos in enumerate(patches_index):
        xs, xe, ys, ye = (patch_pos[0],np.minimum(patch_pos[0]+wdims[0],dims[0]-1),patch_pos[1],np.minimum(patch_pos[1]+wdims[1],dims[1]-1)) # s = start, e = exit
        filtered_image = low_pass_filter_space(image[xs:xe,ys:ye].copy(), parameters['filter_size_patch'])
        filtered_template = low_pass_filter_space(template[xs:xe,ys:ye].copy(), parameters['filter_size_patch'])
        shifts_patch[i], error, phasediff = register_translation(filtered_template, filtered_image, parameters['upsample_factor'],"real",None,None, parameters['max_shifts']) #coordinate given back in order Y,X
    shift_img_x     = shifts_patch[:,0].reshape(pdims)
    shift_img_y     = shifts_patch[:,1].reshape(pdims) 
    new_overlaps    = parameters['overlaps']
    new_strides     = tuple(np.round(np.divide(parameters['strides'], parameters['upsample_factor_grid'])).astype(np.int))
    upsamp_patches_index, upsamp_wdims, upsamp_pdims = get_patches_position(dims, new_strides, new_overlaps)
    shift_img_x     = cv2.resize(shift_img_x, (upsamp_pdims[1],upsamp_pdims[0]), interpolation = cv2.INTER_CUBIC)
    shift_img_y     = cv2.resize(shift_img_y, (upsamp_pdims[1],upsamp_pdims[0]), interpolation = cv2.INTER_CUBIC)
    X,Y,U,V,Xp,Yp = vector_field(shift_img_x,shift_img_y,new_strides,upsamp_wdims,upsamp_pdims,dims)
    
    return (image,X, Y, U, V,Xp,Yp,rect,dims)

def vector_field (matrix_X, matrix_Y,strides,wdims,pdims,dims):
    """
    Creat a vector field from 2 matrix of coordinates
    
    parameters : 
    -matrix_X = matrix of height coordinates of the vector field
    -matrix_Y = matrix of weight coordinates of the vector field
    -strides = np.array, (top,left) coordinates of each patch
    -wdims = np.array, dimension of each patch (h,w)
    -dims = dimension of the image
    -pdims = np.array, (number of patches on heigt, number of patches on weight)"""
    x= np.zeros(pdims[1])
    y= np.zeros(pdims[0])
    for i in range(0,pdims[1]):
        x[i]= np.minimum(strides[1]*i + wdims[1]/2, dims[1])
    for j in range(0,pdims[0]):
        y[j]= np.minimum(strides[0]*j + wdims[0]/2, dims[0])
    X,Y = np.meshgrid(x,y)

    X_flat = np.ravel(X.copy())
    Y_flat = np.ravel(Y.copy())
    U_flat= np.ravel(matrix_Y.copy())
    V_flat = np.ravel(matrix_X.copy())
    xp = np.zeros(U_flat.shape)
    yp = np.zeros(V_flat.shape) 
    xp.fill(np.nan)
    yp.fill(np.nan)
    for i, uf in enumerate(U_flat): 
        if uf == 0 and V_flat[i] == 0 : # if there is no shift
            U_flat[i] = None
            V_flat[i] = None
            xp[i] = X_flat[i]
            yp[i] = Y_flat[i]

    U = U_flat.reshape(pdims)
    V = V_flat.reshape(pdims)
    Xp = xp.reshape(pdims)
    Yp = yp.reshape(pdims)

    return (X,Y,U,V,Xp,Yp)

def join_patches(image,max_shear,upsamp_wdims,new_overlaps,upsamp_pdims,upsamp_patches_index,patch_pos,total_shifts,new_upsamp_patches):
    
    normalizer      = np.zeros_like(image)*np.nan
    new_image = image.copy()

    if max_shear < 0.5:
        np.seterr(divide='ignore')
        # create weight matrix for blending
        # different from original.    
        tmp             = np.ones(upsamp_wdims)    
        tmp[:new_overlaps[0], :] = np.linspace(0, 1, new_overlaps[0])[:, None]
        tmp             = tmp*np.flip(tmp, 0)
        tmp2             = np.ones(upsamp_wdims)    
        tmp2[:, :new_overlaps[1]] = np.linspace(0, 1, new_overlaps[1])[None, :]
        tmp2            = tmp2*np.flip(tmp2, 1)
        blending_func   = tmp*tmp2
        border          = tuple(itertools.product(np.arange(upsamp_pdims[0]),np.arange(upsamp_wdims[0])))
        for i, patch_pos in enumerate(upsamp_patches_index):
            xs, xe, ys, ye = (patch_pos[0],patch_pos[0]+upsamp_wdims[0],patch_pos[1],patch_pos[1]+upsamp_wdims[1])        
            ye = np.minimum(ye, new_image.shape[1])
            xe = np.minimum(xe, new_image.shape[0])
            prev_val_1  = normalizer[xs:xe,ys:ye]
            prev_val    = new_image[xs:xe,ys:ye]

            tmp = new_upsamp_patches[i,xs-patch_pos[0]:xe-patch_pos[0],ys-patch_pos[1]:ye-patch_pos[1]]
            tmp2 = blending_func[xs-patch_pos[0]:xe-patch_pos[0],ys-patch_pos[1]:ye-patch_pos[1]]

            if xs == 0 or ys == 0 or xe == new_image.shape[0] or ye == new_image.shape[1]:
                normalizer[xs:xe,ys:ye] = np.nansum(np.dstack([~np.isnan(tmp)*1*np.ones_like(tmp2), prev_val_1]),-1)                    
                new_image[xs:xe,ys:ye] = np.nansum(np.dstack([tmp*np.ones_like(tmp2), prev_val]),-1)
            else:
                normalizer[xs:xe,ys:ye] = np.nansum(np.dstack([~np.isnan(tmp)*1*tmp2, prev_val_1]),-1)
                new_image[xs:xe,ys:ye] = np.nansum(np.dstack([tmp*tmp2, prev_val]),-1)

        new_image = new_image/normalizer

    else:

        half_overlap_x = np.int(new_overlaps[0] / 2)
        half_overlap_y = np.int(new_overlaps[1] / 2)        
        for i, patch_pos in enumerate(upsamp_patches_index):
            if total_shifts[i].sum() != 0.0 : 
                if patch_pos[0] == 0 and patch_pos[1] == 0:
                    xs = patch_pos[0]
                    xe = patch_pos[0]+upsamp_wdims[0]-half_overlap_x
                    ys = patch_pos[1]
                    ye = patch_pos[1]+upsamp_wdims[1]-half_overlap_y            
                elif patch_pos[0] == 0:
                    xs = patch_pos[0]
                    xe = patch_pos[0]+upsamp_wdims[0]-half_overlap_x
                    ys = patch_pos[1]+half_overlap_y
                    ye = patch_pos[1]+upsamp_wdims[1]-half_overlap_y                
                    ye = np.minimum(ye, new_image.shape[1])
                elif patch_pos[1] == 0:
                    xs = patch_pos[0]+half_overlap_x
                    xe = patch_pos[0]+upsamp_wdims[0]-half_overlap_x
                    xe = np.minimum(xe, new_image.shape[0])
                    ys = patch_pos[1]
                    ye = patch_pos[1]+upsamp_wdims[1]-half_overlap_y                                
                else:
                    xs = patch_pos[0]+half_overlap_x
                    xe = patch_pos[0]+upsamp_wdims[0]-half_overlap_x
                    xe = np.minimum(xe, new_image.shape[0])
                    ys = patch_pos[1]+half_overlap_y
                    ye = patch_pos[1]+upsamp_wdims[1]-half_overlap_y                                
                    ye = np.minimum(ye, new_image.shape[1])
                new_image[xs:xe,ys:ye] = new_upsamp_patches[i,xs-patch_pos[0]:xe-patch_pos[0],ys-patch_pos[1]:ye-patch_pos[1]]
    
    return(new_image)



def get_max_fluo(img, parameters) :
    """ Return the position and value of the maximum of fluorescence of an image 
    
    Parameters : 
    -img : ndarray, the image where you want to detect the fluorescence 
    
    Returns : 
    -max_fluo : tuple, position of the maximum of fluorescence 
    -mf : float,  value of the maximum of fluorescence"""

    filtered_img = low_pass_filter_space(img.copy(), parameters['filter_size_patch'])
    mf = cv2.minMaxLoc(filtered_img)[1] # Value of the maximum of fluo
    max_fluo = cv2.minMaxLoc(filtered_img)[3] # Position of the maximum of fluo (w,h)
    max_fluo = max_fluo[::-1] # (h,w) 

    
    return max_fluo, mf 

def low_pass_filter_space(img_orig, filter_size):
    """ Filter a 2D image

    Parameters : 
    -img_orig : ndarray, the original image.
    -filter_size : the size of the gaussian kernel to filter the whole field of view.
    
    Return : 
    - filtered image"""

    gSig_filt = (filter_size,filter_size)
    ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig_filt])
    ker = cv2.getGaussianKernel(ksize[0], gSig_filt[0])
    ker2D = ker.dot(ker.T)
    nz = np.nonzero(ker2D >= ker2D[:, 0].max())
    zz = np.nonzero(ker2D < ker2D[:, 0].max())
    ker2D[nz] -= ker2D[nz].mean()
    ker2D[zz] = 0
    return cv2.filter2D(np.array(img_orig, dtype=np.float32), -1, ker2D, borderType=cv2.BORDER_REFLECT)


def get_video_info(files):
    """ In order to get the name, duration, start and end of each video
    
    Parameters:
    -files : the pathe where ther is all the video (.avi)
    
    Returns:
    -videos : dictionnary of the videos from the miniscopes
    -video_info : DataFrame of informations about the video
    -dimension (h,w) of each frame """

    video_info  = pd.DataFrame(index = np.arange(len(files)), columns = ['file_name', 'start', 'end', 'duration'])
    videos      = dict.fromkeys(files) # dictionnary creation
    for f in files:
        num                                 = int(re.findall(r'\d+', f)[-1])-1 
        video_info.loc[num,'file_name']     = f
        video                               = av.open(f)
        stream                              = next(s for s in video.streams if s.type == 'video') 
        video_info.loc[num, 'duration']     = stream.duration
        videos[f]                           = video

    video_info['start']     = video_info['duration'].cumsum()-video_info['duration']
    video_info['end']       = video_info['duration'].cumsum()
    video_info              = video_info.set_index(['file_name'], append = True)

    return video_info, videos, (stream.format.height, stream.format.width)



def get_hdf_file(videos, video_info, dims, save_original, **kwargs):
    """
    In order to convert the video into a HDF5 file.
    Parameters : 
    -videos : dictionnary of the videos from the miniscopes
    -video_info : DataFrame of informations about the video
    -dims : dimension (h,w) of each frame
    
    Returns :
    -file : HDF5 file"""

    hdf_mov     = os.path.split(video_info.index.get_level_values(1)[0])[0] + '/' + 'motion_corrected.hdf5'
    file        = hd.File(hdf_mov, "w")
    movie       = file.create_dataset('movie', shape = (video_info['duration'].sum(), np.prod(dims)), dtype = np.float32, chunks=True)
    if save_original:
        original = file.create_dataset('original', shape = (video_info['duration'].sum(), np.prod(dims)), dtype = np.float32, chunks=True)
    for v in tqdm(videos.keys()):
        offset  = int(video_info['start'].xs(v, level=1))
        stream  = next(s for s in videos[v].streams if s.type == 'video')        
        tmp     = np.zeros((video_info['duration'].xs(v, level=1).values[0], np.prod(dims)), dtype=np.float32)
        for i, packet in enumerate(videos[v].demux(stream)):
            frame           = packet.decode_one().to_nd_array(format = 'bgr24')[:,:,0].astype(np.float32)       
            tmp[i]          = frame.reshape(np.prod(dims))
            if i+1 == stream.duration : break                        
            
        movie[offset:offset+len(tmp),:] = tmp[:]
        if save_original:
            original[offset:offset+len(tmp),:] = tmp[:]
        del tmp
    if save_original:
        del original
    del movie 

    file.attrs['folder'] = os.path.split(video_info.index.get_level_values(1)[0])[0]
    file.attrs['filename'] = hdf_mov
    return file

def get_template(movie, dims, start = 0, duration = 1):
    if np.isnan(movie[start:start+duration]).sum(): 
        template     = np.nanmedian(movie[start:start+duration], axis = 0).reshape(dims)
    else :
        template     = np.median(movie[start:start+duration], axis = 0).reshape(dims)
    return template

def get_patches_position(dims, strides, overlaps, **kwargs):
    ''' Return a matrix of the position of each patches without overlapping, the dimension of each patch and the dimension of this matrix 

    Positional arguments :
    -dims : dimension of the template image
    -strides : the dimension of each patches without overlapping
    -overlaps : dimension of overlaps'''

    wdims       = tuple(np.add(strides, overlaps)) #dimension of patches
    # different from caiman implemantion
    height_pos  = np.arange(0, dims[0], strides[0])
    width_pos   = np.arange(0, dims[1], strides[1])
    patches_index = np.atleast_3d(np.meshgrid(height_pos, width_pos, indexing = 'ij'))
    pdims       = patches_index.shape[1:] #dimension of the patches index 
    return patches_index.reshape(patches_index.shape[0], np.prod(patches_index.shape[1:])).transpose(), wdims, pdims

def apply_shift_iteration(img, shift, border_nan=False, border_type=cv2.BORDER_REFLECT):
    """Applied an affine transformation to an image
    
    Parameters:
    -img : ndarray, image to be transformed
    -shift: ndarray, (h,w), the shift to be applied to the original image
    -border_nan : how to deal with the borders
    -border_type : pixel extrapolation method 
    
    Returns:
    - img : ndarray, image transformed"""


    sh_x_n, sh_y_n = shift
    w_i, h_i = img.shape
    M = np.float32([[1, 0, sh_y_n], [0, 1, sh_x_n]])    
    min_, max_ = np.min(img), np.max(img)
    img = np.clip(cv2.warpAffine(img, M, (h_i, w_i), flags = cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT), min_, max_)
    if border_nan:
        max_w, max_h, min_w, min_h = 0, 0, 0, 0
        max_h, max_w = np.ceil(np.maximum((max_h, max_w), shift)).astype(np.int)
        min_h, min_w = np.floor(np.minimum((min_h, min_w), shift)).astype(np.int)
        img[:max_h, :] = np.nan
        if min_h < 0:
            img[min_h:, :] = np.nan
        img[:, :max_w] = np.nan
        if min_w < 0:
            img[:, min_w:] = np.nan
    img[np.isinf(img)] = np.nan
    img[np.isnan(img)] = np.nanmean(img)

    return img

def tile_and_correct(image, template, dims, parameters):
    """ perform piecewise rigid motion correction iteration, by
        1) dividing the FOV in patches
        2) motion correcting each patch separately
        3) upsampling the motion correction vector field
        4) stiching back together the corrected subpatches"""            
        
    image           = image.reshape(dims)    
    template_uncrop = template.copy()
    template        = template.reshape(dims)

    # extract patches positions
    patches_index, wdims, pdims    = get_patches_position(dims, **parameters)

    # extract shifts for each patch
    shifts_patch = np.zeros((len(patches_index),2))
    for i,patch_pos in enumerate(patches_index):
        xs, xe, ys, ye = (patch_pos[0],np.minimum(patch_pos[0]+wdims[0],dims[0]-1),patch_pos[1],np.minimum(patch_pos[1]+wdims[1],dims[1]-1)) # s = start, e = exit
        filtered_image = low_pass_filter_space(image[xs:xe,ys:ye].copy(), parameters['filter_size_patch'])
        filtered_template = low_pass_filter_space(template[xs:xe,ys:ye].copy(), parameters['filter_size_patch'])
        shifts_patch[i], error, phasediff = register_translation(filtered_template, filtered_image, parameters['upsample_factor'],"real",None,None, parameters['max_shifts']) #coordinate given back in order Y,X

    # create a vector field    
    shift_img_x     = shifts_patch[:,0].reshape(pdims)
    shift_img_y     = shifts_patch[:,1].reshape(pdims) 


    # upsampling 
    new_overlaps    = parameters['overlaps']
    new_strides     = tuple(np.round(np.divide(parameters['strides'], parameters['upsample_factor_grid'])).astype(np.int))
    upsamp_patches_index, upsamp_wdims, upsamp_pdims = get_patches_position(dims, new_strides, new_overlaps)

    # resize shift_img_
    shift_img_x     = cv2.resize(shift_img_x, (upsamp_pdims[1],upsamp_pdims[0]), interpolation = cv2.INTER_CUBIC)
    shift_img_y     = cv2.resize(shift_img_y, (upsamp_pdims[1],upsamp_pdims[0]), interpolation = cv2.INTER_CUBIC)

    #create vector field
    x= np.zeros(upsamp_pdims[1])
    y= np.zeros(upsamp_pdims[0])
    for i in range(0,upsamp_pdims[1]):
        x[i]= new_strides[1]*i + upsamp_wdims[1]/2
    for j in range(0,upsamp_pdims[0]):
        y[j]= new_strides[0]*j + upsamp_wdims[0]/2
    X,Y = np.meshgrid(x,y)

    U_flat= np.ravel(shift_img_y.copy())
    V_flat = np.ravel(shift_img_x.copy())
    for i, uf in enumerate(U_flat):
        if uf == 0 and V_flat[i] == 0 :
            U_flat[i] = None
            V_flat[i] = None

        
    U = U_flat.reshape(upsamp_pdims)
    V = V_flat.reshape(upsamp_pdims)

    #apply shift iteration
    num_tiles           = np.prod(upsamp_pdims) #number of patches
    max_shear           = np.percentile([np.max(np.abs(np.diff(ssshh, axis=xxsss))) for ssshh, xxsss in itertools.product([shift_img_x, shift_img_y], [0, 1])], 75)    
    total_shifts        = np.vstack((shift_img_x.flatten(),shift_img_y.flatten())).transpose()
    new_upsamp_patches  = np.ones((num_tiles, upsamp_wdims[0], upsamp_wdims[1]))*np.inf
    for i, patch_pos in enumerate(upsamp_patches_index):
        if total_shifts[i].sum():#where there is a shift
            xs, xe, ys, ye  = (patch_pos[0],np.minimum(patch_pos[0]+upsamp_wdims[0],dims[0]-1),patch_pos[1],np.minimum(patch_pos[1]+upsamp_wdims[1],dims[1]-1))
            patch           = image[xs:xe,ys:ye]
            new_upsamp_patches[i,0:patch.shape[0],0:patch.shape[1]] = apply_shift_iteration(patch.copy(), total_shifts[i], border_nan = True)



    normalizer      = np.zeros_like(image)*np.nan
    new_image       = np.copy(image)
    med             = np.median(new_image)
    if max_shear < 0.5:        
        np.seterr(divide='ignore')
        # create weight matrix for blending
        # different from original.    
        tmp             = np.ones(upsamp_wdims)    
        tmp[:new_overlaps[0], :] = np.linspace(0, 1, new_overlaps[0])[:, None]
        tmp             = tmp*np.flip(tmp, 0)
        tmp2             = np.ones(upsamp_wdims)    
        tmp2[:, :new_overlaps[1]] = np.linspace(0, 1, new_overlaps[1])[None, :]
        tmp2            = tmp2*np.flip(tmp2, 1)
        blending_func   = tmp*tmp2
        border          = tuple(itertools.product(np.arange(upsamp_pdims[0]),np.arange(upsamp_wdims[0])))
        for i, patch_pos in enumerate(upsamp_patches_index):
            xs, xe, ys, ye = (patch_pos[0],patch_pos[0]+upsamp_wdims[0],patch_pos[1],patch_pos[1]+upsamp_wdims[1])        
            ye = np.minimum(ye, new_image.shape[1])
            xe = np.minimum(xe, new_image.shape[0])
            prev_val_1  = normalizer[xs:xe,ys:ye]
            prev_val    = new_image[xs:xe,ys:ye]

            tmp = new_upsamp_patches[i,xs-patch_pos[0]:xe-patch_pos[0],ys-patch_pos[1]:ye-patch_pos[1]]
            tmp2 = blending_func[xs-patch_pos[0]:xe-patch_pos[0],ys-patch_pos[1]:ye-patch_pos[1]]

            if xs == 0 or ys == 0 or xe == new_image.shape[0] or ye == new_image.shape[1]:
                normalizer[xs:xe,ys:ye] = np.nansum(np.dstack([~np.isnan(tmp)*1*np.ones_like(tmp2), prev_val_1]),-1)
                new_image[xs:xe,ys:ye] = np.nansum(np.dstack([tmp*np.ones_like(tmp2), prev_val]),-1)
            else:
                normalizer[xs:xe,ys:ye] = np.nansum(np.dstack([~np.isnan(tmp)*1*tmp2, prev_val_1]),-1)
                new_image[xs:xe,ys:ye] = np.nansum(np.dstack([tmp*tmp2, prev_val]),-1)

        new_image = new_image/normalizer
    else:

        half_overlap_x = np.int(new_overlaps[0] / 2)
        half_overlap_y = np.int(new_overlaps[1] / 2)        
        for i, patch_pos in enumerate(upsamp_patches_index):
            if total_shifts[i].sum() != 0.0 :
                if patch_pos[0] == 0 and patch_pos[1] == 0:
                    xs = patch_pos[0]
                    xe = patch_pos[0]+upsamp_wdims[0]-half_overlap_x
                    ys = patch_pos[1]
                    ye = patch_pos[1]+upsamp_wdims[1]-half_overlap_y            
                elif patch_pos[0] == 0:
                    xs = patch_pos[0]
                    xe = patch_pos[0]+upsamp_wdims[0]-half_overlap_x
                    ys = patch_pos[1]+half_overlap_y
                    ye = patch_pos[1]+upsamp_wdims[1]-half_overlap_y                
                    ye = np.minimum(ye, new_image.shape[1])
                elif patch_pos[1] == 0:
                    xs = patch_pos[0]+half_overlap_x
                    xe = patch_pos[0]+upsamp_wdims[0]-half_overlap_x
                    xe = np.minimum(xe, new_image.shape[0])
                    ys = patch_pos[1]
                    ye = patch_pos[1]+upsamp_wdims[1]-half_overlap_y                                
                else:
                    xs = patch_pos[0]+half_overlap_x
                    xe = patch_pos[0]+upsamp_wdims[0]-half_overlap_x
                    xe = np.minimum(xe, new_image.shape[0])
                    ys = patch_pos[1]+half_overlap_y
                    ye = patch_pos[1]+upsamp_wdims[1]-half_overlap_y                                
                    ye = np.minimum(ye, new_image.shape[1])
                new_patch = new_upsamp_patches[i,xs-patch_pos[0]:xe-patch_pos[0],ys-patch_pos[1]:ye-patch_pos[1]]
                new_patch[np.isinf(new_patch)] = np.nan
                dims_patch = new_patch.shape
                if np.isnan(new_patch).all():
                    new_patch[:,:] = med
                elif np.isnan(new_patch).any():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        new_patch[np.isnan(new_patch)] = np.nanmedian(new_patch)
            
                new_image[xs:xe,ys:ye] = new_patch
 
    return new_image.flatten()

def global_correct(image, template, dims, parameters):
    """ 
        Do a global correction of the image """

    max_dev = parameters['max_deviation_rigid']
    
    image           = image.reshape(dims)    
    template_uncrop = template.copy()
    template        = template_uncrop[max_dev:-max_dev,max_dev:-max_dev]

    # filter the image and the template with a large filter 
    filtered_image = low_pass_filter_space(image.copy(), parameters['filter_size'])
    filtered_template = low_pass_filter_space(template.copy(), parameters['filter_size'])

    # call opencv match template    
    res = cv2.matchTemplate(filtered_image, filtered_template, cv2.TM_CCOEFF_NORMED)  
    avg_metric = np.mean(res)
    top_left = cv2.minMaxLoc(res)[3] #get the maximum location

    
    # FROM PYFLUO https://github.com/bensondaled/pyfluo
    ## from here x and y are reversed in naming convention 
    sh_y,sh_x = top_left
    
    if (0 < top_left[1] < 2 * max_dev-1) and (0 < top_left[0] < 2 * max_dev-1):
        ms_h = ms_w = max_dev
        # if max is internal, check for subpixel shift using gaussian peak registration        
        log_xm1_y = np.log(res[sh_x-1,sh_y])          
        log_xp1_y = np.log(res[sh_x+1,sh_y])             
        log_x_ym1 = np.log(res[sh_x,sh_y-1])             
        log_x_yp1 = np.log(res[sh_x,sh_y+1])             
        four_log_xy = 4*np.log(res[sh_x,sh_y])
        sh_x_n = -(sh_x - ms_h + (log_xm1_y - log_xp1_y) / (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
        sh_y_n = -(sh_y - ms_w + (log_x_ym1 - log_x_yp1) / (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
    else:
        sh_x_n = -(sh_x - max_dev)
        sh_y_n = -(sh_y - max_dev)    
    
    # apply shift using subpixels adjustement
    interpolation = cv2.INTER_LINEAR
    M   = np.float32([[1, 0, sh_y_n], [0, 1, sh_x_n]])
    min_, max_ = np.min(image), np.max(image)
    new_image = cv2.warpAffine(image, M, dims[::-1], flags = interpolation, borderMode=cv2.BORDER_REFLECT) 
    new_image = np.clip(new_image, min_, max_)

    return new_image.flatten()  

def make_corrections(images, template, dims, parameters): 
    ''' Do a global and a loc correction of a cluster of images'''

    for i, img in enumerate(images):
        img_glob = global_correct(img, template, dims, parameters)
        img_loc = tile_and_correct(img_glob, template, dims, parameters)
        images[i] = img_loc
    return images

def map_function(procs, nb_splits, chunk_movie, template, dims, parameters): 
    ''' Do multiprocessing'''    

    if procs is not None:
        pargs = zip(chunk_movie, [template]*nb_splits, [dims]*nb_splits, [parameters]*nb_splits)
        if 'multiprocessing' in str(type(procs)):
            tmp = procs.starmap_async(make_corrections, pargs).get() 
        else:
            tmp = procs.starmap_sync(make_corrections, pargs)            
            procs.results.clear()                    
    else:
        tmp = list(map(make_corrections, chunk_movie, [template]*nb_splits, [dims]*nb_splits, [parameters]*nb_splits))

    return tmp



def normcorre(fnames, procs, parameters):
    """
        see 
        Pnevmatikakis, E.A., and Giovannucci A. (2017). 
        NoRMCorre: An online algorithm for piecewise rigid motion correction of calcium imaging data. 
        Journal of Neuroscience Methods, 291:83-92
        or 
        CaiMan github
    """
    #################################################################################################
    # 1. Load every movies in only one file 
    #################################################################################################
    # files are sorted
    video_info, videos, dims = get_video_info(fnames)
    hdf_mov       = get_hdf_file(videos, video_info, dims, parameters['save_original'])
    
    #################################################################################################
    # 2. Estimate template from first n frame
    #################################################################################################
    template   = get_template(hdf_mov['movie'], dims, start = 0, duration = 500)

    #################################################################################################
    # 3. run motion correction / update template
    #################################################################################################    
    duration    = video_info['duration'].sum()  
    chunk_size  = hdf_mov['movie'].chunks[0] 
    chunk_starts_glob = np.arange(0, duration, chunk_size)
    nb_splits   = os.cpu_count() 

    block_size = parameters['block_size'] 
    coeff_euc = block_size//chunk_size # how many whole chunk there is in a block
    new_block = chunk_size*coeff_euc
    block_starts = np.arange(0,duration,new_block) 
  
   
    for i in range(parameters['nb_round']): # loop on the movie
        for start_block in tqdm(block_starts): # for each block
            chunk_starts_loc = np.arange(start_block,start_block+new_block,chunk_size)
            for start_chunk in tqdm(chunk_starts_loc) : # for each chunk                
                chunk_movie = hdf_mov['movie'][start_chunk:start_chunk+chunk_size]
                index = np.arange(chunk_movie.shape[0])
                splits_index = np.array_split(index, nb_splits)
                list_chunk_movie = [] #split of a chunk
                for idx in splits_index:
                    list_chunk_movie.append(chunk_movie[idx]) #each split of a chunk will be process in a different processor of the computer

                new_chunk = map_function(procs, nb_splits, list_chunk_movie, template, dims, parameters)
                new_chunk_arr = np.vstack(new_chunk)
                hdf_mov['movie'][start_chunk:start_chunk+chunk_size] = np.array(new_chunk_arr) #update of the chunk

            template = get_template(hdf_mov['movie'], dims, start = start_block, duration = new_block) #update the template after each block 
    
    hdf_mov['movie'].attrs['dims'] = dims
    hdf_mov['movie'].attrs['duration'] = duration 

    return hdf_mov, video_info