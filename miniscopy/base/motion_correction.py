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
#import sima_functions as sima
# import scikit_feature as sima

import pandas as pd
import re
import av
from tqdm import tqdm
import os
import h5py as hd
from IPython.core.debugger import Pdb

def low_pass_filter_space(img_orig, filter_size):
    gSig_filt = (filter_size,filter_size)
    ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig_filt])
    # return cv2.GaussianBlur(img_orig, ksize=ksize, sigmaX=gSig_filt[0],sigmaY=gSig_filt[1], borderType=cv2.BORDER_REFLECT) #- cv2.boxFilter(img_orig, ddepth=-1,ksize=ksize, borderType=cv2.BORDER_REFLECT, normalize = True)
    ker = cv2.getGaussianKernel(ksize[0], gSig_filt[0])
    ker2D = ker.dot(ker.T)
    nz = np.nonzero(ker2D >= ker2D[:, 0].max())
    zz = np.nonzero(ker2D < ker2D[:, 0].max())
    ker2D[nz] -= ker2D[nz].mean()
    ker2D[zz] = 0
    #ker -= ker.mean()
    # return cv2.sepFilter2D(np.array(img_orig,dtype=np.float32),-1,kernelX = ker, kernelY = ker, borderType=cv2.BORDER_REFLECT)
    return cv2.filter2D(np.array(img_orig, dtype=np.float32), -1, ker2D, borderType=cv2.BORDER_REFLECT)

def get_video_info(files):
    video_info  = pd.DataFrame(index = np.arange(len(files)), columns = ['file_name', 'start', 'end', 'duration'])
    videos      = dict.fromkeys(files)
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

def get_mmap_file(videos, video_info, dims, **kwargs):
    mmap_mov    = os.path.split(video_info.index.get_level_values(1)[0])[0] + '/' + 'motion_corrected.mmap'
    fp          = np.memmap(mmap_mov, mode='w+', dtype=np.float32, shape=(video_info['duration'].sum(), np.prod(dims))) 
    for v in tqdm(videos.keys()):
        offset  = int(video_info['start'].xs(v, level=1))
        stream  = next(s for s in videos[v].streams if s.type == 'video')        
        for i, packet in enumerate(videos[v].demux(stream)):            
            frame           = packet.decode_one().to_nd_array(format = 'bgr24')[:,:,0].astype(np.float32)       
            # filter the frame
            # if 'filter_size' in kwargs.keys():
            #     frame = low_pass_filter_space(frame.copy(), kwargs['filter_size'])
            fp[i+offset]    = frame.reshape(np.prod(dims))                    
            if i+1 == stream.duration : break                        
        fp.flush()    
    del videos, fp    

    return mmap_mov

def get_hdf_file(videos, video_info, dims, **kwargs):
    hdf_mov     = os.path.split(video_info.index.get_level_values(1)[0])[0] + '/' + 'motion_corrected.hdf5'
    file        = hd.File(hdf_mov, "w")
    movie       = file.create_dataset('movie', shape = (video_info['duration'].sum(), np.prod(dims)), dtype = np.float32, chunks=True)
    for v in tqdm(videos.keys()):
        offset  = int(video_info['start'].xs(v, level=1))
        stream  = next(s for s in videos[v].streams if s.type == 'video')        
        tmp     = np.zeros((video_info['duration'].xs(v, level=1).values[0], np.prod(dims)), dtype=np.float32)
        for i, packet in enumerate(videos[v].demux(stream)):
            frame           = packet.decode_one().to_nd_array(format = 'bgr24')[:,:,0].astype(np.float32)       
            tmp[i]          = frame.reshape(np.prod(dims))
            if i+1 == stream.duration : break                        
            
        movie[offset:offset+len(tmp),:] = tmp[:]
        del tmp

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

def update_template(movie, template, dims, index):    
    if np.isnan(movie).sum():
        template     = np.nanmedian(movie, axis = 0).reshape(dims)
    else :
        template     = np.median(movie, axis = 0).reshape(dims)
    return template

def get_patches_position(dims, strides, overlaps, **kwargs):
    wdims       = tuple(np.add(strides, overlaps))
    # different from caiman implemantion
    height_pos  = np.arange(0, dims[0], strides[0])
    width_pos   = np.arange(0, dims[1], strides[1])
    patches_index = np.atleast_3d(np.meshgrid(height_pos, width_pos, indexing = 'ij'))
    pdims       = patches_index.shape[1:] 
    return patches_index.reshape(patches_index.shape[0], np.prod(patches_index.shape[1:])).transpose(), wdims, pdims

def apply_shift_iteration(img, shift, border_nan=False, border_type=cv2.BORDER_REFLECT):
    # todo todocument

    sh_x_n, sh_y_n = shift
    w_i, h_i = img.shape
    M = np.float32([[1, 0, sh_y_n], [0, 1, sh_x_n]])    
    min_, max_ = np.min(img), np.max(img)
    img = np.clip(cv2.warpAffine(img, M, (h_i, w_i), flags=cv2.INTER_CUBIC, borderMode=border_type), min_, max_)
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

    return img

def tile_and_correct(index, mmap_mov, mmap_templ, dims, parameters):
    """ perform piecewise rigid motion correction iteration, by
            1) dividing the FOV in patches
            2) motion correcting each patch separately
            3) upsampling the motion correction vector field
            4) stiching back together the corrected subpatches
    """            
    template        = np.memmap(mmap_templ, mode = 'r', dtype = np.float32, shape=dims)
    image           = np.memmap(mmap_mov, mode = 'r+', dtype = np.float32, shape=dims, offset = index*np.prod(dims)*32//8)
    # compute rigid shifts and bound | not sure here
    # shift, error, phasediff = sima.register_translation(image, template, **parameters)
    # parameters['shifts_lb'] = np.ceil(np.subtract(shift, parameters['max_deviation_patch'])).astype(int)
    # parameters['shifts_ub'] = np.floor(np.add(shift, parameters['max_deviation_patch'])).astype(int)
    # extract patches positions
    patches_index, wdims, pdims    = get_patches_position(dims, **parameters)
    # extract shifts for each patch
    shifts_patch = np.zeros((len(patches_index),2))
    for i,patch_pos in enumerate(patches_index):
        xs, xe, ys, ye = (patch_pos[0],patch_pos[0]+wdims[0],patch_pos[1],patch_pos[1]+wdims[1])        
        # filter the patch and the template with a small filter 
        filtered_image = low_pass_filter_space(image[xs:xe,ys:ye].copy(), parameters['filter_size_patch'])
        filtered_template = low_pass_filter_space(template[xs:xe,ys:ye].copy(), parameters['filter_size_patch'])        
        shifts_patch[i], error, phasediff = sima.register_translation(filtered_template, filtered_image, parameters['upsample_factor'], parameters['max_deviation_patch'])
        
    # print('local_correction_max :',index,shifts_patch.max(0))

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

    # apply shift iteration
    num_tiles           = np.prod(upsamp_pdims)
    max_shear           = np.percentile([np.max(np.abs(np.diff(ssshh, axis=xxsss))) for ssshh, xxsss in itertools.product([shift_img_x, shift_img_y], [0, 1])], 75)    
    total_shifts        = np.vstack((shift_img_x.flatten(),shift_img_y.flatten())).transpose()
    new_upsamp_patches  = np.ones((num_tiles, upsamp_wdims[0], upsamp_wdims[1]))*np.inf
    for i, patch_pos in enumerate(upsamp_patches_index):
        if total_shifts[i].sum():
            xs, xe, ys, ye  = (patch_pos[0],patch_pos[0]+upsamp_wdims[0],patch_pos[1],patch_pos[1]+upsamp_wdims[1])
            patch           = image[xs:xe,ys:ye]
            new_upsamp_patches[i,0:patch.shape[0],0:patch.shape[1]] = apply_shift_iteration(patch.copy(), total_shifts[i], border_nan = False)


    normalizer      = np.zeros_like(image)*np.nan
    new_image       = np.zeros_like(image)*np.nan    
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
             
    # writing to disk
    new_image[np.isnan(new_image)] = np.mean(new_image)
    image[:] = new_image[:]
    image.flush()
    del image,template
    return index

def tile_and_correct_batch(index_images, mmap_mov, mmap_templ, dims, parameters):
    for i in index_images:        
        tile_and_correct(i, mmap_mov, mmap_templ, dims, parameters)        
    return index_images

def tile_and_correct_helper(args): return tile_and_correct_batch(*args)

def global_correct(image, template, dims, parameters):
    """ 
        Do a global correction of the image
    """
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
    top_left = cv2.minMaxLoc(res)[3]


    # FROM PYFLUO https://github.com/bensondaled/pyfluo
    ## from hereon in, x and y are reversed in naming convention
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
    
        
    # print('global_correction :',sh_x_n,sh_y_n,avg_metric)
    
    # apply shift using subpixels adjustement
    interpolation = cv2.INTER_LINEAR # TODO : ADD THE CHOICE
    M   = np.float32([[1, 0, sh_y_n], [0, 1, sh_x_n]])
    min_, max_ = np.min(image), np.max(image)
    new_image = cv2.warpAffine(image, M, dims[::-1], flags = interpolation, borderMode=cv2.BORDER_REFLECT)
    new_image = np.clip(new_image, min_, max_)

    return new_image.flatten()

def global_correct_batch(images, template, dims, parameters):
    for i, img in enumerate(images):
        images[i] = global_correct(img, template, dims, parameters)
    return images
    
def global_correct_helper(args): return global_correct_batch(*args)

def map_function(procs, function, nb_splits, splits_index, chunk_movie, template, dims, parameters):    
    pargs = zip(chunk_movie, [template]*nb_splits, [dims]*nb_splits, [parameters]*nb_splits)
    if procs is not None:
        if 'multiprocessing' in str(type(procs)):
            tmp = procs.map_async(eval(function), pargs).get()
        else:
            tmp = procs.map_sync(eval(function), pargs)
            procs.results.clear()                    
    else:
        tmp = list(map(eval(function), pargs))
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
    # 1. Load every movies in only one file | Do a filtering with filter_size parameter
    #################################################################################################
    # files are sorted
    video_info, videos, dims = get_video_info(fnames)
    hdf_mov       = get_hdf_file(videos, video_info, dims)
    
    #################################################################################################
    # 2. Estimate template from first n frame
    #################################################################################################
    template   = get_template(hdf_mov['movie'], dims, start = 0, duration = 500)

    #################################################################################################
    # 3. run motion correction / update template
    #################################################################################################    
    duration    = video_info['duration'].sum()
    # batch_index = np.array_split(np.arange(duration), duration//parameters['batch_size'])
    chunk_size  = hdf_mov['movie'].chunks[0]
    chunk_starts = np.arange(0, duration, chunk_size)
    nb_splits   = os.cpu_count()    
        

    for start in tqdm(chunk_starts):            
        # load the chunk of the movie 
        chunk_movie = hdf_mov['movie'][start:start+chunk_size]

        # index for parrallel process of the chunk size
        # index = np.arange(chunk_size)
        index = np.arange(np.minimum(duration-start, chunk_size))
        splits_index = np.array_split(index, nb_splits)

        # to zip later                
        list_chunk_movie = []
        for idx in splits_index:            
            list_chunk_movie.append(chunk_movie[idx])
               
        for j in range(parameters['nb_round']):        
            # # Do a global motion correction of the group
            list_chunk_movie = map_function(procs, 'global_correct_helper', nb_splits, splits_index, list_chunk_movie, template, dims, parameters)
        
            # # # Do the piecewise motion correction of the group
            # TO CHECK THE PADDING
            # map_function(procs, 'tile_and_correct_helper', nb_splits, splits_index, mmap_mov, mmap_templ, dims, parameters)

        # write the chunk to disk
        chunk_movie = np.vstack(list_chunk_movie)
        hdf_mov['movie'][start:start+chunk_size] = chunk_movie[:]

        # # # # Reevaluate the template from the group
        template = update_template(chunk_movie, template, dims, index)
        
    # set some attribute of the hdf file
    hdf_mov['movie'].attrs['dims'] = dims
    hdf_mov['movie'].attrs['duration'] = duration    

    return hdf_mov, video_info
    
