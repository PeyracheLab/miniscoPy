'''
	Author: Elena Kerjean
	Date : 18/06/2018

	Test of the motion correction algorithm
'''
import numpy as np
from time import time
import scipy
import glob
import yaml
import sys,os
import h5py as hd
from time import time
import av
from tqdm import tqdm
from miniscopy.base.motion_correction import normcorre, get_video_info, get_hdf_file
from miniscopy.base.motion_correction import *

# from miniscopy import setup_cluster, CNMFE
# from miniscopy import Movie


folder_name = 'example_movies'
files = glob.glob(os.path.join(folder_name, '*msCam*.avi'))
parameters = yaml.load(open(os.path.join(folder_name, 'parameters.yaml'), 'r'))
parameters['motion_correction']['nb_round'] = 1
fnames = files
parameters = parameters['motion_correction']
parameters['block_size'] = 200
video_info, videos, dims = get_video_info(fnames)
hdf_mov       = get_hdf_file(videos, video_info, dims, parameters['save_original'])
# sys.exit()


# hdf_mov = hd.File('example_movies/motion_corrected.hdf5', 'r+')
duration = hdf_mov['movie'].shape[0]
dims = (480,752)

t1 = time()
template   = get_template(hdf_mov['movie'], dims, start = 0, duration = 500)
print("Get template ", time()-t1)


chunk_size 		= parameters['block_size']- (parameters['block_size']%hdf_mov['movie'].chunks[0])
chunk_starts 	= np.arange(0,duration,chunk_size) 

t1 = time()
for r in range(parameters['nb_round']): # loop on the movie
	for start_chunk in chunk_starts: # for each chunks		
		t2 = time()
		# make_corrections(hdf_mov['movie'], start_chunk, start_chunk+chunk_size, template, dims, parameters) 
		# print(time()-t2)
		movie = hdf_mov['movie']
		images = movie[start_chunk:start_chunk+chunk_size]
		max_dev = parameters['max_deviation_rigid']
		filter_size = parameters['filter_size']
		
		# t1 = time()         

		# kernel for filtering
		kernel  = get_kernel(filter_size)
		ksize   = kernel.shape[0]
		offset  = (ksize-1)//2
		t2 = time()

		# preparing the template
		template_crop   = template.copy()
		template_crop   = template_crop[max_dev:-max_dev,max_dev:-max_dev]
		tdims           = template_crop.shape
		template_crop   = template_crop[np.newaxis]    
		template_padded = pad_array(template_crop, offset)
		t3 = time()

		# padding the images
		images = images.reshape(images.shape[0], dims[0], dims[1])
		images_padded   = pad_array(images, offset)
		t4 = time()

		images_gpu = cp.asarray(images_padded)
		t44 = time()
		kernel_gpu = cp.asarray(kernel)
		t45 = time()
		filtered_images = low_pass_filter_space(images_gpu, kernel_gpu, offset, dims[0], dims[1])
		# filtered_images = low_pass_filter_space(images_padded, kernel, offset, dims[0], dims[1])
		t5 = time()

		# # filtering images and template
		# filtered_template = low_pass_filter_space(template_padded, kernel, offset, tdims[0], tdims[1])
		# filtered_images = low_pass_filter_space(images_padded, kernel, offset, dims[0], dims[1])
		# t5 = time()

		# # match template
		# filtered_template = np.squeeze(filtered_template, 0)
		# res_all = match_template(filtered_images, filtered_template, max_dev)
		# max_loc     = np.zeros((images.shape[0], 2), dtype = np.int)
		# for i in range(images.shape[0]):
		#     res = res_all[i]
		#     max_loc[i] = np.array(np.unravel_index(np.argmax(res.flatten()), res.shape))
		# t6 = time()

		# sh_x_n, sh_y_n = estimate_shifts(res_all, max_loc, max_dev)

		print("kermel for filtering", t2 - t1)
		print("preparing the template", t3 - t2)
		print("padding the images", t4 - t3)
		print("to gpu ", t44-t4, t45-t44)
		print("filtering ", t5 - t4)

		sys.exit()

print("Global motion correction", time()-t1)

