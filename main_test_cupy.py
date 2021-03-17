'''

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
import cupy as cp
import scipy.signal
# from miniscopy import setup_cluster, CNMFE
from miniscopy import Movie



folder_name = 'example_movies'
files = glob.glob(os.path.join(folder_name, '*.avi'))
parameters = yaml.load(open(os.path.join(folder_name, 'parameters.yaml'), 'r'))
parameters['motion_correction']['nb_round'] = 2
fnames = [files[0]]
parameters = parameters['motion_correction']
parameters['block_size'] = 200
video_info, videos, dims = get_video_info(fnames)
hdf_mov       = get_hdf_file(videos, video_info, dims, parameters['save_original'])



# hdf_mov = hd.File('example_movies/motion_corrected.hdf5', 'r+')
duration = hdf_mov['movie'].shape[0]


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
		images = cp.asarray(images)

		# kernel for filtering
		kernel  = get_kernel(filter_size)
		ksize   = kernel.shape[0]
		offset  = (ksize-1)//2		
		t2 = time()

		# preparing the template
		template_crop   = template.copy()
		template_crop   = template_crop[max_dev:-max_dev,max_dev:-max_dev]		
		template_padded	= np.pad(template_crop, offset, mode = 'reflect')
		filtered_template = scipy.signal.fftconvolve(template_padded, kernel, mode = 'same')
		filtered_template = filtered_template[offset:-offset,offset:-offset]
		t3 = time()

		# padding the images
		images = images.reshape(images.shape[0], dims[0], dims[1])
		images_padded = pad_gpu(images, offset)
		t4 = time()

		# filtering images
		kernel = cp.asarray(kernel)		
		new_images = cp.zeros((images.shape[0], dims[0], dims[1]), dtype = np.float32)		
		filtered_images = low_pass_filter_space_gpu(images_padded, new_images, kernel, offset)
		images_padded = None
		t5 = time()

		# match template
		filtered_template = cp.asarray(filtered_template)		
		res_all = match_template(filtered_images, filtered_template, max_dev)		
		# max_loc     = np.zeros((images.shape[0], 2), dtype = np.int)
		# for i in range(images.shape[0]):
		#     res = res_all[i]
		#     max_loc[i] = np.array(np.unravel_index(np.argmax(res.flatten()), res.shape))
		t6 = time()

		# sh_x_n, sh_y_n = estimate_shifts(res_all, max_loc, max_dev)

		print("kernel for filtering", t2 - t1)
		print("preparing the template", t3 - t2)
		print("padding the images", t4 - t3)		
		print("filtering ", t5 - t4)
		print("match template", t6 - t5)
		# print("computing the shift", t7 - t6)
		# print("shifting the image", t8 - t7)
		# print("reshaping ", t9 - t8)		

		

print("Global motion correction", time()-t1)

movie_corrected = data['movie'].value.reshape(1000,480,752)

movie_original = data['original'].value.reshape(1000,480,752)

movie_compare = np.hstack((movie_corrected, movie_original))

mv = Movie(movie_compare)

# TO CLOSE THE WINDOW, PRESS Q
mv.play()

#close the hdf file
data.close()
