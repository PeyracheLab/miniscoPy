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
# video_info, videos, dims = get_video_info(fnames)
# hdf_mov       = get_hdf_file(videos, video_info, dims, parameters['save_original'])
# sys.exit()


hdf_mov = hd.File('example_movies/motion_corrected.hdf5', 'r+')
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
		make_corrections(hdf_mov['movie'], start_chunk, start_chunk+chunk_size, template, dims, parameters) 
		print(time()-t2)
		sys.exit()

print("Global motion correction", time()-t1)

