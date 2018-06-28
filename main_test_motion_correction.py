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
from miniscopy.base.motion_correction import normcorre
from miniscopy import setup_cluster, CNMFE
from miniscopy import Movie



if __name__ == '__main__':
	#############################################################################################################
	# DOWNLOADING AN EXAMPLE FILE IF NOT ALREADY PRESENT
	#############################################################################################################
	folder_name = 'bad_movie/exemple1'
	files = glob.glob(os.path.join(folder_name, '*msCam*.avi'))
	if len(files) == 0:
	    import urllib.request
	    url = "https://www.dropbox.com/s/oqme4771hlf95ds/A0609_msCam1.avi?dl=1"
	    with urllib.request.urlopen(url) as response, open(folder_name+"/A0609_msCam1.avi", 'wb') as out_file:
	        data = response.read()
	        out_file.write(data)
	    files = glob.glob(folder_name+'/*.avi')
	    if len(files) == 0: 
	    	print("No avi files found, please provide one at least")


	#############################################################################################################
	# LOADING PARAMETERS
	#############################################################################################################
	parameters = yaml.load(open(os.path.join(folder_name, 'parameters.yaml'), 'r'))

	#############################################################################################################
	# start a cluster for parallel processing
	#############################################################################################################
	c, procs, n_processes = setup_cluster(backend='local', n_processes=8, single_thread=False)

	#############################################################################################################
	# MOTION CORRECTION | create the motion_corrected.hdf5 file
	#############################################################################################################
	# Here we update the parameters to
	parameters_glob = yaml.load(open(folder_name+'/parameters.yaml', 'r'))
	parameters = parameters_glob['motion_correction']

	# This parameter controls the number of times the whole movie is corrected
	parameters['nb_round'] = 1

	# This parameter allows us to save the original movie to compare with the corrected one
	parameters['save_original'] = True

	# The main function
	data, video_info = normcorre(files, procs, parameters)

	##############################################################################################################
	# PLAYING THE MOVIE
	##############################################################################################################
	# here we play the original movie along with the corrected movie
	# DO NOT TRY WITH BIG MOVIE. THAT WILL GENERATE A MEMORY ERROR

	movie_corrected = data['movie'].value.reshape(1000,480,752)

	movie_original = data['original'].value.reshape(1000,480,752)

	movie_compare = np.hstack((movie_corrected, movie_original))

	mv = Movie(movie_compare)

	# TO CLOSE THE WINDOW, PRESS Q
	mv.play()

	#close the hdf file
	data.close()

	# terminate the cluster
	procs.terminate()
