# -*- coding: utf-8 -*-
""" A set of pre-processing operations in the input dataset:

1. Interpolation of missing data
2. Indentification of saturated pixels
3. Estimation of noise level for each imaged voxel
4. Estimation of global time constants

See Also:
------------

@authors: agiovann epnev
@image docs/img/greedyroi.png
"""
#\package caiman/source_extraction/patchf
#\version   1.0
#\copyright GNU General Public License v2.0
#\date Created on Tue Jun 30 21:01:17 2015

import numpy as np
import cv2
import itertools

from IPython.core.debugger import Pdb


def get_noise_fft(patch, max_num_samples_fft=3072, noise_range=[0.25,0.5], noise_method='logmexp', **kwargs):
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
	duration 	= patch.duration	
	
	if duration > max_num_samples_fft:
		# take first part, middle and end part
		duration = max_num_samples_fft
		idx_frames = np.concatenate((np.arange(1,max_num_samples_fft // 3 + 1),
									np.arange(duration//2 - max_num_samples_fft //6,duration//2 + max_num_samples_fft //6),
									np.arange(duration-max_num_samples_fft//3,duration)))
	else:
		idx_frames = np.arange(duration)
	
	ff 			= np.arange(0, 0.5 + 1. / duration, 1. / duration)
	ind1 		= ff > noise_range[0]
	ind2 		= ff <= noise_range[1]
	ind 		= np.logical_and(ind1, ind2)
	psdx 		= np.zeros((patch.Y.shape[1],ind.sum()))
	# Pdb().set_trace()
	
	for i in range(0, patch.Y.shape[1], patch.chunks[1]): # doing it row by row
		data = patch.patch_group.parent.parent['movie'][:,patch.xy[i:i+patch.chunks[1]]]
		patch.Y[:,i:i+patch.chunks[1]] = data
		sample = data[idx_frames]
		dft = np.fft.fft(sample, axis = 0)[:len(ind)][ind].T
		psdx[i:i+patch.chunks[1]]	= (1./duration) * (dft.real * dft.real)+(dft.imag*dft.imag)

	if noise_method == 'mean':
		noise = np.sqrt(np.mean(psdx, axis=1))
	elif method == 'median':
		noise = np.sqrt(np.median(psdx, axis=1))
	else:
		noise = np.log(psdx + 1e-10)
		noise = np.mean(noise, axis=1)
		noise = np.exp(noise)
		noise = np.sqrt(noise)

	return noise.reshape(patch.dims)
	
def interpolate_missing_data(movie):
	"""
	Interpolate any missing data using nearest neighbor interpolation.
	Missing data is identified as entries with values NaN

	Parameters:
	----------
	Y   np.ndarray (3D)
		

	Returns:
	------
	Y   np.ndarray (3D)
		movie, data with interpolated entries (d1 x d2 x T)
	coor list
		list of interpolated coordinates

	Raise:
	------
		Exception('The algorithm has not been tested with missing values (NaNs). Remove NaNs and rerun the algorithm.')
	"""	
	if np.any(np.isnan(movie)):
		duration = len(movie)				
		for x in range(dims[0]):
			for y in range(dims[1]):
				pixel = movie[:,x,y]
				if np.any(np.isnan(pixel)):
					nans_index = np.where(np.isnan(pixel))[0]
					nnans_index = np.where(~np.isnan(pixel))[0]
					pixel[nans_index] = np.interp(nans_index, nnans_index, pixel[nnans_index])
				movie[:,x,y] = pixel[:]		
	return

def preprocess_data(patch, procs, check_nan, **kwargs):
	"""	

	mainly update spectral noise

	"""
	if check_nan:
		print("TODO : interpolate missing data")		
		sys.exit()
		interpolate_missing_data(movie)
	
	patch.sn = get_noise_fft(patch, **kwargs)	
	return

