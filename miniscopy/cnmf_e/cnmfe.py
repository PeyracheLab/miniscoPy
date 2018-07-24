# -*- coding: utf-8 -*-

"""
	Constrained Nonnegative Matrix Factorization
"""
import numpy as np
import scipy.sparse as spr
from scipy.sparse.csgraph import connected_components
from IPython.core.debugger import Pdb
import sys,os
from scipy.ndimage import center_of_mass
from scipy.stats import pearsonr
import scipy.linalg
from tqdm import tqdm
import h5py as hd

from time import time       

from .utilities import compute_residuals, normalize_AC, copy_data
from .pre_processing import preprocess_data
from .initialization import initialize_components, get_noise_fft, local_correlations_fft
from .deconvolution import constrained_foopsi
from .temporal import update_temporal_components
from .spatial import update_spatial_components

class Patch(object):

	def __init__(self, group, duration, patch_number, patch_index, parameters):
		self.patch_group    = group
		self.parameters     = parameters
		self.duration       = duration
		self.patch_number   = patch_number      
		self.patch_index    = patch_index
		self.x,self.y       = np.where(patch_index) # 2d position
		self.xy             = np.ravel_multi_index((self.x, self.y), patch_index.shape) # 1d position
		self.dims           = (patch_index.sum(0).max(), patch_index.sum(1).max())      
		self.b0             = None
		self.B              = None # The background noise
		self.CdotA          = None # I work in row so inverted compared to matlab
		self.AA             = None 
		self.sn             = np.zeros(self.dims) # The spectral noise for each pixel computed from opencv dft
		self.center         = None # The center points of the neurons
		self.g              = None
		self.YrA            = None # the residuals
		self.count_bgr      = None
		self.count_nrs      = None
		# # create the patch movie      
		self.Y              = self.patch_group.create_dataset('Y', shape = (self.duration, np.prod(self.dims)), dtype = np.float32, chunks = (32,self.dims[1]))
		self.chunks         = self.Y.chunks     
		return
	
	def fit(self, procs):       
		#################################################################################################
		# 1. Preprocess data / Compute spectral noise for each pixel
		#################################################################################################           		
		preprocess_data(self, procs, **self.parameters['preprocess_params'])
				
		#################################################################################################
		# 2. Initialize components / CALL greedyROI_corr in CAIMAN PACKAGE
		#################################################################################################                           
		initialize_components(self, **self.parameters['init_params'])
		
		#################################################################################################
		# 3. Compute residuals on full size patch
		#################################################################################################                           
		compute_residuals(self)

		#################################################################################################
		# 4. Normalize AC
		#################################################################################################       
		normalize_AC(self)
				
		return

class CNMFE(object):

	def __init__(self, file, parameters):       
		self.file           = file      # the hdf5 file of the original movie corrected for motion
		self.filename       = self.file.attrs['filename']
		self.parameters     = parameters
		self.Y              = file['movie']
		self.duration       = file['movie'].attrs['duration']
		self.dims           = tuple(file['movie'].attrs['dims'])                
		self.patch_object   = {}
		self.patch_index    = None # The patch index in 2d      
		self.B              = None # The background noise       
		self.W              = None
		self.CdotA          = None # I work in row so inverted compared to matlab
		self.AA             = None 
		self.sn             = np.zeros(self.dims) # The spectral noise for each pixel computed from opencv dft
		self.center         = None # The center points of the neurons
		self.g              = None
		self.count_nrs      = 0
		self.count_bgr      = 0
		self.C_file         = None
		self.A_file         = None
		self.YrA_file       = None      
		self.bf             = None
		# every patch array will be stored in the group patches in the hdf5 self.file
		self.patch_group    = self.file.create_group('patches')
		self.cnmfe_group    = self.file.create_group('cnmfe')
		return

	def get_patch_index(self, nb_patch, overlaps, **kwargs):
		total_patch = np.prod(nb_patch)
		self.patch_index = np.zeros((total_patch,)+self.dims, dtype= np.bool)
		x_start = np.linspace(0, self.dims[0], nb_patch[0]+1).astype(np.int)
		y_start = np.linspace(0, self.dims[1], nb_patch[1]+1).astype(np.int)
		count = 0       
		for i in range(len(x_start[0:-1])):
			for j in range(len(y_start[0:-1])):                             
				xs = np.maximum(0, x_start[i]-overlaps//2)
				xe = np.minimum(self.dims[0], x_start[i+1]+overlaps//2)
				ys = np.maximum(0, y_start[j]-overlaps//2)
				ye = np.minimum(self.dims[1], y_start[j+1]+overlaps//2)             
				self.patch_index[count][xs:xe,ys:ye] = 1.0
				count += 1
		return

	def fit_patch(self, procs):     
		# first the patch class and patch group
		for i in tqdm(range(len(self.patch_index))):        
			group = self.patch_group.create_group('patch_'+str(i)) # subgroup of the group patches
			self.patch_object[i] = Patch(group, self.duration, i, self.patch_index[i], self.parameters)
			
			# the dataset of the patch movie is instantiated but not copied from the original movie
			# it is done when the first function get_noise_fft is called with chunking
			# todo find a way to do that in parrallel with h5py
			# group.create_dataset('Y', shape = (self.duration, self.patch_object[i].xy.shape[0]), dtype = np.float32, chunks = (32,self.patch_object[i].dims[1]))
			# self.patch_object[i].patch_group = group
			# args.append([self.filename, 'movie', group.name, 'Y', self.patch_object[i].xy, (32,self.patch_object[i].dims[1])])            
			# args.append([group, self.file['movie'], self.patch_object[i].xy, 'Y', (32,self.patch_object[i].dims[1])])

		# # copying the patch movie in the patch group
		# for a in args:
		#   copy_data(*a)
		# # _ = procs.starmap_async(copy_data, args).get()

		# fitting it
		for i in tqdm(range(len(self.patch_index))):            
			self.patch_object[i].fit(procs)         
		return

	def assemble_patch(self, procs):        
		d = np.prod(self.dims)
		patch_centers       = np.zeros((len(self.patch_index), 2))
		for i in range(len(patch_centers)): patch_centers[i] = np.array(center_of_mass(self.patch_object[i].patch_index))
		global_neuron_position = dict.fromkeys(self.patch_object)       
		patch_with_neurons = []
		keep_per_patch = dict.fromkeys(self.patch_object)
		
		for i in tqdm(self.patch_object.keys()):
			patch   = self.patch_object[i]          
			self.count_bgr += patch.b.shape[0]          
			# deleting duplicates           
			global_neuron_position[i] = []
			keep = []
			for j in range(patch.A.shape[0]): # loop through neurons                
				frame = patch.A[j]
				neuron_center = np.array(center_of_mass(frame.reshape(patch.dims)))
				tmp = neuron_center- np.array(patch.dims)/2 + patch_centers[i]
				dist = np.linalg.norm(patch_centers - tmp, keepdims = True, axis = 1)
				if dist.argmin() == i: # if the neuron is close to the ccenter of the current patch
					global_neuron_position[i].append(tmp)
					keep.append(j)          
			keep_per_patch[i] = np.array(keep)
			if len(keep):               
				patch_with_neurons.append(i)
				global_neuron_position[i] = np.vstack(global_neuron_position[i])                
				self.count_nrs += len(keep)         
						
		self.A      = self.cnmfe_group.create_dataset('A', shape = (self.count_nrs, d), chunks = True) # The spatial footprint of size (d,K) with d = (h*w) # python style in lines
		self.C      = self.cnmfe_group.create_dataset('C', shape = (self.duration, self.count_nrs), chunks = True)  # The calcium activities of K neurons of size (K,T) 
		self.YrA    = self.cnmfe_group.create_dataset('YrA', shape = (self.duration, self.count_nrs), chunks = True)  # The calcium activities of K neurons of size (K,T) 
		self.S      = self.cnmfe_group.create_dataset('S', shape = (self.duration, self.count_nrs), chunks = True) # The spiking activity
		self.b      = self.cnmfe_group.create_dataset('b', shape = (self.count_bgr, d), chunks = True) # The background fluorescence
		self.f      = self.cnmfe_group.create_dataset('f', shape = (self.duration, self.count_bgr), chunks = True) # the background noise in time       
		
		self.center = np.zeros((self.count_nrs,2))
		start = 0
		for i in tqdm(patch_with_neurons):                                  
			patch   = self.patch_object[i]          
			keep    = keep_per_patch[i]         
			self.A[start:start+len(keep),patch.xy] = patch.A[keep,:]
			self.C[:,start:start+len(keep)] = patch.C[:,keep] 
			self.YrA[:,start:start+len(keep)] = patch.YrA[:,keep]
			self.center[start:start+len(keep),:] = global_neuron_position[i]            
			start += len(keep)
		
		
		# normalizing A | not sure why here         
		# Im = 1./self.patch_index.sum(0).flatten()
		# for n in range(self.count_nrs):
		#   self.A[n,:] = (self.A[n,:] / Im)[:]
		
		# generating background
		start = 0
		for i in tqdm(self.patch_object.keys()):
			patch = self.patch_object[i]            
			tmp = np.zeros((patch.b.shape[0],self.b.shape[1]))
			tmp[:,patch.xy] = patch.b[:,:]
			self.b[start:start+patch.b.shape[0],:] = tmp
			self.f[:,start:start+patch.b.shape[0]] = patch.f[:,:]
			start += patch.b.shape[0]

		
		#Compressing background components with a low rank NMF
		# for n in range(self.count_bgr):
		#   self.b[n,:] = (self.b[n,:] / Im)[:]

		

		f = np.hstack((np.mean(self.f, 1)[:,np.newaxis], np.random.rand(self.duration, self.parameters['temporal_params']['nb']-1)))
					
		for _ in tqdm(range(10)):
			f = f/np.sqrt(np.power(f, 2.0).sum(0))
			try:                
				ffI = np.linalg.inv(f.T.dot(f))
				fTdotf5 = np.zeros((f.shape[1],self.count_bgr))
				chunk_size_row = self.f.chunks[0]
				for i in range(0, self.duration+chunk_size_row,chunk_size_row):
					fTdotf5[:,:] += f[i:i+chunk_size_row,:].T.dot(self.f[i:i+chunk_size_row,:])
				fTdotf5dotb = np.zeros((f.shape[1],d))
				chunk_size_row = self.b.chunks[1]
				for i in range(0, d+chunk_size_row, chunk_size_row):
					fTdotf5dotb[:,i:i+chunk_size_row] += fTdotf5.dot(self.b[:,i:i+chunk_size_row])
				b = np.fmax(ffI.dot(fTdotf5dotb), 0)
							
			except np.linalg.LinAlgError: # singular matrix
				print("TODO HERE")
				sys.exit()
				# b = np.fmax(spr.csr_matrix(scipy.linalg.lstsq(f, self.f)[0]).dot(self.b), 0)
			
			try:
				bb = np.zeros((self.count_bgr, b.shape[0]))
				chunk_size_col = self.b.chunks[1]
				for i in range(0, d+chunk_size_col, chunk_size_col):
					bb[:,:] += self.b[:,i:i+chunk_size_col].dot(b[:,i:i+chunk_size_col].T)                                          
				bbI = np.linalg.inv(b.dot(b.T))
				# lazy
				f = self.f.value.dot(bb).dot(bbI)               
			except np.linalg.LinAlgError: # singular matrix
				print("TODO HERE")
				sys.exit()
				# f = self.f.dot(scipy.linalg.lstsq(b.T, self.b.T).T)


		nB = np.sqrt(np.power(b, 2.0).sum(1))
		b = b / nB[:,np.newaxis]
		f = f * nB

		self.b.resize(b.shape)
		self.b[:] = b[:]
		self.f.resize(f.shape)
		self.f[:] = f[:]
		
		return
				
	def merge_components(self, procs, thr, **kwargs):                   
		nr = self.count_nrs
		d = np.prod(self.dims)
		merged_ROIs = [0]
		while len(merged_ROIs) > 0:
			# % find graph of overlapping spatial components            
			A_corr = np.zeros((nr,nr))
			chunk_size_row = self.A.chunks[1]
			for i in range(0, d+chunk_size_row, chunk_size_row):
				data = self.A[:,i:i+chunk_size_row]
				A_corr[:,:] += data.dot(data.T)[:,:]
			
			np.fill_diagonal(A_corr, 0)         
			FF2 = A_corr > 0
			C_corr = np.zeros(A_corr.shape)
			for ii in range(nr):
				overlap_indices = np.where(A_corr[ii])[0]               
				if len(overlap_indices) > 0:
					# we chesk the correlation of the calcium traces for each overlapping components                                        
					trace = self.C[:,ii]
					corr_values = np.zeros(len(overlap_indices))
					for t, jj in enumerate(overlap_indices):
						trace2 = self.C[:,jj]
						corr_values[t] = pearsonr(trace, trace2)[0]
					C_corr[ii, overlap_indices] = corr_values

			FF1 = (C_corr + C_corr.T) > thr
			FF3 = FF1*FF2
			
			nb, connected_comp = connected_components(FF3)  # % extract connected components
			
			list_conxcomp = []
			for i in range(nb):  # we list them
				if np.sum(connected_comp == i) > 1:
					list_conxcomp.append((connected_comp == i).T)
			list_conxcomp = np.asarray(list_conxcomp).T

			if list_conxcomp.ndim > 1:
				cor = np.zeros((np.shape(list_conxcomp)[1], 1))
				for i in range(np.size(cor)):
					fm = np.where(list_conxcomp[:, i])[0]
					for j1 in range(np.size(fm)):
						for j2 in range(j1 + 1, np.size(fm)):
							cor[i] = cor[i] + C_corr[fm[j1], fm[j2]]

		
				if np.size(cor) > 1:
					# we get the size (indeces)
					ind = np.argsort(np.squeeze(cor))[::-1]
				else:
					ind = [0]

				# nbmrg = min((np.size(ind), mx))   # number of merging operations
				nbmrg = np.size(ind)
				
				# we initialize the values
				A_merged = np.zeros((nbmrg,d))
				C_merged = np.zeros((self.duration, nbmrg))
				S_merged = np.zeros((self.duration, nbmrg))
				bl_merged = np.zeros((nbmrg, 1))
				c1_merged = np.zeros((nbmrg, 1))
				sn_merged = np.zeros((nbmrg, 1))
				g_merged = np.zeros((nbmrg, self.parameters['temporal_params']['p']))
				merged_ROIs = []

				for i in range(nbmrg):
					merged_ROI = np.where(list_conxcomp[:, ind[i]])[0]
					merged_ROIs.append(merged_ROI)

					# we l2 the traces to have normalization values
					Ctmp = self.C[:,merged_ROI]
					Atmp = self.A[merged_ROI,:]

					C_to_norm = np.sqrt(np.sum(np.power(Ctmp, 2.0), 0))                 
		
					# from here we are computing initial values for C and A                 
					# this is a  big normalization value that for every one of the merged neuron
					C_to_norm = np.sqrt(np.ravel(np.power(Atmp, 2.0).sum(axis=1)) * np.sum(Ctmp ** 2, axis=0))
					indx = np.argmax(C_to_norm)

					# we normalize the values of different A's to be able to compare them efficiently. we then sum them                 
					tmp = C_to_norm[np.newaxis,:].dot(np.eye(len(C_to_norm)))
					computedA = tmp.dot(Atmp).sum(0) 
					
					# we operate a rank one NMF, refining it multiple times (see cnmf demos )
					for _ in range(10):
						computedC = np.maximum(Ctmp.dot(Atmp.dot(computedA.T)) / (computedA.dot(computedA.T)), 0)
						computedA = np.maximum(Atmp.T.dot(Ctmp.T.dot(computedC)).T / (computedC.dot(computedC.T)), 0)

					# then we de-normalize them using A_to_norm
					A_to_norm = np.sqrt(computedA.dot(computedA.T) / np.power(Atmp, 2.0).sum(1).max())
					computedA /= A_to_norm
					computedC *= A_to_norm

					# we then compute the traces ( deconvolution ) to have a clean c and noise in the background
					computedC, bm, cm, gm, sm, ss, lam_ = constrained_foopsi(np.array(computedC).squeeze(), g=None, **self.parameters['temporal_params'])

					A_merged[i] = computedA
					C_merged[:,i] = computedC
					S_merged[:,i] = ss[0:self.duration]
					bl_merged[i] = bm
					c1_merged[i] = cm
					sn_merged[i] = sm
					g_merged[i, :] = gm
				
				# we want to remove merged neuron from the initial part and replace them with merged ones
				neur_id = np.unique(np.hstack(merged_ROIs))
				good_neurons = np.setdiff1d(list(range(nr)), neur_id)
				nb_new = A_merged.shape[0]
				A_new = np.zeros((np.size(good_neurons)+nb_new, d))
				chunk_size_row = self.A.chunks[1]
				start = 0
				for m, n in enumerate(good_neurons):
					for i in range(0, d+chunk_size_row, chunk_size_row):
						A_new[m,i:i+chunk_size_row] = self.A[n,i:i+chunk_size_row]
					start = m+1
				A_new[start:,:] = A_merged[:,:]
				self.A.resize(A_new.shape)
				self.A[:] = A_new[:]                
				
				C_new = np.zeros((self.duration, np.size(good_neurons)+nb_new))             
				chunk_size_col = self.C.chunks[0]               
				start = 0
				for m, n in enumerate(good_neurons):                
					for i in range(0, self.duration+chunk_size_col, chunk_size_col):
						C_new[i:i+chunk_size_col,m] = self.C[i:i+chunk_size_col,m]
					start = m+1
				C_new[:,start:] = C_merged[:,:]
				self.C.resize(C_new.shape)
				self.C[:] = C_new[:]
				
				nr = nr - len(neur_id) + nbmrg

			else:
				print('No neurons merged!')
				merged_ROIs = []
			
		self.count_nrs = nr     
		return

	def evaluate(self, procs):  
		"""
			TODO go in parrallel here
		"""                             
		# update temporal               
		self.C[...] = update_temporal_components(self.Y, self.A, self.C, **self.parameters['temporal_params'])   

		# update spatial
		self.parameters['spatial_params']['se'] = np.ones((1,)*len(self.dims), dtype = np.uint8)                
		self.A[...], self.C[...] = update_spatial_components(self.Y, self.A, self.C, None, self.sn, **self.parameters['spatial_params'])

		# update temporal       
		self.C[...] = update_temporal_components(self.Y, self.A, self.C, **self.parameters['temporal_params'])   
		
		# normalize AC      
		normalize_AC(self, doYrA = False)
		
		return

	def fit(self, procs=None):
		"""
		"""             
		from time import time
		start = time()
		#################################################################################################
		# 1. Get the patches positions
		#################################################################################################               
		self.get_patch_index(**self.parameters['patch_params'])

		#################################################################################################
		# 2. Fit for each patch by calling the class Patch
		#################################################################################################               
		self.fit_patch(procs)

		#################################################################################################
		# 3. Assemble the patch together
		#################################################################################################               
		self.assemble_patch(procs)      

		#################################################################################################
		# 4. Merge taking best neuron
		################################################################################################                                
		self.merge_components(procs, **self.parameters['merge_params'])

		#################################################################################################
		# 5. Reevaluate
		################################################################################################                        
		self.evaluate(procs)

		print("Total time : ", np.round(time() - start, 4), " second")

		return

	def get_correlation_info(self, filter_ = False):
		import cv2
		dims 	= self.dims
		chunk_size  = self.Y.chunks[0]       

		if 'filtered_movie' in self.cnmfe_group.keys():
			data_filtered = self.cnmfe_group['filtered_movie']
		else:
			data_filtered  = self.cnmfe_group.create_dataset('filtered_movie', shape = (self.duration, dims[0], dims[1]), chunks = (chunk_size,dims[0],dims[1]))
				
		if filter_:
			gSig = self.parameters['init_params']['gSig']
			ksize = tuple((3*np.array(gSig)) // 2 * 2 + 1)
		
		
		data_filtered_mean = np.zeros(dims) # in case it's use

		for i in tqdm(range(0, self.duration+chunk_size, chunk_size)):
			data = self.Y[i:i+chunk_size,:]			
			for j, frame in enumerate(data):    
				if filter_:    
					if self.parameters['init_params']['center_psf']:
						tmp = cv2.GaussianBlur(frame.reshape(dims), ksize = ksize, sigmaX = gSig[0], sigmaY = gSig[1], borderType=1)
						tmp2 = cv2.boxFilter(frame.reshape(dims), ddepth=-1, ksize = ksize, borderType = 1)
						data_filtered[i+j] = tmp - tmp2
					else:
						tmp = cv2.GaussianBlur(frame.reshape(dims), ksize = ksize, sigmaX = gSig[0], sigmaY = gSig[1], borderType=1)
						data_filtered[i+j] = tmp
				else:
					data_filtered[i+j] = frame.reshape(dims)
				
				data_filtered_mean += frame.reshape(dims)

		data_filtered_mean /= float(self.duration)
		
		# compute peak-to-noise ratio    
		if self.parameters['init_params']['filter_data_centering']:
			for i in range(0, self.duration+chunk_size, chunk_size):
				data_filtered[i:i+chunk_size] = data_filtered[i:i+chunk_size] - data_filtered_mean
					
		data_max = np.max(data_filtered, axis=0)    
		
		noise_pixel = get_noise_fft(data_filtered)

		pnr = np.divide(data_max, noise_pixel)
		
		# compute correlation image
		cn = local_correlations_fft(data_filtered)

		return cn, pnr





		


