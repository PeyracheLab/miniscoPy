import numpy as np
from time import time
import scipy
import glob
import yaml
import sys,os
import h5py as hd
from time import time
import av

from miniscopy.base.motion_correction import *
from miniscopy import setup_cluster, CNMFE

#############################################################################################################
# DOWNLOADING AN EXAMPLE FILE IF NOT ALREADY PRESENT
#############################################################################################################
folder_name = 'example_movies'
files = glob.glob(folder_name+'/*.avi')
if len(files) == 0:
    import urllib.request
    url = "https://www.dropbox.com/s/0x3twp8bidl9svu/msCam1.avi?dl=1"
    with urllib.request.urlopen(url) as response, open(folder_name+"/msCam1.avi", 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    files = glob.glob(folder_name+'/*.avi')
    if len(files) == 0: print("No avi files found, please provide one at least")

#############################################################################################################
# LOADING PARAMETERS
#############################################################################################################
parameters = yaml.load(open(folder_name+'/parameters.yaml', 'r'))

#############################################################################################################
# start a cluster for parallel processing
#############################################################################################################
c, procs, n_processes = setup_cluster(backend='local', n_processes=8, single_thread=False)

# #############################################################################################################
# # MOTION CORRECTION | create the motion_corrected.hdf5 file
# #############################################################################################################
data, video_info = normcorre(files, procs, parameters['motion_correction'])


#############################################################################################################
# CONSTRAINED NON NEGATIVE MATRIX FACTORIZATION
#############################################################################################################
parameters['cnmfe']['init_params']['thresh_init'] = 1.2
parameters['cnmfe']['init_params']['min_corr'] = 0.8
parameters['cnmfe']['init_params']['min_pnr'] = 1.5

cnm = CNMFE(data, parameters['cnmfe'])

cnm.fit(procs)
#############################################################################################################
# VISUALIZATION
#############################################################################################################
cn, pnr = cnm.get_correlation_info()

dims = cnm.dims
C = cnm.C.value.copy()
A = cnm.A.value.copy()

# A is normalized to 1 for display
A -= np.vstack(A.min(1))
A /= np.vstack(A.max(1))
Atotal = A.sum(0).reshape(dims)


from matplotlib.pyplot import *
import matplotlib.gridspec as gridspec

tmp = Atotal.copy()
tmp[tmp == 0] = np.nan
figure(figsize = (15,5))
gs = gridspec.GridSpec(3,3)
subplot(gs[0:2,0])
imshow(Atotal)
subplot(gs[0:2,1])
imshow(cn)
contour(np.flip(tmp, 0), origin = 'upper', cmap = 'gist_gray')
subplot(gs[0:2,2])
imshow(cn)
contour(np.flip(tmp, 0), origin = 'upper', cmap = 'gist_gray')
subplot(gs[-1,:])
plot(C)

show()
