import numpy as np
from matplotlib.pyplot import *
import pandas as pd
import sys, tempfile
import cv2
import brian2.only as br2
from IPython.core.debugger import Pdb
import h5py as hd
import xgboost as xgb

from miniscopy.base.motion_correction import *
from miniscopy import setup_cluster, CNMFE, generate_data, get_default_parameters



########################################################################################################
# GENERATE AN HD SIGNAL
########################################################################################################
# simulate a 100 Hz camera recording for 10 second
bin_size = 10 #ms
duration = 10000 #ms
hd_signal = pd.Series(index = np.arange(0, duration, bin_size), data = 0)
# the agent rotates on average at pi/2/s
for i in range(1,len(hd_signal)):
    hd_signal.iloc[i] = hd_signal.iloc[i-1] + np.random.normal((np.pi/2)*(bin_size/1000),1)
hd_signal = hd_signal.rolling(center=True, window=10).mean().fillna(0)
hd_signal = hd_signal%(2*np.pi)

########################################################################################################
# GENERATE SPIKING ACTIVITY WITH BRIAN 
########################################################################################################
def get_tuning_curves(n, nb_bins):    
    x   = np.linspace(0, 2*np.pi, nb_bins+1)
    phi = np.linspace(0, 2*np.pi, n+1)[0:-1]
    A   = np.random.uniform(10, 50,n)
    B   = np.random.uniform(5, 10, n)
    C   = np.random.uniform(0, 2, n)
    return pd.DataFrame(index = x, columns = np.arange(n), data = C+A*np.exp(B*np.cos(np.vstack(x) - phi))/np.exp(B))

N = 20 # number of neurons
tcurves     = get_tuning_curves(N, nb_bins = 60)
freq_steps  = tcurves.reindex(hd_signal, method = 'nearest').values
network     = br2.Network()
stim_hd     = br2.TimedArray(freq_steps*br2.Hz, dt = float(bin_size)*br2.ms)
network.add(br2.PoissonGroup(N, rates = 'stim_hd(t, i)'))
network.add(br2.SpikeMonitor(network.objects[0]))
network.run(duration*br2.ms, report = 'text')
spikes      = network.objects[2].spike_trains()
del network

# the firing rate 
frate       = pd.DataFrame(index = hd_signal.index)
for k in spikes.keys():    
    frate[k], bins_edge = np.histogram(spikes[k]/br2.ms, int(duration/bin_size), range = (0, duration))

########################################################################################################
# GENERATE THE CALCIUM MOVIE IN HDF5
########################################################################################################
dims = (480, 752)
movie, true_data = generate_data(frate.values, N, duration//bin_size, dims, framerate = 1/(bin_size*1e-3))

# create a temporary hdfile
# path = tempfile.mkdtemp()
path = '/mnt/DataGuillaume/MINISCOPE/test_hd_video'
filename = path + '/test_hd.hdf5'
data = hd.File(filename, "w")
data.create_dataset('movie', data = movie.astype('float32'), chunks=True)
# need to set some attributes to match the type of file created by the motion correction algorithm 
data.attrs['folder']            = path
data.attrs['filename']          = filename
data['movie'].attrs['duration'] = movie.shape[0]
data['movie'].attrs['dims']     = dims

# path = '/mnt/DataGuillaume/MINISCOPE/test_hd_video'
# filename = path + '/test_hd.hdf5'
# data = hd.File(filename, "r+")
# del data['cnmfe']
# del data['patches']

########################################################################################################
# RUN CNMFE
########################################################################################################
procs = None
# c, procs, n_processes = setup_cluster(backend='local', n_processes=8, single_thread=False)
np.seterr(all='raise')
parameters = get_default_parameters('cnmfe')
# need to change thresh_init to not supress too much pixels when initiating
# do not use for real data unless you know why
parameters['init_params']['thresh_init'] = 0.5
parameters['init_params']['min_pnr'] = 1.5
parameters['init_params']['min_corr'] = 0.6
parameters['init_params']['gSig'] = (4,4)
parameters['init_params']['gSiz'] = (12,12)
parameters['init_params']['filter_data_centering'] = False
# parameters['init_params']['ssub'] = 4
# parameters['init_params']['tsub'] = 4

cnm = CNMFE(data, parameters)
cnm.fit(procs)
sys.exit()

# need to sort the cnmfe neurons by the position of the centers
true_C = true_data['C']
new_C = np.zeros_like(true_C)
idx_sorted = np.zeros(len(cnm.center), dtype=np.int)
for i in range(len(cnm.center)):
    idx = np.sqrt(np.power(cnm.center[i] - true_data['center'],2).sum(1)).argmin()
    idx_sorted[i] = idx
    new_C[:,idx] = cnm.C[:,i]

# normalize each trace between 0 and 1 to compare
new_C -= new_C.min(0)
new_C /= new_C.max(0)
true_C -= true_C.min(0)
true_C /= true_C.max(0)


########################################################################################################
# DECODE HD WIDTH XGBOOST 
########################################################################################################
from sklearn.model_selection import KFold
import xgboost as xgb

def xgb_decodage(Xr, Yr, Xt):      
    params = {'objective': "multi:softprob",
    'eval_metric': "mlogloss", #loglikelihood loss
    'seed': 2925, #for reproducibility
    'silent': 1,
    'learning_rate': 0.01,
    'min_child_weight': 2, 
    'n_estimators': 1000,
    # 'subsample': 0.5,
    'max_depth': 5, 
    'gamma': 0.5,
    'num_class':60}
    
    # binning Yr in 60 classes
    bins = np.linspace(0, 2*np.pi+1e-8, 61)
    clas = np.digitize(Yr, bins).flatten()-1
    x = bins[0:-1] + (bins[1]-bins[0])/2.    
    dtrain = xgb.DMatrix(Xr, label=clas)
    dtest = xgb.DMatrix(Xt)

    num_round = 100
    bst = xgb.train(params, dtrain, num_round)
    
    ymat = bst.predict(dtest)

    return x[np.argmax(ymat,1)]

def fit_cv(X, Y, n_cv=10):
    """
        The function to do the cross-validation
    """
    if np.ndim(X)==1:
        X = np.transpose(np.atleast_2d(X))
    cv_kf = KFold(n_splits=n_cv, shuffle=True, random_state=42)
    skf  = cv_kf.split(X)    
    Y_hat=np.zeros(len(Y))    
    
    for idx_r, idx_t in skf:        
        Xr = X[idx_r, :]
        Yr = Y[idx_r]
        Xt = X[idx_t, :]
        Yt = Y[idx_t]           
        Yt_hat = xgb_decodage(Xr, Yr, Xt)        
        Y_hat[idx_t] = Yt_hat
        
    return Y_hat

# cnm.C is put in a panda dataframe that should become soon a neuroseries object
C = pd.DataFrame(new_C, columns = np.arange(N))

hd_predicted = fit_cv(C.values, hd_signal.values, n_cv = 4)
hd_predicted = pd.Series(data = hd_predicted, index = hd_signal.index)


