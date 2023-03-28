# functions for using tapping or clapping cue embedded accelerometer measurements to align sensor recordings to concert time.


# put this in an early cell of any notebook useing these functions, uncommented. With starting %
# %load_ext autoreload
# %autoreload 1
# %aimport al

import sys
import os
import time
import datetime as dt
import math
import numpy as np 
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import heartpy as hp

from scipy.signal import butter,filtfilt
from scipy import interpolate
from scipy.interpolate import interp1d

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def cue_template_make(peak_times,sf, t_range):
    # peak times is list of time points for onsets to clapping or tapping sequence, in seconds
    # sf is sample frequency in hz
    # buffer is duration of zeros before and after the peak times in generated template, in seconds
    
    peaks = np.array(peak_times)
    peaks = peaks-peaks[0] #assumes first peak is time zero
    c_start = t_range[0]
    c_end = t_range[1] 
    
    cue_sTime = np.linspace(c_start,c_end,sf*(c_end-c_start),endpoint=False)

    cue = pd.DataFrame()
    cue['sTime'] = cue_sTime
    cue['peaks'] = 0
    cue['taps'] = 0
    cue['claps'] = 0
    
    for pk in peak_times:
        cue.loc[find_nearest_idx(cue['sTime'],pk),'peaks'] = 1

    roll_par = int(0.05*sf)
    sum_par = int(0.02*sf)
    ewm_par = int(0.04*sf)
    cue['claps'] =2*cue['peaks'].ewm(span = ewm_par).mean()+ 0.6*cue['peaks'].rolling(roll_par, win_type='gaussian',center=True).sum(std=2)
    
    roll_par = int(0.02*sf)
    sum_par = int(0.02*sf)
    ewm_par = int(0.02*sf)
    cue['taps'] =1.5*cue['peaks'].ewm(span = ewm_par).mean()+ 0.5*cue['peaks'].rolling(roll_par, win_type='gaussian',center=True).sum(std=2)
    
    cue[cue.isna()] = 0
    return cue

def tap_cue_align(cue,sig_ex,sig_ID):
    # a function to take a segment of signal and the tapping cue to determing
    # the best shift value would allow alignment of signal to cue
    signal = sig_ex.copy()
        # make the signal excerpt corr compatible. Inclusing cutting the extreme peaks
    signal[signal.isna()] = 0
    M = signal.quantile(0.998)
    signal = signal/M
    signal[signal>1] = 1
    
    # cue must be sampled at the same steady rate as the signal exerpt
    sampleshift_s = cue.index.to_series().diff().median()
    length = np.min([len(signal),len(cue)])
    
    if signal.diff().abs().sum()<0.1: # signal is flat
        shifts.append(np.nan)
        print('sig_ex is too flat.')
        return
    else:
        fig = plt.figure(figsize=(15,6))
        ax1 = plt.subplot(311)
        signal.plot(label=sig_ID,ax=ax1,)
        cue.plot.line(y='cue',ax=ax1)
        ax1.set_title(sig_ID + ' synch alignment')
        ax1.legend()
        #plt.xlim(cue_range)
        
        ax2 = plt.subplot(312)
        CCC = ax2.xcorr(cue['cue'].iloc[:length], signal.iloc[:length], usevlines=True, maxlags=length-1, normed=True, lw=3)
        ax2.grid(True)
        ax2.set_xticklabels('')
        signal.index = signal.index + sampleshift_s*CCC[0][np.argmax(CCC[1])]
        
        ax1 = plt.subplot(313)
        signal.plot(label=sig_ID,ax=ax1,)
        cue.plot.line(y='cue',ax=ax1)
        #plt.xlim(cue_range)
        ax1.grid(True)
        ax1.set_title('shift '+ str(sampleshift_s*CCC[0][np.argmax(CCC[1])])+ ' s')
        #plt.saveas('')
        plt.show()

    shift_stats = {"s_corr0": CCC[1][CCC[0]==0][0], # alignment quality without adjustment,
                   "s_corr_offset": np.amax(CCC[1]),
                   "s_offset_samples": CCC[0][np.argmax(CCC[1])], # shifts
                   "s_offset_time": sampleshift_s*CCC[0][np.argmax(CCC[1])],
                   "Length_xcorr_samples": len(CCC[0]),
                   "Length_xcorr_time": len(CCC[0])*sampleshift_s,
                   "devID": sig_ID,
                   "auto_offset_time":sampleshift_s*CCC[0][np.argmax(CCC[1])],
                   "Full_CCC": CCC

    }
    return shift_stats
    