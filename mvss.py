# processing code to extract signals from movesense recordings
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
import json 
from scipy.signal import butter, filtfilt, argrelextrema
from scipy import interpolate
from scipy.interpolate import interp1d
import heartpy as hp

def json_extract(json_loc,json_filename,csvfile_destination):
    # a function that should open up a movesnse json 
    # and convert the files to csv in the correction destination folder
    
    # for json files formated as 510_Rsrch_MovesenseLog_2_2023-02-18_16_20_58.json
    d = json_filename.split('.')[0].split('_')
    devID =  int(d[0])
    SessN =  d[3]
    sigType = 'NA'
    fileSize = os.path.getsize(json_loc + json_filename)
    datet= dt.datetime.strptime(d[-4],'%Y-%m-%d') # date of recording
    recend_time = dt.datetime.strptime('-'.join(d[-4:]),'%Y-%m-%d-%H-%M-%S') # times in UTC

    # load the data
    tic = time.time()
    with open(json_loc + json_filename,'r') as f:
        data = json.loads(f.read())
    print([time.time()-tic,fileSize])
    print(data['Meas'].keys())
    
    signal_types = list(data['Meas'].keys())
    
    if 'ECG' in  signal_types:
        signal_type = 'ECG'
        print('Extracting ' + signal_type)
        
        ml_data = pd.json_normalize(data['Meas'][signal_type])
        print(' '.join([signal_type,str(len(ml_data)),str(time.time()-tic),'s']))
        Cols = ml_data.columns
        sig_cols = [x for x in Cols if not x.lower().startswith('time')]
        print(sig_cols)
                        
        # restructure timestamps
        # this format has no external clock reference except for the time stamp from the
        # recording stopping. 
        sample_per_timestamp = len(ml_data.loc[0,sig_cols[0]])
        print(sample_per_timestamp )
        step = ml_data['Timestamp'].diff().mean()/sample_per_timestamp
        times = ml_data['Timestamp']
        print(times.iloc[:3])
        TimeStamps = pd.DataFrame()
        for i in range(sample_per_timestamp):
            TimeStamps = pd.concat([TimeStamps,np.floor(times+i*step)],axis=0)
        TimeStamps = pd.to_numeric(TimeStamps[0], downcast='integer').sort_values(0).reset_index(drop=True)
        print(time.time()-tic)
        rec_dur = pd.to_timedelta(TimeStamps.iloc[-1]-TimeStamps.iloc[0],unit='ms')
        dt_timestamps = recend_time + pd.to_timedelta(TimeStamps,unit='ms')- pd.to_timedelta(TimeStamps.iloc[-1],unit='ms')
        print([recend_time,rec_dur,dt_timestamps.iloc[0]])
        
        out_df = pd.DataFrame(index = TimeStamps) # 
        out_df['dev_dTime'] = dt_timestamps.values
        sig_n =sig_cols[0]
        s = ml_data[sig_n].explode()
        s.index = TimeStamps.values
        out_df[signal_type] = s
        print([signal_type, len(out_df),len(out_df)/rec_dur.total_seconds(), time.time()-tic])


        fn = '_'.join([out_df.iloc[0,0].strftime('%Y-%m-%d_%H%M%S'),'MS',signal_type,d[0],'temp.csv'])
        out_df.to_csv(csvfile_destination+fn,index = False)
        print([fn,time.time()-tic])
        
    if 'IMU9' in  signal_types:
        signal_type = 'IMU9'
        print('Extracting ' + signal_type)
        ml_data = pd.json_normalize(data['Meas'][signal_type])
        Cols = ml_data.columns
        sig_cols = [x for x in Cols if not x.lower().startswith('time')]
        print(sig_cols)
                             
        # restructure timestamps
        # this format has no external clock reference except for the time stamp from the
        # recording stopping. 
        sample_per_timestamp = len(ml_data.loc[0,sig_cols[0]])
        print(sample_per_timestamp )
        step = ml_data['Timestamp'].diff().mean()/sample_per_timestamp
        times = ml_data['Timestamp']
        print(times.iloc[:3])
        TimeStamps = pd.DataFrame()
        for i in range(sample_per_timestamp):
            TimeStamps = pd.concat([TimeStamps,np.floor(times+i*step)],axis=0)
        TimeStamps = pd.to_numeric(TimeStamps[0], downcast='integer').sort_values(0).reset_index(drop=True)
        print(time.time()-tic)
        rec_dur = pd.to_timedelta(TimeStamps.iloc[-1]-TimeStamps.iloc[0],unit='ms')
        dt_timestamps = recend_time + pd.to_timedelta(TimeStamps,unit='ms')- pd.to_timedelta(TimeStamps.iloc[-1],unit='ms')
        print([recend_time,rec_dur,dt_timestamps.iloc[0]])
        
        # restructure measurements
        out_df = pd.DataFrame(index = TimeStamps)
        out_df['dev_dTime'] = dt_timestamps.values
        for sig_n in sig_cols:
            s = ml_data[sig_n].explode()
            s.index = TimeStamps.values
            sig_df = pd.json_normalize(s)
            for c in sig_df.columns:
                out_df['_'.join([sig_n[5:],c])] = sig_df[c].values
            print(time.time()-tic)    
        print([signal_type, len(out_df),len(out_df)/rec_dur.total_seconds(), time.time()-tic])

        fn = '_'.join([out_df.iloc[0,0].strftime('%Y-%m-%d_%H%M%S'),'MS',signal_type,d[0],'temp.csv'])
        out_df.to_csv(csvfile_destination+fn)
        print([fn,time.time()-tic])
    return