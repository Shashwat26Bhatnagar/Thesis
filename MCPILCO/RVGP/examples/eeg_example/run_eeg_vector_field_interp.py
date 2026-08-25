#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import os
import wget
import requests
import subprocess

from scipy.io import savemat

from eeg_utils import load_eeg_data, compute_error_time
from eeg_utils import interpolate_time_range
from eeg_utils import compute_vectorfield_features_time

np.random.seed(0)


directory = "./data"

files = {'s1':'https://dataverse.harvard.edu/api/access/datafile/10649581',
         's2':'https://dataverse.harvard.edu/api/access/datafile/10649580',
         's5':'https://dataverse.harvard.edu/api/access/datafile/10649579'}


for patient_id in files.keys():
    
    sub_directory = os.path.join(directory, f"{patient_id}")
    os.makedirs(sub_directory, exist_ok=True)

    file_path = os.path.join(sub_directory, "obj_flows.mat")
    url = files[patient_id]

    if not os.path.exists(file_path):
        print(f"Downloading {url} using wget...")
        subprocess.run(["wget", "-nc", url, "-O", file_path])
        print(f"File saved to {file_path}")
    else:
        print(f"File {file_path} already exists. Skipping download.")


def find_mat_files(directory):
    mat_files = {}
    for root, dirs, files in os.walk(directory):
        if not 'results.mat' in files:
            for file in files:
                if file.endswith('flows.mat'):
                    subdirectory_name = os.path.basename(root)
                    mat_files[subdirectory_name] = root +  '/'
    return mat_files

directory = './data/'
mat_files = find_mat_files(directory)



for patient_id, folder in mat_files.items():


    vectors_ds, vectors_si, vectors_gt, channel_locations_gt, channel_locations_ds = load_eeg_data(folder ,
                                                                                                   start_time=0,
                                                                                                   end_time=10000)
    
    X = channel_locations_gt[['X','Y','Z']].values.astype(float)
    
    train_idx = np.where(channel_locations_gt.index.isin(channel_locations_ds.index))[0]
    test_idx = np.arange(len(X), dtype=int).reshape(-1, 1)
    
    f_real = vectors_gt[:,:,1:]
    f_spline = vectors_si[:,:,1:] 
    f_down = vectors_ds[:,:,1:]    
   
    max_t = f_down.shape[0]
    gap = 1
    timepoints = list(range(0,max_t,gap))
    f_pred = interpolate_time_range(timepoints, X, f_down, train_idx, test_idx,)
    
    l2_error_ds = compute_error_time(f_pred, f_real, test_idx)
    l2_error_si = compute_error_time(f_spline, f_real, test_idx)
    
    div = {}
    curl = {}
    div['gt'], curl['gt'] = compute_vectorfield_features_time(timepoints, X, f_real)
    div['ds'], curl['ds'] = compute_vectorfield_features_time(timepoints, X, f_pred)
    div['si'], curl['si'] = compute_vectorfield_features_time(timepoints, X, f_spline)
    
    results = {}
    results['f_pred'] = {'f_pred':f_pred}
    results['divergence'] = div
    results['curl'] = curl
    results['error'] = {'rvgp': l2_error_ds, 'spline': l2_error_si}   
    savemat(folder + 'results.mat', mdict=results, oned_as='row')



