import os
from collections import defaultdict
if os.getcwd().split("/")[-1] != 'BENDR-XAI': os.chdir("../")

import mne
import numpy as np
import matplotlib.pyplot as plt
from utils import *

from matplotlib import animation
import matplotlib.cm as cm
import sys
from tqdm import tqdm
from pathlib import Path
import multiprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--high_pass", type=float, default=0.5)
parser.add_argument("--low_pass", type=float, default=70.0)
args = parser.parse_args()

low_pass = args.low_pass
high_pass = args.high_pass

print("High pass", high_pass)
print("Low pass", low_pass)

subjects_dir, subject, trans, src_path, bem_path = get_fsaverage()

random_edf_file_path = 'notebooks/S001R03.edf' 
mmidb_path = r"/work1/s194260/eegmmidb/files"
#mmidb_path = r"data/eegmmidb/files"
parcellation_name = "aparc.a2009s"

info = get_raw(random_edf_file_path, filter=True).info # Just need one raw to get info
src = get_src(src_path)
fwd = get_fwd(info, trans, src_path, bem_path)

labels = get_labels(subjects_dir, parcellation_name = parcellation_name)
visual_region_labels = [[label  for label in labels[hemi] if "occipital" in label.name] for hemi in range(2)]

def calculate_activity_per_label(annotation_dict, labels, compute_inverse):
    activity = {}

    for anno in annotation_dict.keys():
        activity[anno] = np.empty((len(annotation_dict[anno]), sum(len(hemi) for hemi in labels)))
        for i, window in enumerate(annotation_dict[anno]):
            stc = compute_inverse(window)
            activity[anno][i] = np.concatenate(get_power_per_label(stc, labels, standardize=False))

    return activity

def process_file(patient):
    raw_open = get_raw(mmidb_path + f'/{patient}/{patient}R01.edf', filter=True, resample=False, high_pass=8, low_pass=12)
    annotation_open = get_annotations(mmidb_path + f'/{patient}/{patient}R01.edf')
    raw_open = mne.concatenate_raws([eeg for anno in get_window_dict(raw_open, annotation_open).values() for eeg in anno])

    raw_closed = get_raw(mmidb_path + f'/{patient}/{patient}R02.edf', filter=True, resample=False, high_pass=8, low_pass=12)
    annotation_closed = get_annotations(mmidb_path + f'/{patient}/{patient}R02.edf')
    raw_closed = mne.concatenate_raws([eeg for anno in get_window_dict(raw_closed, annotation_closed).values() for eeg in anno])

    cov_open = get_cov(raw_open)
    cov_closed = get_cov(raw_closed)

    snr_array = np.arange(3, 200, 1)
    result_visual = np.empty(len(snr_array))

    for i, snr in tqdm(enumerate(snr_array)):
        operator_open = make_fast_inverse_operator(info, fwd, cov_open, method="eLORETA", snr=snr, nave=1, max_iter=1000, verbose=False)
        operator_closed = make_fast_inverse_operator(info, fwd, cov_closed, method="eLORETA", snr=snr, nave=1, max_iter=1000, verbose=False)

        stc_open = operator_open(raw_open)
        stc_closed = operator_closed(raw_closed)
        
        activity_open = get_power_per_label(stc_open, visual_region_labels, standardize=False)
        activity_closed = get_power_per_label(stc_closed, visual_region_labels, standardize=False)

        activity_open = np.concatenate(activity_open)
        activity_closed = np.concatenate(activity_closed)

        result_visual[i] = np.sum(activity_closed - activity_open)

    return (patient, result_visual)

pool = multiprocessing.Pool()
filepaths = []

patients = [f"S{i:03}" for i in range(1, 110)]

results = []
pbar = tqdm()
for result in pool.imap_unordered(process_file, patients):
    results.append(result)
    pbar.update(1)
    pbar.set_description(result[0])

pool.close()
pool.join()

result_dict = dict()

for result in results:
    result_dict[result[0]]= result[1]

np.save("snr_result", result_dict, allow_pickle=True)