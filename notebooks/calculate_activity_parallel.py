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
mmidb_path = Path(r"/home/s194260/BENDR-XAI/data/eegmmidb/files")
parcellation_name = "aparc.a2009s"
snr = 1.0

info = get_raw(random_edf_file_path, filter=True).info # Just need one raw to get info
src = get_src(src_path)
fwd = get_fwd(info, trans, src_path, bem_path)

labels = get_labels(subjects_dir, parcellation_name = parcellation_name)

def calculate_activity_per_label(annotation_dict, labels, compute_inverse):
    activity = {}

    for anno in annotation_dict.keys():
        activity[anno] = np.empty((len(annotation_dict[anno]), sum(len(hemi) for hemi in labels)))
        for i, window in enumerate(annotation_dict[anno]):
            stc = compute_inverse(window)
            activity[anno][i] = np.concatenate(get_power_per_label(stc, labels, standardize=False))

    return activity

def process_file(filepath):
    raw = get_raw(filepath, filter=True, high_pass=high_pass, low_pass=low_pass, notch=None)
    annotations = get_annotations(filepath)
    annotation_dict = get_window_dict(raw, annotations)

    cov = get_cov(raw)
    compute_inverse = make_fast_inverse_operator(raw.info, fwd, cov, snr=snr)

    activity = calculate_activity_per_label(annotation_dict, labels, compute_inverse)
    return (filepath.parent.name, filepath.name[:-4], activity)

pool = multiprocessing.Pool()
filepaths = []
for (dirpath, _, filenames) in os.walk(mmidb_path):
    for filename in filenames:
        if filename.endswith(".edf"):
            filepaths.append(Path(dirpath) / filename)       

results = []
pbar = tqdm()
for result in pool.imap_unordered(process_file, filepaths):
    results.append(result)
    pbar.update(1)
    pbar.set_description(result[1])

pool.close()
pool.join()

dataset_activity = defaultdict(lambda: {})

for result in results:
    dataset_activity[result[0]][result[1]] = result[2]


dataset_activity = dict(dataset_activity)

np.save("mmidb_{}_{}_{}_{}_parallel".format(parcellation_name, str(round(snr, 1)),
        str(round(high_pass, 1)), str(round(low_pass, 1))), dataset_activity, allow_pickle=True)