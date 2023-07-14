import os
import mne
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pickle
import random
import sys

# Make a list of path to all edf files in the subfolders of /nobackup/tsal-tmp/tuh_eeg
names = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2']

def process_file(filename):
    raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)

    names_check = all([any([name in ch_name and 'P-' not in ch_name for ch_name in raw.ch_names]) for name in names])
    if not names_check:
        return None

    length_check = raw.n_times / raw.info['sfreq'] > 60
    if not length_check:
        return None

    freq_check = raw.info['sfreq'] >= 250
    if not freq_check:
        return None

    raw.filter(1, None, fir_design='firwin', verbose=False)

    idxs = [i for i, ch_name in enumerate(raw.ch_names) if any([name in ch_name and 'P-' not in ch_name for name in names])]
    data = raw.get_data()[idxs]

    min_check = all(data.min(axis=1) < -1e-5)
    if not min_check:
        return None

    max_check = all(data.max(axis=1) > 1e-5)
    if not max_check:
        return None

    std_check = all(data.std(axis=1) > 1e-6) and all(data.std(axis=1) < 1e-2)
    if not std_check:
        return None

    extreme_min_check = np.all(np.abs(data.mean(axis=1) - data.min(axis=1)) < 10 * data.std(axis=1))
    extreme_max_check = np.all(np.abs(data.mean(axis=1) - data.max(axis=1)) < 10 * data.std(axis=1))

    if not extreme_min_check or not extreme_max_check:
        return None

    return filename

#with ProcessPoolExecutor() as executor:
#    files_selected = list(tqdm(executor.map(process_file, files), total=len(files)))
#    files_selected = [file for file in files_selected if file is not None]
 
# Create an empty file tuh_files_selected.pkl

if __name__ == '__main__':
    import pickle
    with open('tuh_files.pkl', 'rb') as f:
        files = pickle.load(f)
        
    random.shuffle(files)
    
    files_selected = []
    
    # Read filename as first argument from command line
    filename = sys.argv[1]
    filename = "tuh_selected" + filename + '.txt'
    print(filename)
    
    open(filename, 'w').close()

    with tqdm(total=len(files)) as pbar:
        for file in files:
            if process_file(file) is not None:
                files_selected.append(file)
                pbar.set_description(str(len(files_selected)))
                # Write the file name to a new line in tuh_files_selected.pkl
                with open(filename, 'a') as f:
                    f.write(file + '\n')
                                    
            pbar.update(1)
            
