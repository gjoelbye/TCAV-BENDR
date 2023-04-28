import os, sys, pickle, random, datetime, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.cm as cm
from tqdm import tqdm
from pathlib import Path
import mne
if os.getcwd().split("/")[-1] != 'BENDR-XAI': os.chdir("../")
from notebooks.utils import *
from functools import partial
import multiprocessing

def read_TUH_edf(file_path, high_pass=0.1, low_pass=100.0, notch=60.0):
    # Read the EDF file
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    # Define the channel map to match 10-20 system channel names
    channel_map = {
        'EEG C3-REF': 'C3', 'EEG P4-REF': 'P4', 'EEG T5-REF': 'P7', 'EEG F8-REF': 'F8', 'EEG F7-REF': 'F7',
        'EEG C4-REF': 'C4', 'EEG PZ-REF': 'Pz', 'EEG FP2-REF': 'Fp2', 'EEG F4-REF': 'F4', 'EEG F3-REF': 'F3',
        'EEG T6-REF': 'P8', 'EEG CZ-REF': 'Cz', 'EEG O2-REF': 'O2', 'EEG O1-REF': 'O1', 'EEG T2-REF': 'FT8',
        'EEG T1-REF': 'FT7', 'EEG T4-REF': 'T8', 'EEG P3-REF': 'P3', 'EEG FZ-REF': 'Fz', 'EEG T3-REF': 'T7',
        'EEG FP1-REF': 'Fp1', 'EEG C4-LE': 'C4', 'EEG P3-LE': 'P3', 'EEG FZ-LE': 'Fz', 'EEG F3-LE': 'F3',
        'EEG FP1-LE': 'Fp1', 'EEG T6-LE': 'P8', 'EEG CZ-LE': 'Cz', 'EEG F8-LE': 'F8', 'EEG O1-LE': 'O1',
        'EEG PZ-LE': 'Pz', 'EEG C3-LE': 'C3', 'EEG FP2-LE': 'Fp2', 'EEG O2-LE': 'O2', 'EEG FP1-LE': 'Fp1',
        'EEG F7-LE': 'F7', 'EEG T1-LE': 'FT7', 'EEG T2-LE': 'FT8', 'EEG P4-LE': 'P4', 'EEG T4-LE': 'T8',
    }

    # Filter the channel_map to include only the channels present in the raw data
    channel_map_sub = {k: v for k, v in channel_map.items() if k in raw.ch_names}
    
    # Standardize the raw data
    mne.datasets.eegbci.standardize(raw)

    # Rename the channels using the filtered channel_map
    raw = raw.rename_channels(channel_map_sub)

    # Create the standard 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')

    # Set the montage for the raw data, ignoring missing channels
    raw = raw.set_montage(montage, on_missing='ignore', verbose=False)
    
    # Pick only channels present in the filtered channel_map
    raw = raw.pick_channels(list(channel_map_sub.values()))

    # Set the average reference for the raw data
    raw = raw.set_eeg_reference(ref_channels='average', projection=True, verbose=False)

    # Apply the average reference projection
    raw.apply_proj(verbose=False)

    # Resample the raw data to 256 Hz
    raw = raw.resample(256)
    
    # Filter the data
    raw = raw.filter(high_pass, low_pass, fir_design='firwin', verbose=False)
    
    # Notch filter at 60 Hz
    raw = raw.notch_filter(notch, fir_design='firwin', verbose=False)
    
    return raw

def process_annotations(window_dict, labels, fwd, compute_inverse):
    # Initialize dictionaries and variables for storing results
    power_dict = {}
    sum_of_src_label = [np.zeros(len(labels[0])), np.zeros(len(labels[1]))]
    total_count = 0
    sum_of_means = np.zeros(fwd['nsource'])

    # Iterate through all annotations
    for annotation in window_dict.keys():
        power_dict[annotation] = np.empty((len(window_dict[annotation]), sum(len(hemi) for hemi in labels)))

        # Iterate through all chopped data segments
        for idx, window in enumerate(window_dict[annotation]):        
            src_estimate = compute_inverse(window)
            sum_of_means += np.mean(src_estimate.data, axis=1)
            total_count += 1

            src_power_label = [np.empty(len(labels[0])), np.empty(len(labels[1]))]

            # Calculate power for each label
            for hemi in range(2):
                for i in range(len(labels[hemi])):
                    src_estimate_label = src_estimate.in_label(labels[hemi][i])
                    src_power_label[hemi][i] = np.mean(src_estimate_label.data**2)
                    sum_of_src_label[hemi][i] += np.mean(src_estimate_label.data)

            power_dict[annotation][idx] = np.concatenate(src_power_label)
            
    # Compute the true mean
    true_mean = (sum_of_means / total_count).reshape(-1, 1)

    return power_dict, true_mean

def compute_variance(window_dict, labels, compute_inverse, true_mean):
    # Initialize the variance dictionary
    variance_dict = {}

    # Iterate through all annotations
    for annotation in window_dict.keys():
        variance_dict[annotation] = np.empty((len(window_dict[annotation]), sum(len(hemi) for hemi in labels)))

        # Iterate through all chopped data segments
        for idx, window in enumerate(window_dict[annotation]):
            src_estimate = compute_inverse(window)
            src_variance_label = [np.empty(len(labels[0])), np.empty(len(labels[1]))]

            # Calculate variance for each label
            for hemi in range(2):
                for i in range(len(labels[hemi])):
                    src_estimate_label = src_estimate.in_label(labels[hemi][i])
                    true_mean_label = true_mean[labels[hemi][i].get_vertices_used()]
                    src_variance_label[hemi][i] = np.mean((src_estimate_label.data - true_mean_label)**2)

            variance_dict[annotation][idx] = np.concatenate(src_variance_label)

    return variance_dict

def process_file(file_path, labels, fwd, high_pass, low_pass, window_length, end_crop, snr):
    raw = read_TUH_edf(file_path, high_pass=high_pass, low_pass=low_pass)
    
    # Length of the recording in seconds
    length = (raw.n_times / raw.info['sfreq'])

    # Diregard the first 5 seconds and the last 5 seconds of the recording
    onset = np.arange(end_crop, length-end_crop, window_length)

    # Duration of each chopped segment
    duration = np.repeat(window_length, len(onset)) - 1/raw.info['sfreq']

    # Description of each chopped segment
    description = np.repeat('T0', len(onset))
    
    # Create the annotations object
    annotations = mne.Annotations(onset=onset, duration=duration, description=description)

    # annotations = mne.Annotations(onset=onset, duration=duration, description=description)
    window_dict, annotations_dict = get_window_dict(raw, annotations)
    
    cov = get_cov(raw) # Get the covariance matrix
    compute_inverse = make_fast_inverse_operator(raw.info, fwd, cov, snr=snr)

    # Process annotations to get power_dict, sum_of_means, and total_count
    power_dict, true_mean = process_annotations(window_dict, labels, fwd, compute_inverse)

    # Compute the variance dictionary
    variance_dict = compute_variance(window_dict, labels, compute_inverse, true_mean)    
    
    return (file_path.name, power_dict, variance_dict, annotations_dict)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--high_pass", type=float, default=0.1, help="High pass frequency in Hz")
    parser.add_argument("--low_pass", type=float, default=100.0, help="Low pass frequency in Hz")
    parser.add_argument("--end_crop", type=float, default=5.0, help="Length of the recording to disregard at the beginning and end in seconds")
    parser.add_argument("--window_length", type=float, default=4.0, help="Length of each chopped segment in seconds")
    parser.add_argument("--n_processes", type=int, default=-1, help="Number of processes to use for multiprocessing")
    parser.add_argument("--snr", type=float, default=100.0, help="Signal to noise ratio for computing the inverse operator")
    parser.add_argument("--edf_dir", type=str, default="/scratch/s194260/tuh_eeg", help="Path to the directory containing the EDF files")
    parser.add_argument("--parcellation_name", type=str, default="HCPMMP1_combined", help="Name of the parcellation to use")
    parser.add_argument("--save_dir", type=str, default="", help="Path to the directory to save the processed data")
    args = parser.parse_args()
    
    end_crop = args.end_crop # Length of the recording to disregard at the beginning and end in seconds
    window_length = args.window_length # Length of each chopped segment in seconds
    n_processes = args.n_processes # Number of processes to use for multiprocessing
    low_pass = args.low_pass # Low pass frequency in Hz
    high_pass = args.high_pass # High pass frequency in Hz
    snr = args.snr # Signal to noise ratio for computing the inverse operator
    edf_dir = Path(args.edf_dir) # Path to the directory containing the EDF files
    parcellation_name = args.parcellation_name # Name of the parcellation to use
    save_dir = Path(args.save_dir) # Path to the directory to save the processed data
    
    now = datetime.datetime.now()
    now_str = now.strftime("%H%M%S_%d%m%y")
    tqdm.write(f"[INFO] Starting at {now_str}")
    
    # Get paths
    subjects_dir, subject, trans, src_path, bem_path = get_fsaverage()
    
    # Get the labels for the parcellation
    labels = get_labels(subjects_dir, parcellation_name = parcellation_name)
    
    tqdm.write(f"[INFO] Loaded {parcellation_name} parcellation")
    
    # Get EDF file paths    
    edf_files = [edf_dir / file for file in os.listdir(edf_dir)]
    
    edf_files = edf_files[0:1]
    
    tqdm.write(f"[INFO] Found {len(edf_files)} EDF files")
    
    # Get forward model
    info = read_TUH_edf(edf_files[0]).info
    fwd = get_fwd(info, trans, src_path, bem_path)
    
    tqdm.write("[INFO] Forward model loaded")
    
    custom_functions = partial(process_file, labels=labels, fwd=fwd, high_pass=high_pass,
                               low_pass=low_pass, window_length=window_length,
                               end_crop=end_crop, snr=snr)
    
    tqdm.write(f"[INFO] Custom functions defined")
    
    if n_processes == -1:
        pool = multiprocessing.Pool()
    else:
        pool = multiprocessing.Pool(processes=n_processes)
        
    tqdm.write(f"[INFO] Starting multiprocessing with {pool._processes} processes with {multiprocessing.cpu_count()} CPUs available")
    
    results = []
    with tqdm(total = len(edf_files)) as pbar:
        for result in pool.imap_unordered(custom_functions, edf_files):
            results.append(result)
            pbar.update(1)
            pbar.set_description(f"Running... {result[0]}")

        pool.close()
        pool.join()
    
    tqdm.write(f"[INFO] Finished multiprocessing")    
    
    dataset = dict()

    for result in results:
        dataset[result[0]] = {
            "power": result[1],
            "variance": result[2],
            "annotations": result[3]
        }
    
    tqdm.write(f"[INFO] Created dataset")
    
    output_name = f"{parcellation_name}_{high_pass}_{low_pass}_{now_str}.npy"
    
    np.save(save_dir / output_name, dataset, allow_pickle=True)
    
    tqdm.write(f"[INFO] Saved dataset. Total time: {str(datetime.datetime.now() - now)}")
    
    
