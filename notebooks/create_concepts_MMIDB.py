import mne
import torch
import pickle
import os, shutil
import numpy as np
from tqdm import tqdm

from utils import load_mmidb_data_dict, get_fsaverage, get_labels, get_annotations, get_window_dict


def pick_and_rename_MMIDB_channels(raw):
    raw = raw.copy()
    mne.channels.rename_channels(raw.info, {'Fp1': 'FP1', 'Fp2': 'FP2', 'Fz': 'FZ', 'Cz': 'CZ', 'Pz': 'PZ'})

    
    if 'P7' in raw.ch_names:
        raw.rename_channels({'P7': 'T5'})
    if 'P8' in raw.ch_names:
        raw.rename_channels({'P8': 'T6'})

    EEG_20_div = [
                'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'T5', 'P3', 'PZ', 'P4', 'T6',
                 'O1', 'O2'
    ]
    
    raw.pick_channels(ch_names=EEG_20_div)
    raw.reorder_channels(EEG_20_div)

    return raw


def get_raw(FILE, high_pass=0.5, low_pass=70, notch=60, SAMPLE_FREQ=256):
    raw = mne.io.read_raw_edf(FILE, verbose=False, preload=True)
    mne.datasets.eegbci.standardize(raw)  # Set channel names

    # set reference to average + projection
    raw = raw.set_eeg_reference(ref_channels='average', projection=True, verbose=False)

    # set montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw = raw.set_montage(montage)

    # resample
    raw.resample(SAMPLE_FREQ, npad="auto", verbose=False)

    # filter
    raw.filter(high_pass, low_pass, fir_design='firwin', verbose=False)
    raw.notch_filter(notch, fir_design='firwin', verbose=False)

    # pick and rename channels
    raw = pick_and_rename_MMIDB_channels(raw)

    return raw


def make_concepts_folders(DATA_PATH_CONCEPTS, bands_to_use, label_names):

    # remove and remake concepts folder
    print('Removing and remaking concepts folder: ', DATA_PATH_CONCEPTS)

    if os.path.exists(DATA_PATH_CONCEPTS):
        shutil.rmtree(DATA_PATH_CONCEPTS)
    os.mkdir(DATA_PATH_CONCEPTS)

    for label_idx in range(len(label_names)):
        for band in bands_to_use:
        # make directory if it doesn't exist
            if not os.path.exists(f'{DATA_PATH_CONCEPTS}{band}_{label_names[label_idx]}'):
                os.makedirs(f'{DATA_PATH_CONCEPTS}{band}_{label_names[label_idx]}')


def min_max_normalize_dn3(x: torch.Tensor, low=-1, high=1):
    if len(x.shape) == 2:
        xmin = x.min()
        xmax = x.max()
        if xmax - xmin == 0:
            x = 0
            return x
    elif len(x.shape) == 3:
        xmin = torch.min(torch.min(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        xmax = torch.max(torch.max(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        constant_trials = (xmax - xmin) == 0
        if torch.any(constant_trials):
            # If normalizing multiple trials, stabilize the normalization
            xmax[constant_trials] = xmax[constant_trials] + 1e-6

    x = (x - xmin) / (xmax - xmin)

    # Now all scaled 0 -> 1, remove 0.5 bias
    x -= 0.5
    # Adjust for low/high bias and scale up
    x += (high + low) / 2

    return (high - low) * x



def save_concept(DATA_PATH_CONCEPTS, raw, concept, patient, run, idx, NUMBER_CHANNELS=20, NUMBER_SAMPLES=1024):

    x = torch.zeros((1, NUMBER_CHANNELS, NUMBER_SAMPLES))
    x[:, :NUMBER_CHANNELS-1, :] = torch.from_numpy(raw.copy().get_data()[:, :NUMBER_SAMPLES].reshape(1, NUMBER_CHANNELS-1, NUMBER_SAMPLES))
    x = min_max_normalize_dn3(x)
    x[:,NUMBER_CHANNELS-1,:] = torch.ones((1, NUMBER_SAMPLES)) * -1  
    
    picklePath = DATA_PATH_CONCEPTS + concept + '/' + patient + run + '_' + idx + '_' + concept + '.pkl'
    with open(picklePath, 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)


def define_concepts(DATA_PATH_RAW, DATA_PATH_CONCEPTS, activity_dict, patients, runs_to_use, bands_to_use, codes_to_use, label_names, WINDOW_LENGTH=4, NUMBER_CHANNELS=20, NUMBER_SAMPLES=1024, resting_state=True):
    number_labels = len(label_names)
    number_bands = len(bands_to_use)

    for patient in tqdm(patients):

        if not resting_state:
            baseline_run = 'R01'
            baseline_activity_mean = np.array([activity_dict[band][patient][patient+baseline_run]['T0'].mean(axis=0) for band in bands_to_use])
            baseline_activity_mean = baseline_activity_mean.reshape(number_bands, number_labels)
            # TODO: Add std

        for run in runs_to_use:
            
            if resting_state:
                baseline_activity_mean = np.array([activity_dict[band][patient][patient+run]['T0'].mean(axis=0) for band in bands_to_use])
                baseline_activity_mean = baseline_activity_mean.reshape(number_bands, number_labels)
                # TODO: Add std


            EDF_FILE = DATA_PATH_RAW+f'{patient}/{patient}{run}.edf'
            raw = get_raw(EDF_FILE)

            if run in ['R01', 'R02']:
                annotations = get_annotations(EDF_FILE, window_length=WINDOW_LENGTH)
            else:
                annotations = get_annotations(EDF_FILE)

            annotation_dict = get_window_dict(raw, annotations)

            for code in codes_to_use:
                for raw_idx, raw in enumerate(annotation_dict[code]):

                    activity = np.array([activity_dict[band][patient][patient+run][code][raw_idx] for band in bands_to_use])
                    activity = np.abs(activity - baseline_activity_mean) #/ baseline_activity_std

                    most_active_band_idx = np.argmax(activity.mean(axis=1))
                    most_active_band = bands_to_use[most_active_band_idx]

                    brain_region_idx = activity[most_active_band_idx].argmax()
                    brain_region = label_names[brain_region_idx]

                    concept = most_active_band + '_' + brain_region

                    save_concept(DATA_PATH_CONCEPTS, raw, concept, patient, run, str(raw_idx), NUMBER_CHANNELS=NUMBER_CHANNELS, NUMBER_SAMPLES=NUMBER_SAMPLES)


if __name__ == '__main__':

    # argparse?
    SNR = 100.0
    PARCELLATION = 'HCPMMP1_combined'

    # argparse
    DATA_PATH_RAW = '/home/williamtheodor/Documents/DL for EEG Classification/data/eegmmidb (raw)/files/'
    DATA_PATH_CONCEPTS = '/home/williamtheodor/Documents/DL for EEG Classification/data/sanity check concepts MMIDB/'
    ACTIVTY_DICT_PATH = '/home/williamtheodor/Documents/DL for EEG Classification/data/'

    # TODO: Update to fit TUH activity dict
    activity_dict = load_mmidb_data_dict(ACTIVTY_DICT_PATH, PARCELLATION, SNR, chop=True)

    patients_to_exclude = ['S088', 'S089', 'S090', 'S092', 'S104', 'S106']
    patients = [key for key in activity_dict['Alpha'].keys() if key not in patients_to_exclude]

    bands = list(activity_dict.keys())
    bands_to_use = bands
    #bands_to_use = ['Alpha', 'Gamma'] #argparse

    baseline_runs = ['R01', 'R02']
    task_runs = ['R03', 'R04', 'R07', 'R08', 'R11', 'R12']

    runs_to_use = task_runs #argparse
    resting_state = False # true if using baseline runs

    codes = ['T0', 'T1', 'T2']
    codes_to_use = ['T0'] #argparse

    subjects_dir, subject, trans, src_path, bem_path = get_fsaverage()
    labels = get_labels(subjects_dir, parcellation_name=PARCELLATION)
    label_names = [label.name for label in np.array(labels).flatten()]

    make_concepts_folders(DATA_PATH_CONCEPTS, bands_to_use, label_names)

    define_concepts(DATA_PATH_RAW, DATA_PATH_CONCEPTS, activity_dict, 
                    patients, runs_to_use, bands_to_use, codes_to_use, label_names, 
                    WINDOW_LENGTH=4, NUMBER_CHANNELS=20, NUMBER_SAMPLES=1024, 
                    resting_state=resting_state)