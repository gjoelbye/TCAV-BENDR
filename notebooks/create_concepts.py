import numpy as np
#import matplotlib.pyplot as plt
#from tqdm import tqdm, trange
#import mne, re, os
import os
import torch
import pickle
import shutil

from utils import load_mmidb_data_dict, get_fsaverage, get_labels




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







def save_concept(DATA_PATH_CONCEPTS, NUMBER_CHANNELS=20, NUMBER_SAMPLES=1024, raw, concept, patient, run, key,):

    x = np.zeros((1, NUMBER_CHANNELS, NUMBER_SAMPLES))
    x[:,:19,:] = raw.copy().get_data()[:,:NUMBER_SAMPLES].reshape(1,NUMBER_CHANNELS-1,NUMBER_SAMPLES)
    x[:,19,:] = np.ones((1, NUMBER_SAMPLES)) * -1  
    x = torch.from_numpy(x).float()
    
    picklePath = DATA_PATH_CONCEPTS + concept + '/' + patient + run + '_' + key + '_' + concept + '.pkl'
    with open(picklePath, 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == '__main__':


    # argparse?
    SNR = 100.0
    PARCELLATION = 'HCPMMP1_combined'

    # argparse
    DATA_PATH_RAW = '/home/williamtheodor/Documents/DL for EEG Classification/data/eegmmidb (raw)/files/'
    DATA_PATH_CONCEPTS = '/home/williamtheodor/Documents/DL for EEG Classification/data/sanity check concepts MMIDB open_closed/'
    ACTIVTY_DICT_PATH = '/home/williamtheodor/Documents/DL for EEG Classification/data/'

    # TODO: Update to fit TUH activity dict
    activty_dict = load_mmidb_data_dict(ACTIVTY_DICT_PATH, PARCELLATION, SNR, chop=True)

    bands = activty_dict.keys()
    bands_to_use = ['Alpha'] #argparse

    baseline_runs = ['R01', 'R02']
    task_runs = ['R03', 'R04', 'R07', 'R08', 'R11', 'R12']


    subjects_dir, subject, trans, src_path, bem_path = get_fsaverage()
    labels = get_labels(subjects_dir, parcellation_name=PARCELLATION)
    label_names = [label.name for label in np.array(labels).flatten()]

    make_concepts_folders(DATA_PATH_CONCEPTS, bands_to_use, label_names)


    

