import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import mne, re, os, pickle, shutil
import torch
import datetime

from utils import *

def get_info_dict(ACTIVITY_PATH):
    info_dict = {}

    for high_pass in ['1.0', '4.0', '8.0', '12.0', '30.0']:
        info_dict[str(high_pass)] = {}

        for snr in ['5.0', '10.0', '50.0', '100.0']:
            info_dict[str(high_pass)][str(snr)] = {}

            for project in ['noproj', 'proj']:
                info_dict[str(high_pass)][str(snr)][str(project)] = {}

    for file in os.listdir(ACTIVITY_PATH):
        high_pass = file.split('_')[2]
        low_pass = file.split('_')[3]
        snr = file.split('_')[4]
        time = file.split('_')[5]
        date = file.split('_')[6]
        p = file.split('_')[7]

        info_dict[str(high_pass)][str(snr)][p] = {'time': time, 'date': date}

    return info_dict

def get_activity_dict(ACTIVITY_PATH, SNR, info_dict, PARCELLATION='HCPMMP1_combined'):

    def sort_dict(dict):
        return {k: dict[k] for k in sorted(dict.keys())}

    delta_activity = np.load(ACTIVITY_PATH + PARCELLATION + '_1.0_4.0_' + str(SNR) + '_' + info_dict['1.0'][str(SNR)]['proj']['time'] + '_' + info_dict['1.0'][str(SNR)]['proj']['date'] + '_proj_test.npy', allow_pickle=True).item()
    theta_activity = np.load(ACTIVITY_PATH + PARCELLATION + '_4.0_8.0_' + str(SNR) + '_' + info_dict['4.0'][str(SNR)]['proj']['time'] + '_' + info_dict['4.0'][str(SNR)]['proj']['date'] + '_proj_test.npy', allow_pickle=True).item()
    alpha_activity = np.load(ACTIVITY_PATH + PARCELLATION + '_8.0_12.0_' + str(SNR) + '_' + info_dict['8.0'][str(SNR)]['proj']['time'] + '_' + info_dict['8.0'][str(SNR)]['proj']['date'] + '_proj_test.npy', allow_pickle=True).item()
    beta_activity = np.load(ACTIVITY_PATH + PARCELLATION + '_12.0_30.0_' + str(SNR) + '_' + info_dict['12.0'][str(SNR)]['proj']['time'] + '_' + info_dict['12.0'][str(SNR)]['proj']['date'] + '_proj_test.npy', allow_pickle=True).item()
    gamma_activity = np.load(ACTIVITY_PATH + PARCELLATION + '_30.0_70.0_' + str(SNR) + '_' + info_dict['30.0'][str(SNR)]['proj']['time'] + '_' + info_dict['30.0'][str(SNR)]['proj']['date'] + '_proj_test.npy', allow_pickle=True).item()

    activity_dict = {
        'Delta': sort_dict(delta_activity),
        'Theta': sort_dict(theta_activity),
        'Alpha': sort_dict(alpha_activity),
        'Beta': sort_dict(beta_activity),
        'Gamma': sort_dict(gamma_activity)
    }

    return activity_dict


def normalize(x, minimum=-0.00125, maximum=0.00125):
    return (x - minimum) / (maximum - minimum) * 2 - 1

def save_concept(DATA_PATH_CONCEPTS, raw, concept, patient, run, idx, NUMBER_CHANNELS=20, NUMBER_SAMPLES=1024):

    x = torch.zeros((1, NUMBER_CHANNELS, NUMBER_SAMPLES))
    x[:, :NUMBER_CHANNELS-1, :] = torch.from_numpy(raw.copy().get_data()[:, :NUMBER_SAMPLES].reshape(1, NUMBER_CHANNELS-1, NUMBER_SAMPLES))
    x = normalize(x)
    x[:,NUMBER_CHANNELS-1,:] = torch.ones((1, NUMBER_SAMPLES)) * -1  
    
    picklePath = DATA_PATH_CONCEPTS + concept + '/' + patient + run + '_' + idx + '_' + concept + '.pkl'
    with open(picklePath, 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

def save_concept(window, concept, DATA_PATH_CONCEPTS, edf_file, idx, NUMBER_CHANNELS=20, NUMBER_SAMPLES=1024):

    x = torch.zeros((1, NUMBER_CHANNELS, NUMBER_SAMPLES))

    x[:, :NUMBER_CHANNELS-1, :] = torch.from_numpy(window)
    x[:,NUMBER_CHANNELS-1,:] = torch.ones((1, NUMBER_SAMPLES)) * -1  
    
    picklePath = DATA_PATH_CONCEPTS + concept + '/' + edf_file[:-4] + '_' + str(idx) + '_' + concept + '.pkl'
    with open(picklePath, 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_concepts(edf_files, DATA_PATH_RAW, DATA_PATH_CONCEPTS, activity_dict, bands, bands_to_use, standardize, sigma, absolute, NUMBER_CHANNELS=20, NUMBER_SAMPLES=1024):

    channel_order = [
                'Fp1', 'Fp2',
        'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
                 'O1', 'O2'
    ]

    subjects_dir, subject, trans, src_path, bem_path = get_fsaverage()
    labels = get_labels(subjects_dir, parcellation_name='HCPMMP1_combined')
    label_names = [label.name for label in np.array(labels).flatten()]

    make_concepts_folders(DATA_PATH_CONCEPTS, bands_to_use, label_names)


    for edf_file in tqdm(edf_files, total=len(edf_files)): 

        FILE = DATA_PATH_RAW + edf_file
        raw = read_TUH_edf(FILE)

        try:
            raw.pick_channels(channel_order)
            raw.reorder_channels(channel_order)
        except:
            continue

        annotations = activity_dict['Alpha'][edf_file]['annotations']['T0']

        baseline_power = np.array([activity_dict[band][edf_file]['power']['T0'].mean(axis=0) for band in bands_to_use])
        baseline_variance = np.array([activity_dict[band][edf_file]['variance']['T0'].mean(axis=0) for band in bands_to_use])

        for idx, annotation in enumerate(annotations):
            window = get_window(raw, annotation).get_data()
            window = normalize(window)

            power = np.array([activity_dict[band][edf_file]['power']['T0'][idx] for band in bands_to_use])

            if standardize == 'subtract':
                power -= baseline_power

                if absolute:
                    power = np.abs(power)

            elif standardize == 'divide':
                power /= baseline_power

            if sigma:
                power /= np.sqrt(baseline_variance)

            concept_idx = np.unravel_index(power.argmax(), power.shape)

            if len(bands_to_use) == 1:
                most_active_band = 'Alpha'
            else:
                most_active_band = bands[concept_idx[0]]

            brain_region_idx = concept_idx[1]
            brain_region = label_names[brain_region_idx]

            concept = most_active_band + '_' + brain_region

            save_concept(window, concept, DATA_PATH_CONCEPTS, edf_file, idx)


if __name__ == '__main__':

    ACTIVITY_PATH = '/home/williamtheodor/Documents/DL for EEG Classification/data/Activity/'

    info_dict = get_info_dict(ACTIVITY_PATH)

    SNR = '100.0'

    activity_dict = get_activity_dict(ACTIVITY_PATH, SNR=SNR, info_dict=info_dict)
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    bands_to_use = bands #['Alpha'] #
    bands_naming = 'all' # 'alpha' #

    standardize = 'subtract' #'divide' # 'subtract'
    sigma = True # False
    absolute = True # False

    PARCELLATION='HCPMMP1_combined'

    edf_files = list(activity_dict['Delta'].keys())

    DATA_PATH_RAW = '/home/williamtheodor/Documents/DL for EEG Classification/data/tuh_eeg/'

    now = datetime.datetime.now()
    now_str = now.strftime("%H%M%S_%d%m%y")

    concept_folder = f'TUH_clean_{bands_naming}_{standardize}_{sigma}sigma_{absolute}abs_{str(SNR)}_{now_str}'
    DATA_PATH_CONCEPTS = f'/home/williamtheodor/Documents/DL for EEG Classification/data/tuh concepts/{concept_folder}/'

    create_concepts(edf_files, DATA_PATH_RAW, DATA_PATH_CONCEPTS, activity_dict, bands, bands_to_use, standardize, sigma, absolute, NUMBER_CHANNELS=20, NUMBER_SAMPLES=1024)