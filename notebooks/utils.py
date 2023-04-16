import mne
from mne.datasets import fetch_fsaverage
import numpy as np
from pathlib import Path
import os

import vtk
from vtk import vtkPolyData, vtkDecimatePro
from vtk.util.numpy_support import vtk_to_numpy

from typing import Dict, List, Tuple, Union

def get_raw(edf_file_path: Path, filter: bool = True,
            high_pass = 0.5, low_pass = 70, notch = 60, resample = 256) -> mne.io.Raw:
    """Reads an edf file and returns a raw object.
    Parameters
    ----------
    edf_file_path : str
        Path to the edf file.
    filter : bool
        Whether to filter the data or not.
    Returns
    -------
    raw : mne.io.Raw
        The raw object.
    """
    raw = mne.io.read_raw_edf(edf_file_path, verbose=False, preload=True)
    mne.datasets.eegbci.standardize(raw)  # Set channel names
    montage = mne.channels.make_standard_montage('standard_1005')

    new_names = dict(
        (ch_name,
        ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp'))
        for ch_name in raw.ch_names)
    raw.rename_channels(new_names)
    
    raw = raw.set_eeg_reference(ref_channels='average', projection=True, verbose = False)
    raw = raw.set_montage(montage); # Set montage

    if filter:
        if resample:
            raw = raw.resample(resample)
        if low_pass and high_pass:
            raw = raw.filter(high_pass, low_pass, verbose = False)
        if notch:
            raw = raw.notch_filter(notch, verbose = False)

    return raw


def pick_and_rename_channels(raw):
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

def get_annotations(edf_file_path: str) -> mne.Annotations:
    """Reads an edf file and returns the annotations.
    Parameters
    ----------
    edf_file_path : str
        Path to the edf file.
    Returns
    -------
    annotations : mne.Annotations
        The annotations.
    """
    annotations = mne.read_annotations(edf_file_path)
    
    if isinstance(window_length, (int, float)):
        window_length = 4.0

        new_onset = []
        new_duration = []
        new_description = []

        for i in range(len(annotations)):  
            for j in range(int(annotations.duration[i] // window_length)):
                new_onset.append(annotations.onset[i] + j * window_length)
                new_duration.append(window_length)
                new_description.append(annotations.description[i])
                
        new_onset = np.array(new_onset, dtype=np.float64)
        new_duration = np.array(new_duration, dtype=np.float64)
        new_description = np.array(new_description, dtype='<U2')

        annotations = mne.Annotations(onset=new_onset, duration=new_duration, description=new_description)

    return annotations

def get_fsaverage(verbose = False):
    """Returns the fsaverage files.
    Parameters
    ----------
    verbose : bool
        Whether to print the progress or not.
    Returns
    ----------
    subjects_dir : str
        The subjects directory.
    subject : str
        The subject.
    trans : str
        The transformation.
    src_path : str
        The source path.
    bem_path : str
        The bem path.
    """
    # Download fsaverage files
    fs_dir = Path(fetch_fsaverage(verbose=False))
    subjects_dir = os.path.dirname(fs_dir)

    # The files live in:
    subject = 'fsaverage'
    trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
    src_path = fs_dir / 'bem' / 'fsaverage-ico-5-src.fif'
    bem_path = fs_dir / 'bem' / 'fsaverage-5120-5120-5120-bem-sol.fif'

    return subjects_dir, subject, trans, src_path, bem_path


def get_fwd(info, trans, src_path, bem_path):
    fwd = mne.make_forward_solution(info, trans=trans, src=src_path, bem=bem_path,
                                     meg=False, eeg=True, verbose=False, mindist=5.0, n_jobs=4)

    #fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose=False)
    # leadfield = fwd['sol']['data']

    return fwd


def get_cov(raw: mne.io.Raw) -> mne.Covariance:
    cov = mne.compute_raw_covariance(raw, verbose=False)
    return cov


def make_fast_inverse_operator(info, fwd, cov, method="eLORETA", snr=3, nave=1, max_iter=100, verbose=False):
    lambda2 = 1/snr**2
    inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov, verbose=verbose)
    inv = mne.minimum_norm.prepare_inverse_operator(inv, nave, lambda2, method=method, method_params={'max_iter': max_iter}, verbose=verbose)
    
    func = lambda x: mne.minimum_norm.apply_inverse_raw(x, inv, lambda2, method=method, nave=nave, prepared=True, method_params={'max_iter': max_iter}, verbose=verbose)
    return func

def get_stc(raw: mne.io.Raw, fwd: mne.forward.Forward, cov: mne.Covariance,
            tmin: float = None, tmax: float = None, snr: float = 3, method: str = "eLORETA",
            nave: int = 1, verbose: bool = False) -> mne.SourceEstimate:
    """Returns the source estimate.
    Parameters
    ----------
    raw : mne.io.Raw
        The raw object.
    fwd : mne.forward.forward.Forward
        The forward solution.
    cov : mne.cov.Covariance
        The covariance matrix.
    tmin : float
        The start time.
    tmax : float
        The end time.
    snr : float
        The signal to noise ratio.
    method : str
        The method to use.
    nave : int
        The number of averages.
    verbose : bool
        Whether to print the progress or not.
    Returns
    ----------
    stc : mne.SourceEstimate
        The source estimate.
    """

    if tmin is not None:
        idx_start = int(tmin * raw.info['sfreq'])
    else:
        idx_start = None
    
    if tmax is not None:
        idx_stop = int(tmax * raw.info['sfreq'])
    else:
        idx_stop = None

    lambda2 = 1.0 / snr ** 2 # Use smaller SNR for raw data

    inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, verbose=verbose)
    stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2, method=method,
                                             nave = nave, start=idx_start, stop=idx_stop,
                                             verbose=verbose)

    return stc

def get_labels(subjects_dir, parcellation_name: str = "aparc_sub", verbose: bool = False) -> List[List[mne.Label]]:
    mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, accept=True)

    labels_lh = mne.read_labels_from_annot('fsaverage', parcellation_name, 'lh', subjects_dir=subjects_dir, verbose=verbose) 
    labels_rh = mne.read_labels_from_annot('fsaverage', parcellation_name, 'rh', subjects_dir=subjects_dir, verbose=verbose)
    labels = [labels_lh, labels_rh]
    return labels


### EXTRACT DATA ###

def get_src(src_path):
    src = mne.read_source_spaces(src_path, verbose=False)
    return src

# Not the same coordinates as from forward solution
def get_vertices(src):
    vertices_lh = src[0]['rr']
    vertices_rh = src[1]['rr']
    vertices = np.array((vertices_lh, vertices_rh))
    return vertices

def get_vertices_tris(src):
    tris_lh = src[0]['tris']
    tris_rh = src[1]['tris']
    tris = np.array((tris_lh, tris_rh))
    return tris

# Not the same coordinates as from forward solution
def get_sources(src):
    sources_lh = src[0]['rr'][src[0]['inuse'].astype(bool)]
    sources_rh = src[1]['rr'][src[1]['inuse'].astype(bool)]
    sources = np.array((sources_lh, sources_rh))   
    return sources

def get_sources_tris(src):
    tris_lh = src[0]['use_tris']
    tris_rh = src[1]['use_tris']
    tris = np.array((tris_lh, tris_rh))
    return tris

def decimate_mesh(vertices, triangles, values=None, reduction = 0.5, verbose = False):
    #vertices_down, triangles_down, color_down = decimate_mesh(vertices, triangles, color, reduction=0.90, verbose=False)
    if values is None:
        values = np.ones(len(vertices))

    pd = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    _ = [points.InsertNextPoint(*vertices[i]) for i in range(len(vertices))]

    cells = vtk.vtkCellArray()
    for triangle in triangles:
        cell = vtk.vtkTriangle()
        Ids = cell.GetPointIds()
        for kId in range(len(triangle)):
            Ids.SetId(kId, triangle[kId])
        cells.InsertNextCell(cell)

    vtkvalues = vtk.vtkFloatArray()
    vtkvalues.SetNumberOfComponents(1)
    vtkvalues.SetNumberOfTuples(len(vertices))
    for i in range(len(vertices)):
        vtkvalues.SetValue(i, values[i])

    pd.GetPointData().SetScalars(vtkvalues)
    pd.SetPoints(points)
    pd.SetPolys(cells)

    decimate = vtkDecimatePro()
    decimate.SetInputData(pd)
    decimate.SetTargetReduction(reduction)
    decimate.Update()

    dpd = vtkPolyData()
    dpd.ShallowCopy(decimate.GetOutput())

    if verbose:
        print("After decimation \n"
                "-----------------\n"
                "There are " + str(dpd.GetNumberOfPoints()) + " vertices.\n"
                "There are " + str(dpd.GetNumberOfPolys()) + " triangles.\n")

    triangles_down = vtk_to_numpy(dpd.GetPolys().GetData())
    triangles_down = triangles_down.reshape(int(len(triangles_down)/4), 4)[:, 1:]
    vertices_down = vtk_to_numpy(dpd.GetPoints().GetData())
    values_down = vtk_to_numpy(dpd.GetPointData().GetScalars())
        
    return vertices_down, triangles_down, values_down


def get_z_values(vertices, tris):
    highest = np.max(vertices[:, :, 2].flatten())
    lowest = np.min(vertices[:, :, 2].flatten())

    vertices_values = (vertices[:, :, 2] - lowest) / (highest - lowest)
    tris_values = np.empty(tris.shape[0:2])

    for i, tri in enumerate(tris):
        for j, (a,b,c) in enumerate(tri):
            tris_values[i, j] = np.mean(vertices_values[i, [a,b,c]])

    return tris_values

def get_tris_idx(vertex_indices, tris):
    assert len(vertex_indices.shape) == 1
    assert len(tris.shape) == 2

    boolean = np.zeros(len(tris))
    boolean[vertex_indices] = True

    idx = 0
    tris_indices = np.zeros(len(tris), dtype=int)

    for i, (a, b, c) in enumerate(tris):
        if np.any(boolean[[a,b,c]]):
            tris_indices[idx] = i
            idx += 1

    return tris_indices[:idx]

def get_power_per_label(stc, labels, standardize = True):
    activity = [np.empty(len(labels[0])), np.empty(len(labels[1]))]

    for hemi in range(2):
        for i in range(len(labels[hemi])):
            activity[hemi][i] = np.mean(stc.in_label(labels[hemi][i]).data**2)

    if standardize:
        min_act = np.min(np.concatenate((activity[0], activity[1])))
        max_act = np.max(np.concatenate((activity[0], activity[1])))

        activity[0] = (activity[0] - min_act) / (max_act - min_act)
        activity[1] = (activity[1] - min_act) / (max_act - min_act)

    return activity

def activity_to_vertex_values(activity, labels, vertices):
    values = np.zeros(vertices.shape[:2])
    for hemi in range(2):
        for i in range(len(labels[hemi])):
            values[hemi][labels[hemi][i].vertices] = activity[hemi][i]
    return values

def activity_to_source_values(activity, labels, sources):
    values = np.zeros(sources.shape[:2])
    for hemi in range(2):
        for i in range(len(labels[hemi])):
            values[hemi][labels[hemi][i].get_vertices_used()] = activity[hemi][i]
    return values

def vertex_values_to_tris_values(values, tris, func=np.mean):
    # tris_values = get_z_values(sources, tris_sources)
    # tris_colors = cm.viridis(tris_values)

    # for hemi in range(2):
    #     for i in tqdm(range(len(labels[hemi]))):
    #         region_source_indices = labels[hemi][i].get_vertices_used()
    #         region_tris_indices = get_tris_idx(region_source_indices, tris_sources[hemi])
    #         tris_colors[hemi][region_tris_indices] = cm.viridis(values[hemi][i])

    tris_values = np.zeros(tris.shape[:2])
    for hemi in range(2):
        for i, (a, b, c) in enumerate(tris[hemi]):
            tris_values[hemi][i] = func(values[hemi][[a,b,c]])
    return tris_values


def get_window(raw, annotation):
    window = raw.copy().crop(tmin=annotation['onset'],
                             tmax=min(raw.times[-1], annotation['onset']+annotation['duration']))
    
    return window

def get_raw_chops(raw, WINDOW_SIZE=4):
    NUMBER_CHOPS = int(60 / WINDOW_SIZE)
    raw_chops = []

    for i in range(NUMBER_CHOPS):
        try:
            raw_chops.append(raw.copy().crop(i*WINDOW_SIZE, (i+1)*WINDOW_SIZE))
        except:
            pass
        
    return raw_chops

def get_windows_dict_baseline(raw):
    raw_chops = get_raw_chops(raw)
    
    return {'T0': raw_chops}

def get_window_dict(raw, annotations):

    window_dict = {}

    for description in np.unique(annotations.description):

        list_of_windows = []
        for annotation in annotations[annotations.description==description]:
            window = get_window(raw, annotation)
            list_of_windows.append(window)

        window_dict[description] = list_of_windows

    return window_dict



### LOAD DATA ###

def load_mmidb_data_dict(DATA_PATH, PARCELLATION, SNR, chop=True):

    def sort_dict(dict):
        return {k: dict[k] for k in sorted(dict.keys())}

    PARCELLATION_PATH = DATA_PATH + 'mmidb_' + PARCELLATION + '/mmidb_'+ PARCELLATION

    if chop:
        delta_activity = np.load(PARCELLATION_PATH + '_' + str(SNR) + '_1.0_4.0_chop_parallel.npy', allow_pickle=True).item()
        theta_activity = np.load(PARCELLATION_PATH + '_' + str(SNR) + '_4.0_8.0_chop_parallel.npy', allow_pickle=True).item()
        alpha_activity = np.load(PARCELLATION_PATH + '_' + str(SNR) + '_8.0_12.0_chop_parallel.npy', allow_pickle=True).item()
        beta_activity = np.load(PARCELLATION_PATH + '_' + str(SNR) + '_12.0_30.0_chop_parallel.npy', allow_pickle=True).item()
        gamma_activity = np.load(PARCELLATION_PATH + '_' + str(SNR) + '_30.0_70.0_chop_parallel.npy', allow_pickle=True).item()
    else:
        delta_activity = np.load(PARCELLATION_PATH + '_' + str(SNR) + '_1.0_4.0_parallel.npy', allow_pickle=True).item()
        theta_activity = np.load(PARCELLATION_PATH + '_' + str(SNR) + '_4.0_8.0_parallel.npy', allow_pickle=True).item()
        alpha_activity = np.load(PARCELLATION_PATH + '_' + str(SNR) + '_8.0_12.0_parallel.npy', allow_pickle=True).item()
        beta_activity = np.load(PARCELLATION_PATH + '_' + str(SNR) + '_12.0_30.0_parallel.npy', allow_pickle=True).item()
        gamma_activity = np.load(PARCELLATION_PATH + '_' + str(SNR) + '_30.0_70.0_parallel.npy', allow_pickle=True).item()

    data_dict = {
        'Delta': sort_dict(delta_activity),
        'Theta': sort_dict(theta_activity),
        'Alpha': sort_dict(alpha_activity),
        'Beta': sort_dict(beta_activity),
        'Gamma': sort_dict(gamma_activity)
    }

    return data_dict