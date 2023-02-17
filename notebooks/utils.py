import mne
from mne.datasets import fetch_fsaverage
import numpy as np
from pathlib import Path
import os

import vtk
from vtk import vtkPolyData, vtkDecimatePro
from vtk.util.numpy_support import vtk_to_numpy


def get_raw(edf_file_path, filter = True):
    raw = mne.io.read_raw_edf(edf_file_path, verbose=False, preload=True)
    mne.datasets.eegbci.standardize(raw)  # Set channel names
    montage = mne.channels.make_standard_montage('standard_1020')
    
    raw = raw.set_eeg_reference(ref_channels='average', projection=True, verbose = False)
    raw = raw.set_montage(montage); # Set montage

    if filter:
        raw = raw.filter(0.5, 70, verbose = False)
        raw = raw.notch_filter(60, verbose = False)



    return raw

def get_fsaverage(verbose = False):
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
                                     meg=False, eeg=True, verbose=False, n_jobs=4)

    #fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose=False)
    # leadfield = fwd['sol']['data']

    return fwd


def get_cov(raw):
    cov = mne.compute_raw_covariance(raw, verbose=False)
    return cov


def get_stc(raw, fwd, cov, tmin=0, tmax=10, snr = 1.0, verbose = False):

    idx_start = int(tmin * raw.info['sfreq'])
    idx_stop = int(tmax * raw.info['sfreq'])

    lambda2 = 1.0 / snr ** 2 # Use smaller SNR for raw data
    method = "eLORETA"

    inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, verbose=verbose)
    stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2, method=method,
                                             nave = 1, start=idx_start, stop=idx_stop,
                                             verbose=verbose)

    return stc

def get_labels(subjects_dir, parcellation_name = "aparc_sub", verbose = False):
    mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, accept=True)

    labels_lh = mne.read_labels_from_annot('fsaverage', parcellation_name, 'lh', subjects_dir=subjects_dir, verbose=False) 
    labels_rh = mne.read_labels_from_annot('fsaverage', parcellation_name, 'rh', subjects_dir=subjects_dir, verbose=False)
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

def vertex_values_to_tris_values(tris_sources, values, func=np.mean):
    # tris_values = get_z_values(sources, tris_sources)
    # tris_colors = cm.viridis(tris_values)

    # for hemi in range(2):
    #     for i in tqdm(range(len(labels[hemi]))):
    #         region_source_indices = labels[hemi][i].get_vertices_used()
    #         region_tris_indices = get_tris_idx(region_source_indices, tris_sources[hemi])
    #         tris_colors[hemi][region_tris_indices] = cm.viridis(values[hemi][i])

    tris_values = np.zeros(tris_sources.shape[:2])
    for hemi in range(2):
        for i, (a, b, c) in enumerate(tris_sources[hemi]):
            tris_values[hemi][i] = func(values[hemi][[a,b,c]])
    return tris_values