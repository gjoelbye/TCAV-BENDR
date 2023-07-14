import os
if os.getcwd().split("/")[-1] != 'BENDR-XAI': os.chdir("../")

import mne
import numpy as np
import matplotlib.pyplot as plt
from utils import *

from matplotlib import animation
import matplotlib.cm as cm
import sys
from tqdm import tqdm

edf_file_path = 'notebooks/S001R10.edf'

subjects_dir, subject, trans, src_path, bem_path = get_fsaverage()
raw = get_raw(edf_file_path)

mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, accept=True)
labels_lh = mne.read_labels_from_annot('fsaverage', 'aparc_sub', 'lh', subjects_dir=subjects_dir, verbose=False)
labels_rh = mne.read_labels_from_annot('fsaverage', 'aparc_sub', 'rh', subjects_dir=subjects_dir, verbose=False)
labels = [labels_lh, labels_rh]

src = get_src(src_path)

vertices = get_vertices(src)
tris_vertices = get_vertices_tris(src)

sources = get_sources(src)
tris_sources = get_sources_tris(src)

tris_values = get_z_values(vertices, tris_vertices)
tris_colors = cm.viridis(tris_values)

for hemi_idx in range(2):
    for i in tqdm(range(len(labels[hemi_idx]))):
        region_source_indices = labels[hemi_idx][i].vertices
        region_tris_indices = get_tris_idx(region_source_indices, tris_vertices[hemi_idx])
        tris_colors[hemi_idx][region_tris_indices] = np.array(cm.tab20(i / len(labels[hemi_idx])))

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1, 1, 1, projection='3d')

def init():
    for i in range(2):
        mesh = ax.plot_trisurf(*vertices[i].T, triangles=tris_vertices[i], linewidth=0.1, shade=False,
                                antialiased=True, edgecolor=(0,0,0,0.5)) #, cmap="viridus")

        mesh.set_facecolors(tris_colors[i])

    ax.set_xlim(-0.06, 0.06)
    ax.set_ylim(-0.08, 0.04)
    ax.set_zlim(-0.04, 0.09)
    ax.set_box_aspect([1,1,1])
    ax.view_init(45, 0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    return fig,


def animate(i):
    ax.view_init(45, i)
    return fig,

# Animate
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=tqdm(range(0, 360, 1), file=sys.stdout), blit=True)
# Save
anim.save('region_animation_high_aparc_sub.gif', fps=30, writer="ffmpeg")
