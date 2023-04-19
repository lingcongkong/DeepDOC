import numpy as np
from nilearn import plotting
from nilearn import surface
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from tqdm import *
import os
import glob
import nibabel as nib
import pickle
from nilearn import datasets

with open('./config.txt', 'rb') as f:
    config = pickle.load(f)
    
    
def get_surf(config, stat_img, save_dir, fig, hemi='rh'):
    fsaverage = datasets.fetch_surf_fsaverage()
    destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
    if hemi == 'rh':
        texture = surface.vol_to_surf(stat_img, fsaverage.pial_right)
        parcellation = destrieux_atlas['map_right']
        regions_dict = config['yellow']
        labels = regions_dict

        # get indices in atlas for these labels
        regions_indices = [np.where(np.array(destrieux_atlas['labels']) == region)[0][0]
                           for region in regions_dict]

        # labels = list(regions_dict.values())

        fig = plt.figure(figsize=(10, 10))
        plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi='right',
        #                                      title='Surface right hemisphere',
                                             colorbar=False, threshold=th,
                                             bg_map=fsaverage.sulc_right, figure=fig,view=view[1])
        plotting.plot_surf_contours(fsaverage.infl_right, parcellation, labels=labels,
                                    levels=regions_indices, figure=fig, legend=False,
                                    colors = [nw_color[i] for i in labels]
        #                             colors=['g', 'k']
                                   )
        fig.savefig(f'{savedir}/rh_{config["view[1]"]}_yellow.{config["fm"]}', dpi=700)
    else:
        texture = surface.vol_to_surf(stat_img, fsaverage.pial_left)
        parcellation = destrieux_atlas['map_left']
        regions_dict = config['yellow']
        labels = regions_dict

        # get indices in atlas for these labels
        regions_indices = [np.where(np.array(destrieux_atlas['labels']) == region)[0][0]
                           for region in regions_dict]

        # labels = list(regions_dict.values())

        fig = plt.figure(figsize=(10, 10))
        plotting.plot_surf_stat_map(fsaverage.infl_left, texture, hemi='left',
        #                                      title='Surface right hemisphere',
                                             colorbar=False, threshold=config['th'],
                                             bg_map=fsaverage.sulc_left, figure=fig,view=config['view'][1])
        plotting.plot_surf_contours(fsaverage.infl_left, parcellation, labels=labels,
                                    levels=regions_indices, figure=fig, legend=False,
                                    colors = [nw_color[i] for i in labels]
        #                             colors=['g', 'k']
                                   )
        fig.savefig(f'{savedir}/rh_{config["view"][1]}_yellow.{config["fm"]}', dpi=700)

