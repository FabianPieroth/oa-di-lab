# ------------------------------------------------------------------------------
#  File: generate_unmixing_figures.py
#  Author: Jan Kukacka
#  Date: 11/2018
# ------------------------------------------------------------------------------
#  Script to generate linear unmixing figures
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import happy as hp
import happy.plots as hpp
from pathlib import Path


def generate_figure(img, us, regressed=True, slice=None, logging_for_documentation=False):
    '''
    Creates plot with unmixing results

    # Arguments:
      - img: np array with OA image of shape (height, width, n_wavelengths)
      - us: np array with ultrasound image of shape (us_height, us_width)

    # Usage:

     project_root_dir = str(Path().resolve().parents[1])
     path_to_raw_files = project_root_dir + '/data/homo/raw/new_in/Study16/Scan_2/'

     img = hp.io.load(path_to_raw_files + 'OA_data_singleSoS_Scan_2.mat', key='OA_high')
     us = hp.io.load(path_to_raw_files + 'US_data_singleSoS_Scan_2.mat', key='US_high')
     fig = generate_figure(img, us)
     fig.show()
    '''
    if np.argmin(img.shape) == 0:
        img = np.moveaxis(img, source=[0, 1, 2], destination=[2, 0, 1])
    h,w,_ = img.shape
    chromophore_names = ['Hb', 'HbO2', 'Fat', 'Water']
    spectra = hp.oa.spectra.clinical()
    # take the slice of the spectra
    if not regressed:
        spectra = spectra[:, slice]
        unmix = hp.oa.unmixing.linear_unmixing(img, spectra=spectra, non_negative=True)
        blood_signal_ratio = hp.oa.analysis.blood_signal_ratio(img, damping=1e-5, spectra=spectra, unmixed=unmix)
    else:
        unmix = img
        blood_signal_ratio = np.zeros((h,w))

    ## Cut-off negative coefficients (when not using non_negative=True)
    unmix = np.maximum(0, unmix)

    # blood_signal_ratio = hp.oa.analysis.blood_signal_ratio(img, damping=1e-5, spectra=spectra, unmixed=unmix)

    ## Number of chromophores
    d = unmix.shape[-1]

    ## Total blood volume
    tbv = np.sum(unmix[...,:2], axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ## Blood oxygenation level
        so2 = np.nan_to_num(unmix[...,1] / tbv)
    if logging_for_documentation:
        fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 5))
    else:
        fig, axes = plt.subplots(1,d, figsize=(5*d,5))

    if us is None:
        us = np.zeros((h,w))

    ## Normalize ultrasound
    if np.max(us) == np.min(us):
        us = us - np.min(us)
    else:
        us = (us - np.min(us)) / (np.max(us) - np.min(us))

    ## Total blood volume plot
    ax = axes[0]
    ax.imshow(us, extent=(-.5,w-.5,h-.5, -.5), cmap='gray')
    im = ax.imshow(tbv, cmap=hpp.cmap(['t', 'orange', 'yellow']), norm=PowerNorm(gamma=.5))
    if logging_for_documentation:
        ax.axis('off')
    else:
        hpp.set_ticks_cm(ax, 100, tbv.shape)
    ax.set_title('Total blood volume')

    if not logging_for_documentation:
        ## SO2 plot
        ax = axes[1]
        ax.imshow(us, extent=(-.5,w-.5,h-.5, -.5), cmap='gray')
        c = hpp.cmap(['xkcd:apple green', 'xkcd:tomato red'])(so2)
        c[...,-1] = blood_signal_ratio
        im = ax.imshow(c)

        ## black colorbar background
        c = np.zeros_like(c)
        c[-20:,:,-1] = 1
        im = ax.imshow(c)
        ## colorbar
        c = np.zeros_like(c)
        colorbar = np.linspace(0,1,w)
        c[-20:] = hpp.cmap(['xkcd:apple green', 'xkcd:tomato red'])(colorbar)[None]
        c[-20:,:,-1] = np.linspace(0,1,20)[::-1,None]
        im = ax.imshow(c)
        ## colorbar text
        ax.text(10,h-10,'0%', color='w', verticalalignment='center', fontsize=8)
        ax.text(w-10,h-10,'100%', color='w', verticalalignment='center', horizontalalignment='right', fontsize=8)
        ax.text(w/2,h-10,'sO$_2$', color='w', verticalalignment='center', horizontalalignment='center', fontsize=8)

        hpp.set_ticks_cm(ax, 100, tbv.shape)
        ax.set_title('Blood oxygenation')

    cmaps = [hpp.cmap(['t', 'xkcd:darkish purple', 'xkcd:pinky purple']),
             hpp.cmap(['t', 'xkcd:dark sky blue', 'xkcd:bright sky blue'])]
    if logging_for_documentation:
        for j in range(2,d):
            ax = axes[j-1]
            ax.imshow(us, extent=(-.5,w-.5,h-.5, -.5), cmap='gray')
            im = ax.imshow(unmix[..., j], cmap=cmaps[j-2], norm=PowerNorm(gamma=.5))
            if logging_for_documentation:
                ax.axis('off')
            else:
                hpp.set_ticks_cm(ax, 100, tbv.shape)
            ax.set_title(chromophore_names[j])
    else:
        for j in range(2,d):
            ax = axes[j]
            ax.imshow(us, extent=(-.5,w-.5,h-.5, -.5), cmap='gray')
            im = ax.imshow(unmix[..., j], cmap=cmaps[j-2], norm=PowerNorm(gamma=.5))
            hpp.set_ticks_cm(ax, 100, tbv.shape)
            ax.set_title(chromophore_names[j])

    return fig


def main():
    project_root_dir = str(Path().resolve().parents[2])
    path_to_raw_files = project_root_dir + '/data/homo/raw/new_in/Study26/Scan_2/'

    img = hp.io.load(path_to_raw_files + 'OA_data_singleSoS_Scan_2.mat', key='OA_high')
    # us = hp.io.load(path_to_raw_files + 'US_data_singleSoS_Scan_2.mat', key='US_high')
    us = np.zeros((401,401))
    fig = generate_figure(img, us)
    fig.show()


if __name__ == "__main__":
    main()
