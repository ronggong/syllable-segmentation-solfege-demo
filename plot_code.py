# -*- coding: utf-8 -*-
"""some plot function mainly for debugging,
used in proposed_method_pipeline.py"""
import matplotlib
matplotlib.use('TkAgg')

from general.parameters import *
import numpy as np
import matplotlib.pyplot as plt


def figure_plot_joint(mfcc_line,
                      obs_syllable,
                      boundaries_syllable_start_time):
    # plot Error analysis figures
    plt.figure(figsize=(16, 4))
    # class weight
    ax1 = plt.subplot(211)
    y = np.arange(0, 80)
    x = np.arange(0, mfcc_line.shape[0]) * hopsize_t
    plt.pcolormesh(x, y, np.transpose(mfcc_line[:, 80 * 7:80 * 8]))

    ax1.set_ylabel('Mel bands', fontsize=12)
    ax1.get_xaxis().set_visible(False)
    ax1.axis('tight')

    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(np.arange(0, len(obs_syllable)) * hopsize_t, obs_syllable)
    for bsst in boundaries_syllable_start_time:
        plt.axvline(bsst, color='r', linewidth=2)

    ax2.set_ylabel('ODF syllable', fontsize=12)
    ax2.axis('tight')

    plt.show()

