# -*- coding: utf-8 -*-
"""some plot function mainly for debugging,
used in proposed_method_pipeline.py"""
import matplotlib
matplotlib.use('TkAgg')

from general.parameters import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def figure_plot_joint(score_png,
                      mfcc_line,
                      obs_syllable,
                      boundaries_syllable_start_time,
                      labels_syllable):
    # plot Error analysis figures
    plt.figure(figsize=(16, 6))
    # class weight
    ax1 = plt.subplot(311)
    img = mpimg.imread(score_png)
    ax1.imshow(img)

    ax2 = plt.subplot(312)
    y = np.arange(0, 80)
    x = np.arange(0, mfcc_line.shape[0]) * hopsize_t
    plt.pcolormesh(x, y, np.transpose(mfcc_line[:, 80 * 7:80 * 8]))

    ax2.set_ylabel('Mel bands', fontsize=12)
    ax2.get_xaxis().set_visible(False)
    ax2.axis('tight')

    ax3 = plt.subplot(313, sharex=ax2)
    plt.plot(np.arange(0, len(obs_syllable)) * hopsize_t, obs_syllable)
    for bsst in boundaries_syllable_start_time:
        plt.axvline(bsst, color='r', linewidth=2)
    for ii_bsst, bsst in enumerate(boundaries_syllable_start_time):
        ax3.annotate(labels_syllable[ii_bsst], xy=(bsst, 1.0), xytext=(bsst, 1.0))

    ax3.set_ylabel('ODF syllable', fontsize=12)
    ax3.axis('tight')

    plt.show()

