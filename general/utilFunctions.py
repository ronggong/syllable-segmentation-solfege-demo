import numpy as np


def smooth_obs(obs):
    """
    hanning window smooth the onset observation function
    :param obs: syllable/phoneme onset function
    :return:
    """
    hann = np.hanning(5)
    hann /= np.sum(hann)

    obs = np.convolve(hann, obs, mode='same')

    return obs


def parse_score(filename_score):
    """
    parse the score
    :param filename_score:
    :return: syllable duration array, syllable labels list
    """
    with open(filename_score, 'r') as scorefile:
        data = scorefile.readlines()
        syllable_durations, syllable_labels = [], []
        for line in data:
            syllable_labels.append(line.split()[0])
            syllable_durations.append(float(line.split()[1]))
    syllable_durations = np.array(syllable_durations)
    return syllable_durations, syllable_labels