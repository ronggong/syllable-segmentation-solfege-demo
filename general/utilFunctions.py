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
        syllable_durations, syllable_labels, beats = [], [], []
        tempo = float(data[0].split()[1])
        for line in data[1:]:
            list_line = line.split()
            if len(list_line) == 3:
                beats.append(list_line[2])
            else:
                beats.append(None)
            syllable_labels.append(list_line[0])
            syllable_durations.append(float(list_line[1]))
    syllable_durations = np.array(syllable_durations)
    return tempo, syllable_durations, syllable_labels, beats


def get_onset_time_syllable_duration_ref(syllable_durations, len_audio):
    """
    get onset time positions from the syllable durations
    :param syllable_durations:
    :param len_audio:
    :return:
    """
    # normalize the syllable durations
    sd_norm = syllable_durations / np.sum(syllable_durations)

    onset_time_norm = np.cumsum(sd_norm)

    # insert the 0 to the beginning or the excerpt
    onset_time_norm = np.insert(onset_time_norm[:-1], 0, 0.0)

    onset_time = onset_time_norm * len_audio

    return onset_time, sd_norm * len_audio


if __name__ == '__main__':
    filename_score = '../data/score_exercise_01.txt'
    tempo, syllable_durations, syllable_lists, beats = parse_score(filename_score=filename_score)
    print(tempo)
    print(syllable_durations)
    print(syllable_lists)
    print(beats)

    get_onset_time_syllable_duration_ref(syllable_durations=syllable_durations,
                                         len_audio=1.0)