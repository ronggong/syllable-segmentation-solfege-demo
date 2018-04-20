import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis


class FeatureExtraction(object):
    """
    extract rhythmic beat deviation features
    """
    def __init__(self,
                 onset_time_ref,
                 syllable_durations_ref,
                 onset_time_detected,
                 syllable_durations_detected,
                 beats):
        self.onset_time_ref = onset_time_ref
        self.syllable_durations_ref = syllable_durations_ref
        self.onset_time_detected = onset_time_detected
        self.syllable_durations_detected = syllable_durations_detected
        self.beats = beats

    def onset_deviation(self):
        return np.abs(self.onset_time_ref - self.onset_time_detected)

    def syllable_durations_weighted_onset_deviation(self, od):
        return od/self.syllable_durations_ref

    def duration_deviation(self):
        return np.abs(self.syllable_durations_ref - self.syllable_durations_detected)

    def syllable_durations_weighted_duration_deviation(self, dd):
        return dd/self.syllable_durations_ref

    def on_beat_deviation(self, deviation):
        indices = [i for i, x in enumerate(self.beats) if x == "on"]
        return deviation[indices]

    def off_beat_deviation(self, deviation):
        indices = [i for i, x in enumerate(self.beats) if x == "off"]
        return deviation[indices]

    def other_beat_deviation(self, deviation):
        indices = [i for i, x in enumerate(self.beats) if x is None]
        return deviation[indices]

    @staticmethod
    def statistics_deviation(deviation):
        return [np.min(deviation), np.max(deviation), np.median(deviation),
                np.mean(deviation), np.std(deviation), skew(deviation), kurtosis(deviation)]


if __name__ == '__main__':
    # test variables
    onset_time_ref = np.array([0.,         2.72727891, 3.06818878, 3.40909864, 3.7500085,  4.09091837,
                              4.43182823, 4.7727381,  5.11364796, 5.45455782, 6.13637755, 6.81819728,
                              7.50001701, 8.18183673, 9.54547619])
    syllable_durations_ref = np.array([2.72727891, 0.34090986, 0.34090986, 0.34090986, 0.34090986, 0.34090986,
                                      0.34090986, 0.34090986, 0.34090986, 0.68181973, 0.68181973, 0.68181973,
                                      0.68181973, 1.36363946, 1.36363946])
    onset_time_detected = np.array([0.,   2.59, 3.02, 3.3,  3.69, 4.,
                                    4.35, 4.71, 5.04, 5.39, 6.07, 6.54, 7.3,  7.91, 9.56])
    syllable_durations_detected = np.array([2.59, 0.43, 0.28, 0.39,
                                            0.31, 0.35, 0.36, 0.33,
                                            0.35, 0.68, 0.47, 0.76,
                                            0.61, 1.65, 1.34])
    beats = [None, 'on', None, 'off', None, 'on', None, 'off', None, 'on', 'off', 'on', 'off', 'on', 'on']

    fe = FeatureExtraction(onset_time_ref=onset_time_ref[1:],
                           syllable_durations_ref=syllable_durations_ref[1:],
                           onset_time_detected=onset_time_detected[1:],
                           syllable_durations_detected=syllable_durations_detected[1:],
                           beats=beats[1:])

    # general features
    od = fe.onset_deviation()
    sdwod = fe.syllable_durations_weighted_onset_deviation(od)
    dd = fe.duration_deviation()
    sdwdd = fe.syllable_durations_weighted_duration_deviation(dd)

    # on beat features
    od_on = fe.on_beat_deviation(od)
    sdwod_on = fe.on_beat_deviation(sdwod)
    dd_on = fe.on_beat_deviation(dd)
    sdwdd_on = fe.on_beat_deviation(sdwdd)

    # off beat features
    od_off = fe.off_beat_deviation(od)
    sdwod_off = fe.off_beat_deviation(sdwod)
    dd_off = fe.off_beat_deviation(dd)
    sdwdd_off = fe.off_beat_deviation(sdwdd)

    # other beats features
    od_other = fe.other_beat_deviation(od)
    sdwod_other = fe.other_beat_deviation(sdwod)
    dd_other = fe.other_beat_deviation(dd)
    sdwdd_other = fe.other_beat_deviation(sdwdd)

    # calculate feature statistics
    feature_set = fe.statistics_deviation(od) + fe.statistics_deviation(sdwod) + \
                  fe.statistics_deviation(dd) + fe.statistics_deviation(sdwdd)

    print(feature_set)