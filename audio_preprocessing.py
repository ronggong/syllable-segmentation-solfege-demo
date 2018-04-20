from madmom.processors import SequentialProcessor
from general.Fprev_sub import Fprev_sub
import soundfile as sf
import numpy as np
import webrtcvad
import contextlib
import resampy
import wave
import os

EPSILON = np.spacing(1)


def _nbf_2D(log_mel, nlen):
    """shift the feature and concatenate it in both left and right sides for nlen"""

    log_mel = np.array(log_mel).transpose()
    log_mel_out = np.array(log_mel, copy=True)
    for ii in range(1, nlen + 1):
        log_mel_right_shift = Fprev_sub(log_mel, w=ii)
        log_mel_left_shift = Fprev_sub(log_mel, w=-ii)
        log_mel_out = np.vstack((log_mel_right_shift, log_mel_out, log_mel_left_shift))
    feature = log_mel_out.transpose()
    return feature


class MadmomMelbankProcessor(SequentialProcessor):

    def __init__(self, fs, hopsize_t):
        from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
        from madmom.audio.stft import ShortTimeFourierTransformProcessor
        from madmom.audio.filters import MelFilterbank
        from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                              LogarithmicSpectrogramProcessor)

        # define pre-processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=fs)
        frames = FramedSignalProcessor(frame_size=2048, hopsize=int(fs*hopsize_t))
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(
            filterbank=MelFilterbank, num_bands=80, fmin=27.5, fmax=16000,
            norm_filters=True, unique_filters=False)
        spec = LogarithmicSpectrogramProcessor(log=np.log, add=EPSILON)

        single = SequentialProcessor([frames, stft, filt, spec])

        pre_processor = SequentialProcessor([sig, single])

        super(MadmomMelbankProcessor, self).__init__([pre_processor])


def get_log_mel_madmom(audio_fn, fs, hopsize_t, channel):
    """
    calculate log mel feature by madmom
    :param audio_fn:
    :param fs:
    :param hopsize_t:
    :param channel:
    :return:
    """
    madmomMelbankProc = MadmomMelbankProcessor(fs, hopsize_t)
    mfcc = madmomMelbankProc(audio_fn)

    if channel == 1:
        mfcc = _nbf_2D(mfcc, 7)
    else:
        mfcc_conc = []
        for ii in range(3):
            mfcc_conc.append(_nbf_2D(mfcc[:,:,ii], 7))
        mfcc = np.stack(mfcc_conc, axis=2)
    return mfcc


def feature_reshape(feature, nlen=10):
    """
    reshape mfccBands feature into n_sample * n_row * n_col
    :param feature:
    :param nlen:
    :return:
    """

    n_sample = feature.shape[0]
    n_row = 80
    n_col = nlen*2+1

    feature_reshaped = np.zeros((n_sample,n_row,n_col),dtype='float32')
    # print("reshaping feature...")
    for ii in range(n_sample):
        # print ii
        feature_frame = np.zeros((n_row,n_col),dtype='float32')
        for jj in range(n_col):
            feature_frame[:,jj] = feature[ii][n_row*jj:n_row*(jj+1)]
        feature_reshaped[ii,:,:] = feature_frame
    return feature_reshaped


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, hopsize_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    framesize = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    hopsize = int(sample_rate * (hopsize_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    offset_timestamp = (float(hopsize) / sample_rate) / 2.0
    duration = (float(framesize) / sample_rate) / 2.0
    while offset + framesize < len(audio):
        yield Frame(audio[offset:offset + framesize], timestamp, duration)
        timestamp += offset_timestamp
        offset += hopsize


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def VAD(wav_file, hopsize_t):

    resample_fs = 32000
    current_path = os.path.dirname(os.path.realpath(__file__))
    path_temp_wav = os.path.join(current_path, '.', 'temp', 'temp.wav')
    wav_data, wav_fs = sf.read(wav_file)
    vad_results = np.array([], dtype=np.bool)

    # convert the audio to the 1 channel
    if len(wav_data.shape) == 2:
        if wav_data.shape[1] == 2:
            wav_data = (wav_data[:, 0] + wav_data[:, 1]) / 2.0

    # resample the audio samples
    wav_data_32000 = resampy.resample(wav_data, wav_fs, resample_fs)

    # write the audio
    sf.write(path_temp_wav, wav_data_32000, resample_fs)

    # read the wav in bytes, the length will be 2 times of the normal wav data
    wav_data, wav_fs = read_wave(path_temp_wav)

    # gnerate frames
    frames = frame_generator(frame_duration_ms=30, hopsize_ms=hopsize_t*1000, audio=wav_data, sample_rate=wav_fs)

    vad = webrtcvad.Vad()

    # mode 0-3, 3 is the most aggressive one
    vad.set_mode(2)
    for frame in frames:
        is_speech = vad.is_speech(buf=frame.bytes, sample_rate=wav_fs)
        vad_results = np.append(vad_results, is_speech)

    return vad_results.astype(int)
