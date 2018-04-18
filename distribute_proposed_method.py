import os
import pickle
import numpy as np
import soundfile as sf
from keras.models import load_model
from audio_preprocessing import get_log_mel_madmom
from audio_preprocessing import feature_reshape

import pyximport
pyximport.install(reload_support=True,
                  setup_args={'include_dirs': np.get_include()})

import viterbiDecodingPhonemeSeg

from general.parameters import hopsize_t
from general.parameters import varin
from general.utilFunctions import smooth_obs

from plot_code import figure_plot_joint

root_path = os.path.join(os.path.dirname(__file__))

joint_cnn_model_path = os.path.join(root_path, 'cnnModels', 'joint')

# load keras joint cnn model
model_joint = load_model(os.path.join(joint_cnn_model_path, 'jan_joint0.h5'))

# load log mel feature scaler
scaler_joint = pickle.load(open(os.path.join(joint_cnn_model_path, 'scaler_joint.pkl'), 'rb'), encoding='latin')

# load wav, duration and labels
wav_file = './Mbrola/solfege_demo_44.wav'

# parse score
with open('./solfege_score.txt', 'r') as scorefile:
    data = scorefile.readlines()
    syllable_durations, syllable_labels = [], []
    for line in data:
        syllable_labels.append(line.split()[0])
        syllable_durations.append(float(line.split()[1]))
syllable_durations = np.array(syllable_durations)

print('syllable durations (second):')
print(syllable_durations)
print('\n')

print('syllable labels:')
print(syllable_labels)
print('\n')

# get wav duration
data_wav, fs_wav = sf.read(wav_file)
time_wav = len(data_wav)/float(fs_wav)

# calculate log mel feature
log_mel_old = get_log_mel_madmom(wav_file, fs=fs_wav, hopsize_t=hopsize_t, channel=1)
log_mel = scaler_joint.transform(log_mel_old)
log_mel = feature_reshape(log_mel, nlen=7)
log_mel = np.expand_dims(log_mel, axis=1)

# get the onset detection function
obs_syllable, obs_phoneme = model_joint.predict(log_mel, batch_size=128, verbose=2)

# post-processing the detection function
obs_syllable = np.squeeze(obs_syllable)
obs_phoneme = np.squeeze(obs_phoneme)

obs_syllable = smooth_obs(obs_syllable)
obs_phoneme = smooth_obs(obs_phoneme)

obs_syllable[0] = 1.0
obs_syllable[-1] = 1.0

# normalize the syllable durations
syllable_durations *= time_wav / np.sum(syllable_durations)

# decoding syllable boundaries
boundaries_syllable = viterbiDecodingPhonemeSeg.viterbiSegmental2(obs_syllable, syllable_durations, varin)

# syllable boundaries
boundaries_syllable_start_time = np.array(boundaries_syllable[:-1])*hopsize_t
boundaries_syllable_end_time = np.array(boundaries_syllable[1:])*hopsize_t

print('Detected syllable onset times (second):')
print(boundaries_syllable_start_time)
print('\n')

figure_plot_joint(mfcc_line=log_mel_old,
                  obs_syllable=obs_syllable,
                  boundaries_syllable_start_time=boundaries_syllable_start_time)