import pickle

from mat4py import loadmat
import numpy as np
import mne
from mne.decoding import UnsupervisedSpatialFilter
from mne.time_frequency import tfr_array_morlet
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def get_epochs(sample, num_channels=None, scale=False):
    matlab_data = loadmat('SensorSpace/FLISj{}.mat'.format(sample))

    data = matlab_data['data']

    x_train = np.array(data['X_train'])
    x_train = np.transpose(x_train, (1, 0))

    if num_channels:
        x_train = x_train[:num_channels]

    if scale:
        x_train = StandardScaler().fit_transform(x_train)

    y_train = np.array(data['Y_train'])
    y_train = np.transpose(y_train, (1, 0))

    x_train_epochs = np.array(np.array_split(x_train, int(x_train.shape[1] / 50), axis=1))

    print("Trials: {}".format(x_train_epochs.shape[0]))
    print("Channels: {}".format(x_train_epochs.shape[1]))

    info = mne.create_info(ch_names=x_train.shape[0], sfreq=100, ch_types='mag')

    value, time = np.where(y_train == 1)

    events = np.arange(start=0, stop=x_train.shape[1], step=50)
    events = np.expand_dims(events, axis=1)
    events = np.pad(events, ((0, 0), (0, 2)), mode='constant', constant_values=0)

    for i in range(int(x_train.shape[1] / 50)):
        label = i * 50
        ind = np.where(time == label)
        events[i][2] = value[ind]

    epochs = mne.EpochsArray(x_train_epochs, info=info, events=events)

    return epochs


def get_y_train(sample):
    matlab_data = loadmat('SensorSpace/FLISj{}.mat'.format(sample))
    data = matlab_data['data']
    y_train = np.array(data['Y_train'])

    y_train_samples = [np.where(x == 1)[0][0] for x in y_train]
    return np.array(y_train_samples)


def get_y_train_sorted(sample):
    matlab_data = loadmat('SensorSpace/FLISj{}.mat'.format(sample))
    data = matlab_data['data']
    y_train = np.array(data['Y_train'])

    y_train_samples = [np.where(x == 1)[0][0] for x in y_train]
    y_train_samples = np.array(y_train_samples)
    return np.sort(y_train_samples)


def get_raw_data(sample, scale=False):
    matlab_data = loadmat('SensorSpace/FLISj{}.mat'.format(sample))
    data = matlab_data['data']
    x_train = np.array(data['X_train'])
    y_train = np.array(data['Y_train'])

    y_train_samples = [np.where(x == 1)[0][0] for x in y_train]
    y_train_samples = np.array(y_train_samples)

    if scale:
        x_train = StandardScaler().fit_transform(x_train)

    return x_train, y_train_samples

def save_wavelet_complex():
    all_x_train_samples = []

    for sample in range(1, 22):
        print("sample {}".format(sample))
        epochs = get_epochs(sample, scale=False)
        freqs = np.logspace(*np.log10([2, 25]), num=15)
        n_cycles = freqs / 4.

        print("applying morlet wavelet")
        wavelet_output = tfr_array_morlet(epochs.get_data(), sfreq=epochs.info['sfreq'], freqs=freqs, n_cycles=n_cycles,
                                          output='complex')

        all_x_train_freqs = []

        for freq in range(wavelet_output.shape[2]):
            print("frequency: {}".format(freqs[freq]))

            wavelet_epochs = wavelet_output[:, :, freq, :]
            wavelet_epochs = np.append(wavelet_epochs.real, wavelet_epochs.imag, axis=1)

            wavelet_info = mne.create_info(ch_names=wavelet_epochs.shape[1], sfreq=epochs.info['sfreq'], ch_types='mag')
            wavelet_epochs = mne.EpochsArray(wavelet_epochs, info=wavelet_info, events=epochs.events)

            pca = UnsupervisedSpatialFilter(PCA(n_components=80), average=False)
            print('fitting pca')
            reduced = pca.fit_transform(wavelet_epochs.get_data())
            print('fitting done')

            x_train = reduced.transpose(0, 2, 1).reshape(-1, reduced.shape[1])
            all_x_train_freqs.append(x_train)

        all_x_train_samples.append(all_x_train_freqs)

    print('saving x_train for all samples')
    pickle.dump(all_x_train_samples, open("DataTransformed/wavelet_complex/x_train_all_samples.pkl", "wb"))
    print("x_train saved")


def get_freq_pipelines(num_freqs):
    all_freq_pipelines = []
    for freqency_index in range(num_freqs):
        pipe = make_pipeline(WaveletTransform(freqency_index),
                             LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        pipe = ("pipe_{}".format(freqency_index), pipe)
        all_freq_pipelines.append(pipe)
    return all_freq_pipelines


class WaveletTransform(BaseEstimator, TransformerMixin):
    def __init__(self, frequency_index):
        self.frequency_index = frequency_index

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        x_train = X.copy()
        x_train = x_train[:, np.arange(start=self.frequency_index * 80, stop=(self.frequency_index * 80) + 80, step=1)]
        return x_train