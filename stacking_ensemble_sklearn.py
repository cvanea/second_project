import pickle
import numpy as np
import mne
from mne.time_frequency import tfr_array_morlet
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from dim_reduction import pca
from utils import get_y_train, get_epochs


def main():
    base_model_type = "lda"
    save_dir = "Results/ensembles/stacking_ensemble"

    freqs = np.logspace(*np.log10([2, 25]), num=15)
    n_cycles = freqs / 4.

    all_results = np.zeros((21, 50))

    for sample in range(21):
        print("sample {}".format(sample))

        epochs = get_epochs(sample + 1, scale=False)
        x_train = epochs.get_data().reshape(epochs.get_data().shape[0], -1)

        time_results = np.zeros(50)

        for time in range(50):
            print("time {}".format(time))
            y_train = get_y_train(sample + 1)

            intervals = np.arange(start=time, stop=y_train.shape[0], step=50)
            y_train = y_train[intervals]

            all_freq_pipelines = []

            for freqency_index in range(15):
                pipe = make_pipeline(WaveletTransform(freqency_index, freqs, n_cycles, epochs.events, time),
                                     LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
                pipe = ("pipe_{}".format(freqency_index), pipe)
                all_freq_pipelines.append(pipe)

            model = StackingClassifier(estimators=all_freq_pipelines, final_estimator=LogisticRegression(), cv=5, stack_method='predict_proba')
            scores = cross_val_score(model, x_train, y_train, cv=5)
            print("Time {} accuracy: %0.2f (+/- %0.2f)".format(time) % (scores.mean(), scores.std() * 2))
            time_results[time] = scores.mean()

        all_results[sample] = time_results


class WaveletTransform(BaseEstimator, TransformerMixin):
    def __init__(self, frequency_index, freqs, n_cycles, epochs_events, time):
        self.frequency_index = frequency_index
        self.freqs = freqs
        self.n_cycles = n_cycles
        self.epochs_events = epochs_events
        self.time = time

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("getting freq {} data".format(self.frequency_index))
        x_train = X.copy()
        x_train = x_train.reshape(X.shape[0], -1, 50)
        print("applying morlet wavelet")
        wavelet_output = tfr_array_morlet(x_train, sfreq=100, freqs=self.freqs, n_cycles=self.n_cycles,
                                          output='complex')
        wavelet_epochs = wavelet_output[:, :, self.frequency_index, :]
        wavelet_epochs = np.append(wavelet_epochs.real, wavelet_epochs.imag, axis=1)

        wavelet_info = mne.create_info(ch_names=wavelet_epochs.shape[1], sfreq=100, ch_types='mag')
        wavelet_epochs = mne.EpochsArray(wavelet_epochs, info=wavelet_info)

        reduced = pca(80, wavelet_epochs, plot=False)
        x_train = reduced.transpose(0, 2, 1).reshape(-1, reduced.shape[1])

        intervals = np.arange(start=self.time, stop=x_train.shape[0], step=50)
        x_sample = x_train[intervals, :]

        return x_sample


if __name__ == "__main__":
    main()
