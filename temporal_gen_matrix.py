import os

import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mne.decoding import GeneralizingEstimator, LinearModel, cross_val_multiscore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.time_frequency import tfr_array_morlet

from utils import get_epochs
from dim_reduction import pca


def main():
    model_type = "lda"
    exp_name = "temporal_gen_matrix/"

    for i, sample in enumerate(range(1, 22)):
        print("sample {}".format(sample))

        if not os.path.isdir("Results/{}/{}/sample_{}".format(model_type, exp_name, sample)):
            os.mkdir("Results/{}/{}/sample_{}".format(model_type, exp_name, sample))

        epochs = get_epochs(sample, scale=False)
        y_train = epochs.events[:, 2]

        freqs = np.logspace(*np.log10([2, 25]), num=15)
        n_cycles = freqs / 4.

        print("applying morlet wavelet")

        wavelet_output = tfr_array_morlet(epochs.get_data(), sfreq=epochs.info['sfreq'], freqs=freqs, n_cycles=n_cycles,
                                          output='complex')

        freq_results = np.zeros((wavelet_output.shape[2], 50, 50))

        for freq in range(wavelet_output.shape[2]):
            print("frequency: {}".format(freqs[freq]))
            curr_freq = str(round(freqs[freq], 2))

            wavelet_epochs = wavelet_output[:, :, freq, :]
            wavelet_epochs = np.append(wavelet_epochs.real, wavelet_epochs.imag, axis=1)

            wavelet_info = mne.create_info(ch_names=wavelet_epochs.shape[1], sfreq=epochs.info['sfreq'], ch_types='mag')
            wavelet_epochs = mne.EpochsArray(wavelet_epochs, info=wavelet_info, events=epochs.events)

            x_train = pca(80, wavelet_epochs, plot=False)

            model = LinearModel(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
            time_gen = GeneralizingEstimator(model, n_jobs=1, scoring='accuracy', verbose=True)
            scores = cross_val_multiscore(time_gen, x_train, y_train, cv=5, n_jobs=1)
            scores = np.mean(scores, axis=0)
            freq_results[freq] = scores

            sns.set()
            ax = sns.lineplot(epochs.times, np.diag(scores))
            ax.set(ylim=(0, 0.8), xlabel='Timepoints', ylabel='Accuracy',
                   title='Cross Val Accuracy {} for Subject {} for Freq {}'.format(model_type, sample, curr_freq))
            ax.axvline(x=0.15, color='b', linestyle='-')
            ax.axhline(0.12, color='k', linestyle='--')
            ax.figure.savefig("Results/{}/{}/sample_{}/freq_{}_accuracy.png"
                              .format(model_type, exp_name, sample, curr_freq), dpi=300)
            plt.clf()

            fig, ax = plt.subplots(1, 1)
            im = ax.imshow(scores, interpolation='lanczos', origin='lower', cmap='RdBu_r',
                           extent=epochs.times[[0, -1, 0, -1]], vmin=0., vmax=0.8)
            ax.set_xlabel('Testing Time (s)')
            ax.set_ylabel('Training Time (s)')
            ax.set_title('Temporal generalization for Subject {} at Freq {}'.format(sample, curr_freq))
            ax.axvline(0.15, color='k', linestyle='--')
            ax.axhline(0.15, color='k', linestyle='--')
            plt.colorbar(im, ax=ax)
            ax.grid(False)
            ax.figure.savefig("Results/{}/{}/sample_{}/freq_{}_matrix.png"
                              .format(model_type, exp_name, sample, curr_freq), dpi=300)
            plt.clf()
            # plt.show()

        freq_results = freq_results.reshape(freq_results.shape[0], -1)
        all_results_df = pd.DataFrame(freq_results)
        all_results_df.to_csv("Results/{}/{}/sample_{}/all_freq_matrix_results.csv".format(model_type, exp_name, sample))


if __name__ == "__main__":
    main()
