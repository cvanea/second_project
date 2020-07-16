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
    exp_name = "freq_gen_matrix/"

    for i, sample in enumerate(range(1, 22)):
        print("sample {}".format(sample))

        if not os.path.isdir("Results/{}/{}/sample_{}".format(model_type, exp_name, sample)):
            os.mkdir("Results/{}/{}/sample_{}".format(model_type, exp_name, sample))

        epochs = get_epochs(sample, scale=False)
        y_train = epochs.events[:, 2]

        freqs = np.logspace(*np.log10([2, 25]), num=15)
        n_cycles = freqs / 4.
        string_freqs = [round(x, 2) for x in freqs]

        print("applying morlet wavelet")

        wavelet_output = tfr_array_morlet(epochs.get_data(), sfreq=epochs.info['sfreq'], freqs=freqs, n_cycles=n_cycles,
                                          output='complex')

        time_results = np.zeros((wavelet_output.shape[3], len(freqs), len(freqs)))

        for time in range(wavelet_output.shape[3]):
            print("time: {}".format(time))

            wavelet_epochs = wavelet_output[:, :, :, time]
            wavelet_epochs = np.append(wavelet_epochs.real, wavelet_epochs.imag, axis=1)

            wavelet_info = mne.create_info(ch_names=wavelet_epochs.shape[1], sfreq=epochs.info['sfreq'], ch_types='mag')
            wavelet_epochs = mne.EpochsArray(wavelet_epochs, info=wavelet_info, events=epochs.events)

            x_train = pca(80, wavelet_epochs, plot=False)

            model = LinearModel(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
            freq_gen = GeneralizingEstimator(model, n_jobs=1, scoring='accuracy', verbose=True)
            scores = cross_val_multiscore(freq_gen, x_train, y_train, cv=5, n_jobs=1)
            scores = np.mean(scores, axis=0)
            time_results[time] = scores

            sns.set()
            ax = sns.barplot(np.sort(string_freqs), np.diag(scores), )
            ax.set(ylim=(0, 0.8), xlabel='Frequencies', ylabel='Accuracy',
                   title='Cross Val Accuracy {} for Subject {} for Time {}'.format(model_type, sample, time))
            ax.axhline(0.12, color='k', linestyle='--')
            ax.figure.set_size_inches(8, 6)
            ax.figure.savefig("Results/{}/{}/sample_{}/time_{}_accuracy.png"
                              .format(model_type, exp_name, sample, time), dpi=300)
            plt.close('all')
            # plt.show()

            fig, ax = plt.subplots(1, 1)
            im = ax.imshow(scores, interpolation='lanczos', origin='lower', cmap='RdBu_r',
                           extent=[2, 25, 2, 25], vmin=0., vmax=0.8)
            ax.set_xlabel('Testing Frequency (hz)')
            ax.set_ylabel('Training Frequency (hz)')
            ax.set_title('Frequency generalization for Subject {} at Time {}'.format(sample, time))
            plt.colorbar(im, ax=ax)
            ax.grid(False)
            ax.figure.savefig("Results/{}/{}/sample_{}/time_{}_matrix.png"
                              .format(model_type, exp_name, sample, time), dpi=300)
            plt.close('all')
            # plt.show()

        time_results = time_results.reshape(time_results.shape[0], -1)
        all_results_df = pd.DataFrame(time_results)
        all_results_df.to_csv("Results/{}/{}/sample_{}/all_time_matrix_results.csv".format(model_type, exp_name, sample))


if __name__ == "__main__":
    main()
