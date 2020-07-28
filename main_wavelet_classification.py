import os

import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mne.time_frequency import tfr_array_morlet

from utils import get_epochs, get_y_train
from dim_reduction import kernel_pca, ica, pca
from models import linear_models, nonlinear_models


def main():
    model_type = "lda"
    exp_name = "wavelet_class/lsqr/complex"

    for sample in range(1, 22):
        print("sample {}".format(sample))

        if not os.path.isdir("Results/{}/{}/sample_{}".format(model_type, exp_name, sample)):
            os.mkdir("Results/{}/{}/sample_{}".format(model_type, exp_name, sample))

        epochs = get_epochs(sample, scale=False)

        freqs = np.logspace(*np.log10([2, 25]), num=15)
        n_cycles = freqs / 4.

        print("applying morlet wavelet")

        # returns (n_epochs, n_channels, n_freqs, n_times)
        if exp_name.split("/")[-1] == "real" or exp_name.split("/")[-1] == "complex":
            wavelet_output = tfr_array_morlet(epochs.get_data(), sfreq=epochs.info['sfreq'], freqs=freqs,
                                              n_cycles=n_cycles, output='complex')
        elif exp_name.split("/")[-1] == "power":
            wavelet_output = tfr_array_morlet(epochs.get_data(), sfreq=epochs.info['sfreq'], freqs=freqs,
                                              n_cycles=n_cycles, output='power')
        elif exp_name.split("/")[-1] == "phase":
            wavelet_output = tfr_array_morlet(epochs.get_data(), sfreq=epochs.info['sfreq'], freqs=freqs,
                                              n_cycles=n_cycles, output='phase')
        else:
            raise ValueError("{} not an output of wavelet function".format(exp_name.split("/")[-1]))

        y_train = get_y_train(sample)

        freq_results = np.zeros((wavelet_output.shape[2], 50))

        for freq in range(wavelet_output.shape[2]):
            print("frequency: {}".format(freqs[freq]))

            wavelet_epochs = wavelet_output[:, :, freq, :]

            if exp_name.split("/")[-1] == "real":
                wavelet_epochs = wavelet_epochs.real
            if exp_name.split("/")[-1] == "complex":
                wavelet_epochs = np.append(wavelet_epochs.real, wavelet_epochs.imag, axis=1)

            wavelet_info = mne.create_info(ch_names=wavelet_epochs.shape[1], sfreq=epochs.info['sfreq'], ch_types='mag')
            wavelet_epochs = mne.EpochsArray(wavelet_epochs, info=wavelet_info, events=epochs.events)

            reduced = pca(80, wavelet_epochs, plot=False)
            x_train = reduced.transpose(0, 2, 1).reshape(-1, reduced.shape[1])

            results = linear_models(x_train, y_train, model_type=model_type)
            freq_results[freq] = results

            curr_freq = str(round(freqs[freq], 2))

            sns.set()
            ax = sns.lineplot(data=results, dashes=False)
            ax.set(ylim=(0, 0.7), xlabel='Time', ylabel='Accuracy',
                   title='Cross Val Accuracy {} for Subject {} for Freq {}'.format(model_type, sample, curr_freq))
            plt.axvline(x=15, color='b', linestyle='--')
            ax.figure.savefig("Results/{}/{}/sample_{}/freq_{}.png".format(model_type, exp_name, sample, curr_freq),
                              dpi=300)
            plt.clf()

        all_results_df = pd.DataFrame(freq_results)
        all_results_df.to_csv("Results/{}/{}/sample_{}/all_freq_results.csv".format(model_type, exp_name, sample))


if __name__ == "__main__":
    main()
