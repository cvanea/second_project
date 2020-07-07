import os

import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mne.time_frequency import tfr_array_morlet

from utils import get_epochs, get_y_train, get_raw_data
from dim_reduction import kernel_pca, ica, pca
from models import linear_models, nonlinear_models

def main():
    model_type = "lda"
    exp_name = "wavelet_class/lsqr"

    for i, sample in enumerate(range(1, 22)):
        print("sample {}".format(sample))

        if not os.path.isdir("Results/{}/{}/sample_{}".format(model_type, exp_name, sample)):
            os.mkdir("Results/{}/{}/sample_{}".format(model_type, exp_name, sample))

        epochs = get_epochs(sample)

        freqs = np.logspace(*np.log10([2, 42]), num=20)
        n_cycles = freqs / 4.

        print("applying morlet wavelet")
        # returns (n_epochs, n_channels, n_freqs, n_times)
        complex = tfr_array_morlet(epochs.get_data(), sfreq=epochs.info['sfreq'], freqs=freqs, n_cycles=n_cycles,
                                 output='complex')

        y_train = get_y_train(sample)

        freq_results = np.zeros((complex.shape[2], 50))

        for freq in range(complex.shape[2]):
            print("frequency: {}".format(freqs[freq]))

            complex_epochs = complex[:, :, freq, :]
            real_epochs = complex_epochs.real

            real_epochs = mne.EpochsArray(real_epochs, info=epochs.info, events=epochs.events)

            real_reduced = pca(80, real_epochs, plot=False)
            x_train = real_reduced.transpose(0, 2, 1).reshape(-1, real_reduced.shape[1])

            results = linear_models(x_train, y_train, model_type=model_type)
            freq_results[freq] = results

            curr_freq = str(round(freqs[freq], 2))

            sns.set()
            ax = sns.lineplot(data=results, dashes=False)
            ax.set(ylim=(0, 0.7), xlabel='Time', ylabel='Accuracy',
                   title='Cross Val Accuracy {} for sample {} for freq {}'.format(model_type, sample, curr_freq))
            plt.axvline(x=15, color='b', linestyle='--')
            ax.figure.savefig("Results/{}/{}/sample_{}/freq_{}.png".format(model_type, exp_name, sample, curr_freq),
                              dpi=300)
            plt.clf()

        all_results_df = pd.DataFrame(freq_results)
        all_results_df.to_csv("Results/{}/{}/sample_{}/all_freq_results.csv".format(model_type, exp_name, sample))


if __name__ == "__main__":
    main()