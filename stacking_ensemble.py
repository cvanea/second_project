import mne
import numpy as np
from mne.time_frequency import tfr_array_morlet

from dim_reduction import pca
from utils import get_epochs, get_y_train


def main():
    freqs = np.logspace(*np.log10([2, 25]), num=15)
    n_cycles = freqs / 4.

    all_results = np.zeros(50)

    for sample in range(1, 22):
        print("sample {}".format(sample))
        y_train = get_y_train(sample)
        epochs = get_epochs(sample, scale=False)
        print("applying wavelet")
        wavelet_output = tfr_array_morlet(epochs.get_data(), sfreq=epochs.info['sfreq'], freqs=freqs,
                                          n_cycles=n_cycles, output='complex')

        if sample == 1:
            all_wavelet_ouputs = wavelet_output
            all_y_train = y_train
        else:
            all_wavelet_ouputs = np.append(all_wavelet_ouputs, wavelet_output, axis=0)
            all_y_train = np.append(all_y_train, y_train, axis=0)


    for freq in range(all_wavelet_ouputs.shape[2]):
        print("frequency: {}".format(freqs[freq]))

        all_wavelet_epochs = all_wavelet_ouputs[:, :, freq, :]
        all_wavelet_epochs = np.append(all_wavelet_epochs.real, all_wavelet_epochs.imag, axis=1)

        wavelet_info = mne.create_info(ch_names=all_wavelet_epochs.shape[1], sfreq=epochs.info['sfreq'], ch_types='mag')
        wavelet_epochs = mne.EpochsArray(all_wavelet_epochs, info=wavelet_info, events=epochs.events)

        reduced = pca(80, wavelet_epochs, plot=False)
        x_train = reduced.transpose(0, 2, 1).reshape(-1, reduced.shape[1])





if __name__ == "__main__":
    main()