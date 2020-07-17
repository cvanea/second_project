import mne
import numpy as np
from mne.time_frequency import tfr_array_morlet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle

from utils import get_epochs, get_y_train
from dim_reduction import pca


def main():
    model_type = "lda"
    exp_name = "wavelet_class/lsqr/complex"

    save_dir = "Results/{}/{}".format(model_type, exp_name)

    sample_models = []

    for sample in range(1, 22):
        print("sample {}".format(sample))

        epochs = get_epochs(sample, scale=False)

        freqs = np.logspace(*np.log10([2, 25]), num=15)
        n_cycles = freqs / 4.

        print("applying morlet wavelet")
        # returns (n_epochs, n_channels, n_freqs, n_times)
        wavelet_output = tfr_array_morlet(epochs.get_data(), sfreq=epochs.info['sfreq'], freqs=freqs, n_cycles=n_cycles,
                                   output='complex')
        y_train = get_y_train(sample)

        freq_models = []

        for freq in range(wavelet_output.shape[2]):
            print("frequency: {}".format(freqs[freq]))

            wavelet_epochs = wavelet_output[:, :, freq, :]
            wavelet_epochs = np.append(wavelet_epochs.real, wavelet_epochs.imag, axis=1)

            wavelet_info = mne.create_info(ch_names=wavelet_epochs.shape[1], sfreq=epochs.info['sfreq'], ch_types='mag')
            wavelet_epochs = mne.EpochsArray(wavelet_epochs, info=wavelet_info, events=epochs.events)

            reduced = pca(80, wavelet_epochs, plot=False)
            x_train = reduced.transpose(0, 2, 1).reshape(-1, reduced.shape[1])

            time_models = []

            for time in range(50):
                print("time {}".format(time))
                intervals = np.arange(start=time, stop=x_train.shape[0], step=50)

                x_sample = x_train[intervals, :]
                y_sample = y_train[intervals]
                model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                model.fit(x_sample, y_sample)

                time_models.append(model)

            freq_models.append(time_models)

        sample_models.append(freq_models)
        print('saving models for sample {}'.format(sample))
        pickle.dump(freq_models, open("{}/sample_{}/all_freq_models.pkl".format(save_dir, sample), "wb"))
        print("models saved")

    print('saving models for all samples')
    pickle.dump(sample_models, open("{}/all_models.pkl".format(save_dir), "wb"))
    print("models saved")


if __name__ == "__main__":
    main()