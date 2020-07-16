import numpy as np
import mne
from mat4py import loadmat
import matplotlib.pyplot as plt
from mne.time_frequency import psd_welch, tfr_morlet


def main():
    matlab_data = loadmat('SensorSpace/FLISj4.mat')

    data = matlab_data['data']

    num_channels = 273

    x_train = np.array(data['X_train'])
    x_train = np.transpose(x_train, (1, 0))
    x_train_small = x_train[:num_channels]
    x_train_epochs = np.array(np.array_split(x_train_small, 160, axis=1))

    y_train = np.array(data['Y_train'])
    y_train = np.transpose(y_train, (1, 0))
    value, time = np.where(y_train == 1)

    events = np.arange(start=0, stop=8000, step=50)
    events = np.expand_dims(events, axis=1)
    events = np.pad(events, ((0, 0), (0, 2)), mode='constant', constant_values=0)

    for i in range(160):
        sample = i * 50
        ind = np.where(time == sample)
        events[i][2] = value[ind]

    info = mne.create_info(ch_names=num_channels, sfreq=100, ch_types='mag')
    epochs = mne.EpochsArray(x_train_epochs, info=info, events=events)

    mne_wavelet_plot(epochs)

def mne_wavelet_plot(epochs):
    freqs = np.logspace(*np.log10([2, 42]), num=15)

    n_cycles = freqs / 4.  # different number of cycle per frequency
    power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            return_itc=True, decim=3, n_jobs=1)

    channel_picks = 0

    # epochs.plot_image(picks=[channel_picks], evoked=True, combine='mean')

    power.plot(picks=[channel_picks], baseline=(0, 0.05), mode='logratio', title=power.ch_names[channel_picks])
    # itc.plot(picks=[272], baseline=(0, 0.05), mode='logratio', title=power.ch_names[272])

def plot_psd_welch(epochs):
    kwargs = dict(fmin=2, fmax=50, n_jobs=1, n_per_seg=10)

    psds_welch_mean, freqs_mean = psd_welch(epochs, average='mean', **kwargs)
    psds_welch_median, freqs_median = psd_welch(epochs, average='median', **kwargs)

    psds_welch_mean = 10 * np.log10(psds_welch_mean)
    psds_welch_median = 10 * np.log10(psds_welch_median)

    ch_name = '0'
    ch_idx = epochs.info['ch_names'].index(ch_name)
    epo_idx = 0

    _, ax = plt.subplots()
    ax.plot(freqs_mean, psds_welch_mean[epo_idx, ch_idx, :], color='k',
            ls='-', label='mean of segments')
    ax.plot(freqs_median, psds_welch_median[epo_idx, ch_idx, :], color='k',
            ls='--', label='median of segments')

    ax.set(title='Welch PSD ({}, Epoch {})'.format(ch_name, epo_idx),
           xlabel='Frequency (Hz)', ylabel='Power Spectral Density (dB)')
    ax.legend(loc='upper right')
    plt.show()


def plot_psd(epochs):
    epochs.plot_psd(fmin=2, fmax=50, average=True, spatial_colors=False)


if __name__ == "__main__":
    main()
