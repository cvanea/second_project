import numpy as np
import mne
from mat4py import loadmat
import matplotlib.pyplot as plt


def main():
    matlab_data = loadmat('SensorSpace/FLISj4.mat')

    data = matlab_data['data']

    num_channels = 273

    x_train = np.array(data['X_train'])
    x_train = np.transpose(x_train, (1, 0))
    x_train_small = x_train[:num_channels]

    y_train = np.array(data['Y_train'])
    y_train = np.transpose(y_train, (1, 0))

    x_train_epochs = np.array(np.array_split(x_train_small, 160, axis=1))

    # info = mne.create_info(ch_names=num_channels, sfreq=100)
    # average_across_trials_plot(info, x_train_epochs, y_train, "0", all=False)

    info = mne.create_info(ch_names=num_channels, sfreq=100, ch_types='mag')
    evoked_stats_plot(info, x_train_epochs, y_train, label="0")

    # info = mne.create_info(ch_names=num_channels, sfreq=100, ch_types='mag')
    # epochs_stats_plot(info, x_train_epochs, y_train)

    # info = mne.create_info(ch_names=x_train_small.shape[0], sfreq=100)
    # raw_plot(info, x_train_small)

def average_across_trials_plot(info, x_train, y_train, label, all=False):
    value, time = np.where(y_train == 1)

    events = np.arange(start=0, stop=8000, step=50)
    events = np.expand_dims(events, axis=1)
    events = np.pad(events, ((0, 0), (0, 2)), mode='constant', constant_values=0)

    for i in range(160):
        sample = i * 50
        ind = np.where(time == sample)
        events[i][2] = value[ind]

    epochs = mne.EpochsArray(x_train, info=info, events=events)
    if not all:
        evoked_data = epochs[label].average(picks='all')
    else:
        evoked_data = epochs.average(picks='all')
        label = "all"

    ax = evoked_data.plot(window_title='Average Across Trials for Label {}'.format(label), show=False)
    plt.axvline(x=0.15, color='b', linestyle='--')
    plt.show()

def evoked_stats_plot(info, x_train, y_train, label):
    value, time = np.where(y_train == 1)

    events = np.arange(start=0, stop=8000, step=50)
    events = np.expand_dims(events, axis=1)
    events = np.pad(events, ((0, 0), (0, 2)), mode='constant', constant_values=0)

    for i in range(160):
        sample = i * 50
        ind = np.where(time == sample)
        events[i][2] = value[ind]

    epochs = mne.EpochsArray(x_train, info=info, events=events)
    evoked_data = epochs[label].average(picks='all')

    evoked_data.plot_image(picks='data', show=False)
    plt.axvline(x=0.15, color='b', linestyle='--')
    plt.show()

def epochs_stats_plot(info, x_train, y_train):
    value, time = np.where(y_train == 1)

    events = np.arange(start=0, stop=8000, step=50)
    events = np.expand_dims(events, axis=1)
    events = np.pad(events, ((0, 0), (0, 2)), mode='constant', constant_values=0)

    for i in range(160):
        sample = i * 50
        ind = np.where(time==sample)
        events[i][2] = value[ind]

    epochs = mne.EpochsArray(x_train, info=info, events=events)

    fig = epochs.plot_image(picks='data', show=False, evoked=True, combine='mean')
    fig[0].axes[1].axvline(x=0.15, color='b', linestyle='--')
    plt.show()

def events_plot(info, x_train, y_train):
    value, time = np.where(y_train == 1)

    events = np.arange(start=0, stop=8000, step=50)
    events = np.expand_dims(events, axis=1)
    events = np.pad(events, ((0, 0), (0, 2)), mode='constant', constant_values=0)

    for i in range(160):
        sample = i * 50
        ind = np.where(time==sample)
        events[i][2] = value[ind]

    sorted_inds = np.argsort(events[:, 2])
    events = np.take(events, sorted_inds, axis=0)

    epochs = mne.EpochsArray(x_train, info=info, events=events)
    epochs.plot(picks='all', scalings='auto', show=True, block=True)

def raw_plot(info, data):
    raw = mne.io.RawArray(data, info)

    raw.plot(n_channels=data.shape[0], scalings='auto', title='Data from arrays', show=True, block=True,
             color='darkblue')


if __name__ == "__main__":
    main()
