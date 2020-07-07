import numpy as np
from mat4py import loadmat
import mne
import seaborn as sns
import matplotlib.pyplot as plt
from mne.baseline import rescale
from mne.stats import bootstrap_confidence_interval


def main():
    matlab_data = loadmat('SensorSpace/FLISj1.mat')

    data = matlab_data['data']

    x_train = np.array(data['X_train'])
    x_train = np.transpose(x_train, (1, 0))
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

    info = mne.create_info(ch_names=x_train.shape[0], sfreq=100, ch_types='mag')
    raw = mne.io.RawArray(x_train, info)

    iter_freqs = [('Theta', 4, 7), ('Alpha', 8, 12), ('Beta', 13, 25), ('Gamma', 30, 45)]

    frequency_map = list()

    for band, fmin, fmax in iter_freqs:
        raw.load_data()
        raw.filter(fmin, fmax, n_jobs=1, l_trans_bandwidth=1, h_trans_bandwidth=1, picks='all')

        epochs = mne.Epochs(raw, events, baseline=None, preload=True, picks='all', tmin=0, tmax=0.49)

        epochs.subtract_evoked()

        epochs.apply_hilbert(envelope=True, picks='all')
        frequency_map.append(((band, fmin, fmax), epochs.average(picks='all')))
        del epochs
    del raw

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(10, 7))
    colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
    for ((freq_name, fmin, fmax), average), color, ax in zip(
            frequency_map, colors, axes.ravel()[::-1]):
        times = average.times * 1e3
        gfp = np.sum(average.data ** 2, axis=0)
        gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
        ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
        ax.axhline(0, linestyle='--', color='grey', linewidth=2)
        ci_low, ci_up = bootstrap_confidence_interval(average.data, random_state=0,
                                                      stat_fun=stat_fun)
        ci_low = rescale(ci_low, average.times, baseline=(None, 0))
        ci_up = rescale(ci_up, average.times, baseline=(None, 0))
        ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=color, alpha=0.3)
        ax.grid(True)
        ax.set_ylabel('GFP')
        ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                    xy=(0.95, 0.8),
                    horizontalalignment='right',
                    xycoords='axes fraction')
        ax.set_xlim(0, 500)

    axes.ravel()[-1].set_xlabel('Time [ms]')

    plt.show()

def stat_fun(x):
    """Return sum of squares."""
    return np.sum(x ** 2, axis=0)


if __name__ == "__main__":
    main()