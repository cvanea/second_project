import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA, KernelPCA
import pandas as pd

from utils import get_epochs

def main():
    sample = 1
    exp_name = "ICA"

    epochs = get_epochs(sample)

    num_components = 80
    label = -1

    # reduced_data = pca_per_label(num_components, epochs)
    # reduced_data = pca(num_components, epochs, label)
    # reduced_data = kernel_pca(num_components, epochs, label=label)
    reduced_data = ica(num_components, epochs, plot=True)

    # changes shape from (trials, components, times) to (all_times, components)
    reduced_data_all = reduced_data.transpose(0, 2, 1).reshape(-1, reduced_data.shape[1])
    results_df = pd.DataFrame(reduced_data_all)
    results_df.to_csv("Results/log_reg/{}/reduced_data{}.csv".format(exp_name, sample))
    print('reduced_data saved')


def kernel_pca(num_components, epochs, kernel='linear', label=-1, plot=True):
    data = epochs.get_data()

    pca = UnsupervisedSpatialFilter(KernelPCA(num_components, kernel=kernel), average=False)
    print('fitting pca')
    pca_data = pca.fit_transform(data)
    print('fitting done')

    if label != -1:
        all_labels = epochs.events[:, 2]
        inds_to_keep = np.where(all_labels == label)
        pca_subdata = pca_data[inds_to_keep]
        pca_data = pca_subdata

    if plot:
        info = mne.create_info(pca_data.shape[1], epochs.info['sfreq'])

        ev = mne.EvokedArray(np.mean(pca_data, axis=0), info=info)
        if label != -1:
            ev.plot(show=False, window_title="PCA", time_unit='s', titles="Kernel PCA for label {}".format(label))
        else:
            ev.plot(show=False, window_title="PCA", time_unit='s')
        plt.axvline(x=0.15, color='b', linestyle='--')
        plt.show()

    return pca_data

def kernel_pca_per_label(num_components, epochs, kernel='linear', plot=True):
    data = epochs.get_data()

    all_labels = epochs.events[:, 2]

    for label in range(8):
        label_inds = np.where(all_labels == label)
        data_per_label = data[label_inds]

        pca = UnsupervisedSpatialFilter(KernelPCA(num_components, kernel=kernel), average=False)
        print('fitting pca for label {} and kernel {}'.format(label, kernel))
        pca_data = pca.fit_transform(data_per_label)
        print('fitting done')

        if label == 0:
            all_pca = pca_data
        else:
            all_pca = np.concatenate((all_pca, pca_data))

        if plot:
            info = mne.create_info(pca_data.shape[1], epochs.info['sfreq'])

            ev = mne.EvokedArray(np.mean(pca_data, axis=0), info=info)

            ev.plot(show=False, window_title="PCA", time_unit='s',
                    titles="Kernel PCA for label {} and kernel".format(label, kernel))
            plt.axvline(x=0.15, color='b', linestyle='--')
            plt.show()

    return all_pca


def pca(num_components, epochs, label=-1, plot=True):
    data = epochs.get_data()

    pca = UnsupervisedSpatialFilter(PCA(n_components=num_components), average=False)
    print('fitting pca')
    pca_data = pca.fit_transform(data)
    print('fitting done')

    if label != -1:
        all_labels = epochs.events[:, 2]
        inds_to_keep = np.where(all_labels == label)
        pca_subdata = pca_data[inds_to_keep]
        pca_data = pca_subdata

    if plot:
        info = mne.create_info(pca_data.shape[1], epochs.info['sfreq'])
        ev = mne.EvokedArray(np.mean(pca_data, axis=0), info=info)
        if label != -1:
            ev.plot(show=False, window_title="PCA", time_unit='s', titles="PCA for label {}".format(label))
        else:
            ev.plot(show=False, window_title="PCA", time_unit='s')
        plt.axvline(x=0.15, color='b', linestyle='--')
        plt.show()

    return pca_data

def pca_per_label(num_components, epochs, plot=True):
    data = epochs.get_data()

    all_labels = epochs.events[:, 2]

    for label in range(8):
        label_inds = np.where(all_labels == label)
        data_per_label = data[label_inds]

        pca = UnsupervisedSpatialFilter(PCA(n_components=num_components), average=False)
        print('fitting pca for label {}'.format(label))
        pca_data = pca.fit_transform(data_per_label)
        print('fitting done')

        if label == 0:
            all_pca = pca_data
        else:
            all_pca = np.concatenate((all_pca, pca_data))

        if plot:
            info = mne.create_info(pca_data.shape[1], epochs.info['sfreq'])

            ev = mne.EvokedArray(np.mean(pca_data, axis=0), info=info)

            ev.plot(show=False, window_title="PCA", time_unit='s', titles="PCA for label {}".format(label))
            plt.axvline(x=0.15, color='b', linestyle='--')
            plt.show()

    return all_pca

def ica(num_components, epochs, plot=True):
    data = epochs.get_data()

    ica = UnsupervisedSpatialFilter(FastICA(n_components=num_components, max_iter=2000), average=False)
    print('fitting ica')
    ica_data = ica.fit_transform(data)
    print('fitting done')

    info = mne.create_info(ica_data.shape[1], epochs.info['sfreq'])

    if plot:
        ev = mne.EvokedArray(np.mean(ica_data, axis=0), info=info)
        ev.plot(show=False, window_title="ICA", time_unit='s', titles="ICA")
        plt.axvline(x=0.15, color='b', linestyle='--')
        plt.show()

    return ica_data


if __name__ == "__main__":
    main()