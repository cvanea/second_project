import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mne
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.preprocessing import StandardScaler
import pandas as pd

from utils import get_epochs, get_y_train


def main():
    sample = 1
    exp_name = "ICA"

    epochs = get_epochs(sample)

    num_components = 80
    label = -1

    # reduced_data = pca(num_components, epochs, label)
    # reduced_data = kernel_pca(num_components, epochs, label=label)
    # reduced_data = ica(num_components, epochs, plot=True)

    # vis_embeddings("se", epochs, sample)

    # save_reduced(reduced_data, exp_name, sample)


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


def save_reduced(reduced_data, exp_name, sample):
    # changes shape from (trials, components, times) to (all_times, components)
    reduced_data_all = reduced_data.transpose(0, 2, 1).reshape(-1, reduced_data.shape[1])
    results_df = pd.DataFrame(reduced_data_all)
    results_df.to_csv("Results/log_reg/{}/reduced_data{}.csv".format(exp_name, sample))
    print('reduced_data saved')


def vis_embeddings(dim_red_method, epochs, sample):
    n_comp = 2

    x_train = epochs.get_data()
    x_train = x_train.transpose(0, 2, 1).reshape(-1, x_train.shape[1])
    x_train = StandardScaler().fit_transform(x_train)
    y_train = get_y_train(sample)

    inds = np.arange(15, 8000, 50)
    x_train = x_train[inds]
    y_train = y_train[inds]

    print('fitting {}'.format(dim_red_method))
    if dim_red_method == 'pca':
        pca = PCA(n_components=n_comp)
        reduced_data = pca.fit_transform(x_train)
    elif dim_red_method == 'ica':
        ica = FastICA(n_components=n_comp)
        reduced_data = ica.fit_transform(x_train)
    elif dim_red_method == 'se':
        se = SpectralEmbedding(n_components=n_comp)
        reduced_data = se.fit_transform(x_train)
    elif dim_red_method == 'tsne':
        pca = PCA(n_components=50)
        pca_data = pca.fit_transform(x_train)
        tsne = TSNE(n_components=n_comp, verbose=1, perplexity=10, learning_rate=200)
        reduced_data = tsne.fit_transform(pca_data)
    else:
        raise ValueError("{} method not implemented".format(dim_red_method))
    print('fitting done')

    if n_comp == 2:
        reduced_data_df = pd.DataFrame(data=reduced_data, columns=['PC1', 'PC2'])
    elif n_comp == 3:
        reduced_data_df = pd.DataFrame(data=reduced_data, columns=['PC1', 'PC2', 'PC3'])
    y_train_df = pd.DataFrame(data=y_train, columns=["labels"])
    final_df = pd.concat([reduced_data_df, y_train_df[['labels']]], axis=1)

    if n_comp == 2:
        sns.set()
        palette = sns.color_palette("bright", 8)
        ax = sns.scatterplot(x='PC1', y='PC2', hue='labels', data=final_df, palette=palette, legend='full')
        ax.set(xlabel='PC1', ylabel='PC2', title='2 component {}'.format(dim_red_method))
        plt.show()
    elif n_comp == 3:
        ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
        ax.scatter(xs=final_df["PC1"], ys=final_df["PC2"], zs=final_df["PC2"], c=final_df["labels"], cmap='tab10')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.show()


if __name__ == "__main__":
    main()
