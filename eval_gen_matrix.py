import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def main():
    matrix_type = "freq"

    results_dir = "Results/lda/{}_gen_matrix/".format(matrix_type)

    # gen_matrix_per_sample(matrix_type, results_dir)
    gen_average_matrix(matrix_type, results_dir)


def gen_average_matrix(matrix_type, results_dir):

    for sample in range(1, 22):
        if matrix_type == "temporal":
            sample_result = pd.read_csv(results_dir + "sample_{}/all_freq_matrix_results.csv".format(sample),
                                        index_col=0)
        elif matrix_type == "freq":
            sample_result = pd.read_csv(results_dir + "sample_{}/all_time_matrix_results.csv".format(sample),
                                        index_col=0)
        else:
            raise ValueError("{} is not a matrix type".format(matrix_type))

        if sample == 1:
            all_results = sample_result
        else:
            all_results = all_results.append(sample_result)

    average_results = all_results.groupby(level=0).mean()

    if matrix_type == "temporal":
        average_results = average_results.to_numpy().reshape(sample_result.shape[0], 50, 50)
    else:
        average_results = average_results.to_numpy().reshape(sample_result.shape[0], 15, 15)

    if matrix_type == "temporal":
        make_temporal_plot(average_results, results_dir, sample=None)
    else:
        make_freq_plot(average_results, results_dir, sample=None)


def gen_matrix_per_sample(matrix_type, results_dir):

    for sample in range(1, 22):
        print("sample {}".format(sample))

        if matrix_type == "temporal":
            temporal_matrix(sample, results_dir)
        elif matrix_type == "freq":
            frequency_matrix(sample, results_dir)
        else:
            raise ValueError("{} is not a matrix type".format(matrix_type))


def frequency_matrix(sample, results_dir):
    sample_result = pd.read_csv(results_dir + "sample_{}/all_time_matrix_results.csv".format(sample), index_col=0)
    sample_result = sample_result.to_numpy().reshape(sample_result.shape[0], 15, 15)

    make_freq_plot(sample_result, results_dir, sample=sample)


def temporal_matrix(sample, results_dir):
    sample_result = pd.read_csv(results_dir + "sample_{}/all_freq_matrix_results.csv".format(sample), index_col=0)
    sample_result = sample_result.to_numpy().reshape(sample_result.shape[0], 50, 50)

    make_temporal_plot(sample_result, results_dir, sample=sample)


def make_temporal_plot(data, results_dir, sample=None):
    freqs = np.logspace(*np.log10([2, 25]), num=15)

    print("making final plot")
    sns.set()
    freq_count = 0
    fig2 = plt.figure(figsize=(14, 8), constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=5, nrows=3, figure=fig2)
    if sample:
        fig2.suptitle('Subject {} Temporal Generalisation Matrices'.format(sample), fontsize=16)
    else:
        fig2.suptitle('Average Across Subject Temporal Generalisation Matrices', fontsize=16)
    for row in range(0, 3):
        for col in range(0, 5):
            print("computing plot {}".format(freq_count))
            ax = fig2.add_subplot(spec2[row, col])
            im = ax.imshow(data[freq_count], interpolation='lanczos', origin='lower', cmap='RdBu_r',
                           extent=[0., 0.49, 0., 0.49], vmin=0., vmax=0.8)
            ax.set_title('Freq {}'.format(str(round(freqs[freq_count], 2))))
            ax.axvline(0.15, color='k', linestyle='--')
            ax.axhline(0.15, color='k', linestyle='--')
            ax.grid(False)

            if col == 4:
                plt.colorbar(im, ax=ax)
            if row == 2:
                ax.set_xlabel('Testing Time (s)')
            if col == 0:
                ax.set_ylabel('Training Time (s)')

            freq_count += 1
    if sample:
        fig2.savefig(results_dir + "sample_{}/all_freq_matrices.png".format(sample), dpi=300)
    else:
        fig2.savefig(results_dir + "all_avg_freq_matrices.png", dpi=300)
    plt.close('all')


def make_freq_plot(data, results_dir, sample=None):
    print("making final plot")
    sns.set()
    time_count = 0
    fig2 = plt.figure(figsize=(20, 12), constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=10, nrows=5, figure=fig2)
    if sample:
        fig2.suptitle('Subject {} Frequency Generalisation Matrices'.format(sample), fontsize=16)
    else:
        fig2.suptitle('Average Across Subject Frequency Generalisation Matrices', fontsize=16)
    for row in range(0, 5):
        for col in range(0, 10):
            print("computing plot {}".format(time_count))
            ax = fig2.add_subplot(spec2[row, col])
            im = ax.imshow(data[time_count], interpolation='lanczos', origin='lower', cmap='RdBu_r',
                           extent=[2, 25, 2, 25], vmin=0., vmax=0.8)
            ax.set_title('Time {}'.format(time_count))
            ax.grid(False)

            if col == 9:
                plt.colorbar(im, ax=ax)
            if row == 4:
                ax.set_xlabel('Testing Frequency (hz)')
            if col == 0:
                ax.set_ylabel('Training Frequency (hz)')

            time_count += 1
    if sample:
        fig2.savefig(results_dir + "sample_{}/all_time_matrices.png".format(sample), dpi=300)
    else:
        fig2.savefig(results_dir + "/all_avg_time_matrices.png", dpi=300)
    plt.close('all')


if __name__ == "__main__":
    main()
