import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def main():
    freqs = np.logspace(*np.log10([2, 15]), num=15)
    plot_avg_all_subjects(15, freqs)
    plot_for_each_subject(15, freqs)


def plot_for_each_subject(hz, freqs):
    for sample in range(1, 22):
        print("sample {}".format(sample))
        results_dir = "Results/lda/wavelet_class/lsqr/complex/{}hz/sample_{}/".format(hz, sample)
        sample_result = pd.read_csv(results_dir + "all_freq_results.csv", index_col=0)

        average_result = sample_result.T

        string_freqs = [str(round(x, 2)) for x in freqs]

        sns.set()
        sns.set_palette(sns.color_palette("cubehelix", 20))
        ax = average_result.plot()
        ax.set(ylim=(0, 1), xlabel='Timepoints', ylabel='Accuracy', title='Cross Val Accuracy for Subject {}'
               .format(sample))
        plt.legend(loc="upper center", ncol=5, fontsize='x-small', labels=string_freqs)
        plt.axvline(x=15, color='b', linestyle='--')
        plt.axhline(0.125, color='k', linestyle='--')
        ax.figure.savefig(results_dir + "all_freqs.png", dpi=300)
        plt.clf()


def plot_avg_all_subjects(hz, freqs):
    results_dir = "Results/lda/wavelet_class/lsqr/complex/{}hz/".format(hz)

    for sample in range(1, 22):
        sample_result = pd.read_csv(results_dir + "sample_{}/all_freq_results.csv".format(sample), index_col=0)

        if sample == 1:
            all_sample_results = sample_result
        else:
            all_sample_results = all_sample_results.append(sample_result)

    average_results = all_sample_results.groupby(level=0).mean().T

    string_freqs = [str(round(x, 2)) for x in freqs]

    sns.set()
    sns.set_palette(sns.color_palette("cubehelix", 20))
    ax = average_results.plot()
    ax.set(ylim=(0, 0.6), xlabel='Timepoints', ylabel='Accuracy', title='Average Cross Val Accuracy for All Subjects')
    plt.legend(loc="lower center", ncol=5, fontsize='x-small', labels=string_freqs)
    plt.axvline(x=15, color='b', linestyle='--')
    plt.axhline(0.125, color='k', linestyle='--')
    ax.figure.savefig(results_dir + "avg_all_samples_zoom.png", dpi=300)
    plt.clf()


if __name__ == "__main__":
    main()
