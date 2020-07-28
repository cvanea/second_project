import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def main():

    results_dir = "Results/lda/wavelet_class/lsqr/complex/"

    for sample in range(1, 22):
        sample_result = pd.read_csv(results_dir + "sample_{}/all_freq_results.csv".format(sample), index_col=0)

        if sample == 1:
            all_sample_results = sample_result
        else:
            all_sample_results = all_sample_results.append(sample_result)

    average_results = all_sample_results.groupby(level=0).mean().T

    freqs = np.logspace(*np.log10([2, 42]), num=20)

    string_freqs = [str(round(x, 2)) for x in freqs]

    sns.set()
    # sns.set_palette(sns.color_palette(sns.cubehelix_palette(20, reverse=True)))
    sns.set_palette(sns.color_palette("cubehelix", 20))
    ax = average_results.plot()
    ax.set(ylim=(0, 0.6), xlabel='Timepoints', ylabel='Accuracy', title='Average Cross Val Accuracy for All Samples')
    plt.legend(loc="lower center", ncol=5, fontsize='x-small', labels=string_freqs)
    plt.axvline(x=15, color='b', linestyle='--')
    plt.axhline(0.125, color='k', linestyle='--')
    ax.figure.savefig(results_dir + "avg_all_samples_zoom.png", dpi=300)
    plt.clf()


if __name__ == "__main__":
    main()