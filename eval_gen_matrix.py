import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def main():

    results_dir = "Results/lda/temporal_gen_matrix/"

    freqs = np.logspace(*np.log10([2, 25]), num=15)

    for sample in range(1, 22):
        sample_result = pd.read_csv(results_dir + "sample_{}/all_freq_matrix_results.csv".format(sample), index_col=0)
        sample_result = sample_result.to_numpy().reshape(sample_result.shape[0], 50, 50)

        print("making final plot")
        sns.set()
        freq_count = 0
        fig2 = plt.figure(figsize=(14, 8), constrained_layout=True)
        spec2 = gridspec.GridSpec(ncols=5, nrows=3, figure=fig2)
        fig2.suptitle('Subject {} Temporal Generalisation Matrices'.format(sample), fontsize=16)
        for row in range(0, 3):
            for col in range(0, 5):
                print("computing plot {}".format(freq_count))
                ax = fig2.add_subplot(spec2[row, col])
                im = ax.imshow(sample_result[freq_count], interpolation='lanczos', origin='lower', cmap='RdBu_r',
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
        fig2.savefig(results_dir + "sample_{}/all_freq_matrices.png".format(sample), dpi=300)
        plt.clf()


if __name__ == "__main__":
    main()