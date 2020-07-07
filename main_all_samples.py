import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils import get_epochs, get_y_train, get_raw_data
from dim_reduction import kernel_pca, ica, pca
from models import linear_models, nonlinear_models

def main():
    model_type = "log_reg"
    exp_name = "PCA/all_samples"

    all_sample_results = np.zeros((21, 50))

    for i, sample in enumerate(range(1, 22)):
        print("sample {}".format(sample))

        if exp_name == "raw/all_samples":
            x_train, y_train = get_raw_data(sample)
        else:
            epochs = get_epochs(sample)
            reduced_data = pca(80, epochs, plot=False)
            x_train = reduced_data.transpose(0, 2, 1).reshape(-1, reduced_data.shape[1])
            y_train = get_y_train(sample)

        results = linear_models(x_train, y_train, model_type=model_type)
        all_sample_results[i] = results

        sns.set()
        ax = sns.lineplot(data=results, dashes=False)
        ax.set(ylim=(0, 0.6), xlabel='Time', ylabel='Accuracy',
               title='Cross Val Accuracy {} for sample {}'.format(model_type, sample))
        plt.axvline(x=15, color='b', linestyle='--')
        ax.figure.savefig("Results/{}/{}/sample{}".format(model_type, exp_name, sample), dpi=300)
        # plt.show()
        plt.clf()

    all_results_df = pd.DataFrame(all_sample_results)
    all_results_df.to_csv("Results/{}/{}/all_sample_results.csv".format(model_type, exp_name))

    average_results = np.mean(all_sample_results, axis=0)
    sns.set()
    ax = sns.lineplot(data=average_results, dashes=False)
    ax.set(ylim=(0, 0.6), xlabel='Time', ylabel='Accuracy',
           title='Average Cross Val Accuracy {} across all samples'.format(model_type))
    plt.axvline(x=15, color='b', linestyle='--')
    ax.figure.savefig("Results/{}/{}/average_all_samples".format(model_type, exp_name), dpi=300)
    # plt.show()
    plt.clf()


if __name__ == "__main__":
    main()