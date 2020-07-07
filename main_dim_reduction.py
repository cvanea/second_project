import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils import get_epochs, get_y_train
from dim_reduction import kernel_pca, kernel_pca_per_label, ica
from models import linear_models, nonlinear_models

def main():
    sample = 4
    model_type = "svm"
    exp_name = "ICA/sample_{}".format(sample)

    hps = ["linear", "poly", "rbf", "sigmoid"]

    epochs = get_epochs(sample)

    all_results = np.zeros((len(hps), 50))

    for i, hp in enumerate(hps):
        print("iteration with hp set to: {}".format(hp))
        reduced_data = ica(70, epochs, plot=False)
        x_train = reduced_data.transpose(0, 2, 1).reshape(-1, reduced_data.shape[1])

        y_train = get_y_train(sample)

        results = nonlinear_models(x_train, y_train, model_type=model_type, kernel=hp)
        all_results[i] = results

        sns.set()
        ax = sns.lineplot(data=results, dashes=False)
        ax.set(ylim=(0, 0.6), xlabel='Time', ylabel='Accuracy',
               title='Cross Val Accuracy {} and {} comps'.format(model_type, hp))
        plt.axvline(x=15, color='b', linestyle='--')
        ax.figure.savefig("Results/{}/{}/kernel_{}_sample{}".format(model_type, exp_name, hp, sample), dpi=300)
        # plt.show()
        plt.clf()

    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv("Results/{}/{}/all_kernels_sample{}.csv".format(model_type, exp_name, sample))

if __name__ == "__main__":
    main()