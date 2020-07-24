import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from utils import get_y_train


def main():
    base_model_type = "lda"
    base_model_dir = "wavelet_class/lsqr/complex"
    save_dir = "Results/stacking_ensemble/per_sample"

    load_dir = "Results/{}/{}".format(base_model_type, base_model_dir)

    all_sample_preds = np.array(pickle.load(open(load_dir + "/all_cv_preds.pkl", "rb")))

    all_sample_results = np.zeros((21, 50))

    for sample in range(21):
        print("sample {}".format(sample))
        sample_y_train = get_y_train(sample + 1)
        freq_preds = all_sample_preds[sample]

        results = np.zeros(50)

        for time in range(50):
            intervals = np.arange(start=time, stop=sample_y_train.shape[0], step=50)
            y_train = sample_y_train[intervals]
            time_preds = freq_preds[:, time]

            y_train = np.tile(y_train, 15)
            x_train = [data for freq_data in time_preds for data in freq_data]
            x_train = np.array(x_train)

            meta_model = LogisticRegression()
            scores = cross_val_score(meta_model, x_train, y_train, cv=5)

            print("Time {} accuracy: %0.2f (+/- %0.2f)".format(time) % (scores.mean(), scores.std() * 2))

            results[time] = scores.mean()

        all_sample_results[sample] = results

        sns.set()
        ax = sns.lineplot(data=results, dashes=False)
        ax.set(ylim=(0, 0.7), xlabel='Timepoints', ylabel='Accuracy',
               title='Cross Val Accuracy Stacking Ensemble {} Base Models for Sample {}'.format(base_model_type, sample+1))
        plt.axvline(x=15, color='b', linestyle='--')
        ax.figure.savefig("{}/LOOCV_sample_{}.png".format(save_dir, sample+1), dpi=300)
        plt.clf()

    sns.set()
    ax = sns.lineplot(data=np.mean(all_sample_results, axis=0), dashes=False)
    ax.set(ylim=(0, 0.6), xlabel='Timepoints', ylabel='Accuracy',
           title='Average Cross Val Accuracy Stacking Ensemble {} Base Models for All Samples'.format(base_model_type))
    plt.axvline(x=15, color='b', linestyle='--')
    ax.figure.savefig("{}/LOOCV_all_samples.png".format(save_dir), dpi=300)
    plt.clf()

    results_df = pd.DataFrame(np.mean(all_sample_results, axis=0))
    results_df.to_csv("{}/LOOCV_all_samples.csv".format(save_dir))


if __name__ == "__main__":
    main()
