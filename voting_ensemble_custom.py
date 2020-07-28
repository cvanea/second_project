import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score

from utils import get_y_train


def main():
    mode = "hard"

    base_model_type = "lda"
    base_model_dir = "wavelet_class/lsqr/complex"
    save_dir = "Results/ensembles/voting_ensemble/{}".format(mode)

    load_dir = "Results/{}/{}".format(base_model_type, base_model_dir)

    if mode == "soft":
        all_sample_preds = np.array(pickle.load(open(load_dir + "/all_proba_preds.pkl", "rb")))
    else:
        all_sample_preds = np.array(pickle.load(open(load_dir + "/all_preds.pkl", "rb")))

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
            if mode == "soft":
                summed_preds = np.sum(time_preds, axis=0)
                class_preds = np.argmax(summed_preds, axis=1)
            else:
                time_preds = [np.vstack(time_preds).astype(np.int)][0]
                class_preds = [np.bincount(sample).argmax() for sample in time_preds.T]

            acc_score = accuracy_score(y_train, class_preds)

            print("Time {} accuracy: %0.2f".format(time) % (acc_score))

            results[time] = acc_score

        all_sample_results[sample] = results

        sns.set()
        ax = sns.lineplot(data=results, dashes=False)
        ax.set(ylim=(0, 1), xlabel='Timepoints', ylabel='Accuracy',
               title='Cross Val Accuracy Voting Ensemble for Subject {}'.format(sample + 1))
        plt.axvline(x=15, color='b', linestyle='--')
        plt.axhline(0.125, color='k', linestyle='--')
        ax.figure.savefig("{}/LOOCV_sample_{}.png".format(save_dir, sample + 1), dpi=300)
        plt.clf()

    sns.set()
    ax = sns.lineplot(data=np.mean(all_sample_results, axis=0), dashes=False)
    ax.set(ylim=(0, 1), xlabel='Timepoints', ylabel='Accuracy',
           title='Average Cross Val Accuracy Voting Ensemble for All Subjects'.format(base_model_type))
    plt.axvline(x=15, color='b', linestyle='--')
    plt.axhline(0.125, color='k', linestyle='--')
    ax.figure.savefig("{}/LOOCV_all_samples.png".format(save_dir), dpi=300)
    plt.clf()

    results_df = pd.DataFrame(np.mean(all_sample_results, axis=0))
    results_df.to_csv("{}/LOOCV_all_samples.csv".format(save_dir))


if __name__ == "__main__":
    main()
