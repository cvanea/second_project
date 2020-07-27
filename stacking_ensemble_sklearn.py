import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from utils import get_y_train


def main():
    save_dir = "Results/ensembles/stacking_ensemble/per_sample"

    all_x_train = pickle.load(open("DataTransformed/wavelet_complex/x_train_all_samples.pkl", "rb"))

    all_results = np.zeros((21, 50))

    all_freq_pipelines = []
    for freqency_index in range(15):
        pipe = make_pipeline(WaveletTransform(freqency_index),
                             LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        pipe = ("pipe_{}".format(freqency_index), pipe)
        all_freq_pipelines.append(pipe)

    for sample in range(21):
        print("sample {}".format(sample))

        all_y_train = get_y_train(sample + 1)
        sample_x_train = np.array(all_x_train[sample])

        time_results = np.zeros(50)

        for time in range(50):
            intervals = np.arange(start=time, stop=all_y_train.shape[0], step=50)
            y_train = all_y_train[intervals]
            x_train = sample_x_train[:, intervals]
            x_train = x_train.transpose(1, 0, 2).reshape(x_train.shape[1], -1)

            model = StackingClassifier(estimators=all_freq_pipelines, final_estimator=LogisticRegression(),
                                       cv=5, stack_method='predict_proba')
            scores = cross_val_score(model, x_train, y_train, cv=5)
            print("Time {} accuracy: %0.2f (+/- %0.2f)".format(time) % (scores.mean(), scores.std() * 2))
            time_results[time] = scores.mean()

        sns.set()
        ax = sns.lineplot(data=time_results, dashes=False)
        ax.set(ylim=(0, 1), xlabel='Timepoints', ylabel='Accuracy',
               title='Cross Val Accuracy Stacking Ensemble for Sample {}'.format(sample + 1))
        plt.axvline(x=15, color='b', linestyle='--')
        plt.axhline(0.125, color='k', linestyle='--')
        ax.figure.savefig("{}/LOOCV_sample_{}.png".format(save_dir, sample + 1), dpi=300)
        plt.clf()

        all_results[sample] = time_results

    sns.set()
    ax = sns.lineplot(data=np.mean(all_results, axis=0), dashes=False)
    ax.set(ylim=(0, 1), xlabel='Timepoints', ylabel='Accuracy',
           title='Average Cross Val Accuracy Stacking Ensemble for All Samples')
    plt.axvline(x=15, color='b', linestyle='--')
    plt.axhline(0.125, color='k', linestyle='--')
    ax.figure.savefig("{}/LOOCV_all_samples.png".format(save_dir), dpi=300)
    plt.clf()

    results_df = pd.DataFrame(np.mean(all_results, axis=0))
    results_df.to_csv("{}/LOOCV_all_samples.csv".format(save_dir))


class WaveletTransform(BaseEstimator, TransformerMixin):
    def __init__(self, frequency_index):
        self.frequency_index = frequency_index

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        x_train = X.copy()
        x_train = x_train[:, np.arange(start=self.frequency_index * 80, stop=(self.frequency_index * 80) + 80, step=1)]
        return x_train


if __name__ == "__main__":
    main()
