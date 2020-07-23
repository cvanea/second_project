import pickle
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from utils import get_y_train


def main():
    base_model_type = "lda"
    save_dir = "Results/stacking_ensemble"

    all_x_train = pickle.load(open("DataTransformed/wavelet_complex/x_train_all_samples.pkl", "rb"))

    all_freq_pipelines = []

    for freq in range(15):
        pipe = make_pipeline(WaveletTransform(freq), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        pipe = ("pipe_{}".format(freq), pipe)
        all_freq_pipelines.append(pipe)

    all_results = np.zeros((21, 50))

    for sample in range(21):
        print("sample {}".format(sample))

        time_results = np.zeros(50)

        for time in range(50):
            x_train = all_x_train[sample]
            y_train = get_y_train(sample + 1)

            intervals = np.arange(start=time, stop=y_train.shape[0], step=50)

            for freq in range(15):
                x_train[freq] = x_train[freq][intervals, :]
            y_train = y_train[intervals]

            model = StackingClassifier(all_freq_pipelines, LogisticRegression(), cv=5, stack_method='predict_proba')
            scores = cross_val_score(model, x_train, y_train, cv=5)
            print("Time {} accuracy: %0.2f (+/- %0.2f)".format(time) % (scores.mean(), scores.std() * 2))
            time_results[time] = scores.mean()

        all_results[sample] = time_results


class WaveletTransform(BaseEstimator, TransformerMixin):
    def __init__(self, frequency_index):
        self.frequency_index = frequency_index

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("taking freq {} data".format(self.frequency_index))
        X_ = X.copy()
        X_ = X_[self.frequency_index]
        return X_


if __name__ == "__main__":
    main()
