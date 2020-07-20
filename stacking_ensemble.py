import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier

from utils import get_y_train


def main():
    base_model_type = "lda"
    exp_name = "wavelet_class/lsqr/complex"
    meta_model_type = 'log_reg'

    load_dir = "Results/{}/{}".format(base_model_type, exp_name)

    freqs = np.logspace(*np.log10([2, 25]), num=15)
    string_freqs = [str(round(x, 2)) for x in freqs]

    all_sample_models = np.array(pickle.load(open(load_dir + "/all_models.pkl", "rb")))

    all_y_train = []

    for sample in range(1, 22):
        all_y_train.append(get_y_train(sample))

    all_x_train = pickle.load(open("DataTransformed/wavelet_complex/x_train_all_samples.pkl", "rb"))

    for time in range(50):
        print("time {}".format(time))

        sample_models = all_sample_models[:, :, time]

        sample_y_train = []
        sample_predictions_proba = []

        for sample in range(21):
            intervals = np.arange(start=time, stop=all_y_train[sample].shape[0], step=50)
            sample_y_train.append(all_y_train[sample][intervals])

            freq_proba = []
            for freq in range(15):
                # all_x_train[sample][freq] = all_x_train[sample][freq][intervals, :]

                base_x_train = all_x_train[sample][freq][intervals, :]

                prediction_proba = sample_models[sample][freq].predict_proba(base_x_train)
                freq_proba.append(prediction_proba)

            sample_predictions_proba.append(freq_proba)

        sample_predictions_proba = [np.vstack(sample).astype(np.float) for sample in sample_predictions_proba]

        sample_y_train = np.array(sample_y_train)
        sample_y_train = np.repeat(sample_y_train[:, np.newaxis], 15, axis=1)
        sample_y_train = [np.vstack(sample).astype(np.int) for sample in sample_y_train]
        final = []
        for sample in range(21):
            final.append([data for freq_data in sample_y_train[sample] for data in freq_data])
        sample_y_train = final

        x_train = sample_predictions_proba[:-1]
        x_train = [data for freq_data in x_train for data in freq_data]
        print("length of x_train: {}".format(len(x_train)))
        y_train = sample_y_train[:-1]
        y_train = [data for freq_data in y_train for data in freq_data]
        # print("length of y_train: {}".format(len(y_train)))

        x_val = sample_predictions_proba[-1:]
        x_val = [data for freq_data in x_val for data in freq_data]
        print("length of x_val: {}".format(len(x_val)))
        y_val = sample_y_train[-1:]
        y_val = [data for freq_data in y_val for data in freq_data]
        # print("length of y_val: {}".format(len(y_val)))

        meta_model = LogisticRegression()
        meta_model.fit(x_train, y_train)

        print(meta_model.score(x_val, y_val))


if __name__ == "__main__":
    main()