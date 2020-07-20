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

        sample_x_train = []
        sample_y_train = []
        sample_predictions_proba = []

        for sample in range(21):
            intervals = np.arange(start=time, stop=all_y_train[sample].shape[0], step=50)
            sample_y_train.append(all_y_train[sample][intervals])

            y_train = all_y_train[sample][intervals]

            freq_proba = []

            for freq in range(15):
                all_x_train[sample][freq] = all_x_train[sample][freq][intervals, :]

                x_train = all_x_train[sample][freq]

                prediction_proba = sample_models[sample][freq].predict_proba(x_train)
                freq_proba.append(prediction_proba)

            sample_predictions_proba.append(freq_proba)

            # sample_x_train.append(all_x_train[sample])

        sample_predictions_proba = [np.vstack(sample).astype(np.float) for sample in sample_predictions_proba]

        sample_y_train = np.array(sample_y_train)
        sample_y_train = np.repeat(sample_y_train[:, np.newaxis], 15, axis=1)
        sample_y_train = [np.vstack(sample).astype(np.int) for sample in sample_y_train]
        final = []
        for sample in range(21):
            final.append([data for freq_data in sample_y_train[sample] for data in freq_data])
        sample_y_train = final

        


        # # per sample, per freq
        # sample_x_train = np.array(sample_x_train)
        # # per sample, all freq stacked together
        # sample_x_train = [np.vstack(sample).astype(np.float) for sample in sample_x_train]
        #
        #
        # val_base_models = [model for freq_models in sample_models[-2:] for model in freq_models]
        #
        # base_models = [model for freq_models in sample_models[:-2] for model in freq_models]
        #
        # meta_model = LogisticRegression()
        #
        # sclf = StackingClassifier(classifiers=base_models, meta_classifier=meta_model, verbose=1,
        #                           fit_base_estimators=False)
        # sclf.fit(x_train, y_train)
        #
        # # print(sclf.score(val_x_train, val_y_train))
        #
        # print("done")

if __name__ == "__main__":
    main()