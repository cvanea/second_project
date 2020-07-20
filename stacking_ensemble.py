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

        for sample in range(21):
            intervals = np.arange(start=time, stop=all_y_train[sample].shape[0], step=50)
            sample_y_train.append(all_y_train[sample][intervals])

            for freq in range(15):
                all_x_train[sample][freq] = all_x_train[sample][freq][intervals, :]

            sample_x_train.append(all_x_train[sample])

        sample_x_train = np.array(sample_x_train)
        sample_y_train = np.array(sample_y_train)

        val_base_models = [model for freq_models in sample_models[-2:] for model in freq_models]
        val_x_train = [data for freq_data in sample_x_train[-2:] for data in freq_data]
        val_y_train = np.repeat(sample_y_train[-2:][:, np.newaxis], 15, axis=1)
        val_y_train = [data for freq_data in val_y_train for data in freq_data]

        base_models = [model for freq_models in sample_models[:-2] for model in freq_models]
        x_train = [data for freq_data in sample_x_train[:-2] for data in freq_data]
        y_train = np.repeat(sample_y_train[:-2][:, np.newaxis], 15, axis=1)
        y_train = [data for freq_data in y_train for data in freq_data]

        meta_model = LogisticRegression()

        sclf = StackingClassifier(classifiers=base_models, meta_classifier=meta_model, verbose=1,
                                  fit_base_estimators=False)
        sclf.fit(x_train, y_train)

        print(sclf.score(val_x_train, val_y_train))

        print("done")

if __name__ == "__main__":
    main()