import numpy as np
from mat4py import loadmat
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd

def main():
    sample = 1
    model_type = "svm"
    exp_name = "raw"

    x_train, y_train = get_raw_data(sample)
    # x_train, y_train = get_pca_data(sample, exp_name)
    # x_train, y_train = get_pca_per_label_data(sample, exp_name)

    results = np.zeros(50)

    for time in range(50):
        intervals = np.arange(start=time, stop=8000, step=50)

        x_sample = x_train[intervals, :]
        y_sample = y_train[intervals]

        # model = LogisticRegression(solver='liblinear', max_iter=3000)
        # model = LinearSVC(dual=False, max_iter=3000)
        kernel = "sigmoid"
        model = SVC(kernel=kernel)

        scores = cross_val_score(model, x_sample, y_sample, cv=5)

        print("Time {} accuracy: %0.2f (+/- %0.2f)".format(time) % (scores.mean(), scores.std() * 2))

        results[time] = scores.mean()

    results_df = pd.DataFrame(results)

    results_df.to_csv("Results/{}/{}/{}_sample{}.csv".format(model_type, exp_name, kernel, sample))


def get_pca_data(sample, exp_name):
    matlab_data = loadmat('SensorSpace/FLISj{}.mat'.format(sample))
    data = matlab_data['data']
    y_train = np.array(data['Y_train'])

    y_train_samples = [np.where(x == 1)[0][0] for x in y_train]
    y_train_samples = np.array(y_train_samples)

    x_train = pd.read_csv("Results/log_reg/{}/reduced_data{}.csv".format(exp_name, sample), index_col=0, header=0)

    return x_train.to_numpy(), y_train_samples

def get_pca_per_label_data(sample, exp_name):
    matlab_data = loadmat('SensorSpace/FLISj{}.mat'.format(sample))
    data = matlab_data['data']
    y_train = np.array(data['Y_train'])

    y_train_samples = [np.where(x == 1)[0][0] for x in y_train]
    y_train_samples = np.array(y_train_samples)
    y_train_samples = np.sort(y_train_samples)

    x_train = pd.read_csv("Results/log_reg/{}/per_label/pca_data{}.csv".format(exp_name, sample), index_col=0, header=0)

    return x_train.to_numpy(), y_train_samples


def get_raw_data(sample):
    matlab_data = loadmat('SensorSpace/FLISj{}.mat'.format(sample))
    data = matlab_data['data']
    x_train = np.array(data['X_train'])
    y_train = np.array(data['Y_train'])

    y_train_samples = [np.where(x == 1)[0][0] for x in y_train]
    y_train_samples = np.array(y_train_samples)

    return x_train, y_train_samples


if __name__ == "__main__":
    main()
