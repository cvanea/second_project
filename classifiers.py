import numpy as np
from mat4py import loadmat
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd

def main():
    matlab_data = loadmat('SensorSpace/FLISj1.mat')

    data = matlab_data['data']

    x_train = np.array(data['X_train'])
    x_train = np.transpose(x_train, (1, 0))

    y_train = np.array(data['Y_train'])
    y_train = np.transpose(y_train, (1, 0))

    results = np.zeros((8, 50))

    for label in range(8):
        for time in range(50):
            intervals = np.arange(start=time, stop=8000, step=50)
            # data samples for classifier at specific time
            x_sample = x_train[:, intervals].transpose((1, 0))
            # label samples for classifier at specific time for specific label
            y_sample = y_train[:, intervals][label]

            # model = LogisticRegression(solver='liblinear')
            model = LinearSVC(dual=False, max_iter=3000)

            scores = cross_val_score(model, x_sample, y_sample, cv=5)

            print(
                "Time {} label {} accuracy: %0.2f (+/- %0.2f)".format(time, label) % (scores.mean(), scores.std() * 2))

            results[label, time] = scores.mean()

    results_df = pd.DataFrame({"0": results[0], "1": results[1], "2": results[2], "3": results[3], "4": results[4],
                               "5": results[5], "6": results[6], "7": results[7]})

    results_df.to_csv("Results/linear_svm/sample1.csv")

if __name__ == "__main__":
    main()
