import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

def linear_models(x_train, y_train, model_type='log_reg'):
    results = np.zeros(50)

    for time in range(50):
        intervals = np.arange(start=time, stop=x_train.shape[0], step=50)

        x_sample = x_train[intervals, :]
        y_sample = y_train[intervals]

        if model_type == 'log_reg':
            model = LogisticRegression(solver='liblinear', max_iter=3000)
        elif model_type == 'linear_svm':
            model = LinearSVC(dual=False, max_iter=3000)
        elif model_type == "lda":
            model = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
        else:
            raise ValueError("model {} isn't implemented".format(model_type))

        scores = cross_val_score(model, x_sample, y_sample, cv=5)

        print("Time {} accuracy: %0.2f (+/- %0.2f)".format(time) % (scores.mean(), scores.std() * 2))

        results[time] = scores.mean()

    return results

def nonlinear_models(x_train, y_train, model_type='svm', kernel='poly'):
    results = np.zeros(50)

    for time in range(50):
        intervals = np.arange(start=time, stop=x_train.shape[0], step=50)

        x_sample = x_train[intervals, :]
        y_sample = y_train[intervals]

        if model_type == 'svm':
            model = SVC(kernel=kernel)
        elif model_type == 'qda':
            model = QuadraticDiscriminantAnalysis()
        else:
            raise ValueError("model {} isn't implemented".format(model_type))

        scores = cross_val_score(model, x_sample, y_sample, cv=5)

        print("Time {} accuracy: %0.2f (+/- %0.2f)".format(time) % (scores.mean(), scores.std() * 2))

        results[time] = scores.mean()

    return results