import numpy as np
import mne
from mat4py import loadmat
from mne import combine_evoked
from mne.viz import plot_events
from sklearn.linear_model import LogisticRegression


def main():
    matlab_data = loadmat('SensorSpace/FLISj1.mat')

    data = matlab_data['data']

    x_train = np.array(data['X_train'])
    x_train = np.transpose(x_train, (1, 0))

    y_train = np.array(data['Y_train'])
    y_train = np.transpose(y_train, (1, 0))

    value, time = np.where(y_train == 1)

    trials = np.arange(start=0, stop=8000, step=50)
    trials = np.expand_dims(trials, axis=1)
    trials = np.pad(trials, ((0, 0), (0, 2)), mode='constant', constant_values=0)

    for i in range(160):
        sample = i * 50
        ind = np.where(time==sample)
        trials[i][2] = value[ind]

    x_train_epochs = np.array(np.array_split(x_train, 160, axis=1))
    info = mne.create_info(ch_names=len(x_train), sfreq=100)
    epochs = mne.EpochsArray(x_train_epochs, info=info, events=trials)

    

    # model = LogisticRegression()

    # model.fit(trials, y_train)





if __name__ == "__main__":
    main()
