from mat4py import loadmat
import numpy as np
import mne


def get_epochs(sample, num_channels=None):
    matlab_data = loadmat('SensorSpace/FLISj{}.mat'.format(sample))

    data = matlab_data['data']

    x_train = np.array(data['X_train'])
    x_train = np.transpose(x_train, (1, 0))
    if num_channels:
        x_train = x_train[:num_channels]

    y_train = np.array(data['Y_train'])
    y_train = np.transpose(y_train, (1, 0))

    x_train_epochs = np.array(np.array_split(x_train, int(x_train.shape[1] / 50), axis=1))

    print("Trials: {}".format(x_train_epochs.shape[0]))
    print("Channels: {}".format(x_train_epochs.shape[1]))

    info = mne.create_info(ch_names=x_train.shape[0], sfreq=100, ch_types='mag')

    value, time = np.where(y_train == 1)

    events = np.arange(start=0, stop=x_train.shape[1], step=50)
    events = np.expand_dims(events, axis=1)
    events = np.pad(events, ((0, 0), (0, 2)), mode='constant', constant_values=0)

    for i in range(int(x_train.shape[1] / 50)):
        label = i * 50
        ind = np.where(time == label)
        events[i][2] = value[ind]

    epochs = mne.EpochsArray(x_train_epochs, info=info, events=events)

    return epochs


def get_y_train(sample):
    matlab_data = loadmat('SensorSpace/FLISj{}.mat'.format(sample))
    data = matlab_data['data']
    y_train = np.array(data['Y_train'])

    y_train_samples = [np.where(x == 1)[0][0] for x in y_train]
    return np.array(y_train_samples)


def get_y_train_sorted(sample):
    matlab_data = loadmat('SensorSpace/FLISj{}.mat'.format(sample))
    data = matlab_data['data']
    y_train = np.array(data['Y_train'])

    y_train_samples = [np.where(x == 1)[0][0] for x in y_train]
    y_train_samples = np.array(y_train_samples)
    return np.sort(y_train_samples)


def get_raw_data(sample):
    matlab_data = loadmat('SensorSpace/FLISj{}.mat'.format(sample))
    data = matlab_data['data']
    x_train = np.array(data['X_train'])
    y_train = np.array(data['Y_train'])

    y_train_samples = [np.where(x == 1)[0][0] for x in y_train]
    y_train_samples = np.array(y_train_samples)

    return x_train, y_train_samples
