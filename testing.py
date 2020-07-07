from mat4py import loadmat
import numpy as np

def main():
    sample = 8

    matlab_data = loadmat('SensorSpace/FLISj{}.mat'.format(sample))
    data = matlab_data['data']
    x_train = np.array(data['X_train'])
    y_train = np.array(data['Y_train'])

    y_train_samples = [np.where(x == 1)[0][0] for x in y_train]
    y_train_samples = np.array(y_train_samples)



if __name__ == "__main__":
    main()