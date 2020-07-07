import numpy as np
import wavelet
import matplotlib.pyplot as plt


def main():
    sample_rate = 200
    seconds = 5
    time_vect = np.linspace(0, seconds, sample_rate * seconds)
    x = np.random.randn(*time_vect.shape) * 0.25
    x[sample_rate * 1:sample_rate * 2] += np.sin(2 * np.pi * 10 * time_vect[sample_rate * 1:sample_rate * 2])
    x[sample_rate * 3:sample_rate * 4] += np.sin(2 * np.pi * 22 * time_vect[sample_rate * 3:sample_rate * 4])
    x[sample_rate * 1:sample_rate * 4] += 4 * np.sin(2 * np.pi * 50 * time_vect[sample_rate * 1:sample_rate * 4])
    # Increasing ncycles will increase frequency resolution and decrease time resolution (and vice versa)
    freq_vect = np.linspace(1, 70)
    cwt = wavelet.morlet2(x, freq_vect, sample_rate, ncycles=10, ret_mode='power')

    plt.figure()
    plt.subplot(311)
    plt.plot(time_vect, x, 'k')
    plt.subplot(3, 1, (2, 3))
    plt.pcolormesh(time_vect, freq_vect, cwt)

    plt.show()


if __name__ == "__main__":
    main()
