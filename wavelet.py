from scipy import signal
import numpy as np


def morlet(x, freqs, sample_rate, w=3, s=1):
    M = 3 * s * w * sample_rate / freqs
    cwt = np.zeros((len(M), x.shape[0]), dtype=complex)
    for ii in range(len(M)):
        mlt = signal.morlet(M[ii], w=w, s=s)
        a = signal.convolve(x, mlt.real, mode='same', method='fft')
        b = signal.convolve(x, mlt.imag, mode='same', method='fft')
        cwt[ii, :] = a + 1j * b
    return cwt


def morlet2(x, freqs, sample_rate, window_len=4, ncycles=5, ret_basis=False, ret_mode='power'):
    cwt = np.zeros((len(freqs), *x.shape[:]), dtype=complex)

    # Get morlet basis
    mlt = get_morlet_basis(freqs, ncycles, window_len, sample_rate)

    for ii in range(len(freqs)):
        a = signal.convolve(x, mlt[ii, :].real, mode='same', method='fft')
        b = signal.convolve(x, mlt[ii, :].imag, mode='same', method='fft')
        cwt[ii, ...] = a + 1j * b

    if ret_mode == 'power':
        cwt = np.power(np.abs(cwt), 2)
    elif ret_mode == 'amplitude':
        cwt = np.abs(cwt)
    elif ret_mode != 'complex':
        raise ValueError('\'ret_mode not recognised, please use one of \{\'power\',\'amplitude\',\'complex\'}')

    if ret_basis:
        return cwt, mlt
    else:
        return cwt


def get_morlet_basis(freq, ncycles, window_len, sample_rate):
    """
    Parameters
    ----------
    freq : array_like
        Array of frequency values in Hz
    ncycles : int
        Width of wavelets in number of cycles
    window_len : scalar
        Length of wavelet window
    sample_rate : scalar
        Sampling frequency of data in Hz
    Returns
    -------
    2D array
        Complex valued array containing morlet wavelets [nfreqs x window_len]
    """
    # This uses broadcasting to avoid loops at 2 points.

    time_vect = np.linspace(-window_len / 2, window_len / 2, sample_rate * window_len)

    # 1) Use broadcasting to make all waves at once, avoid a for f in freq loop
    ft = freq[:, None] * time_vect
    wave = np.exp(2 * np.pi * 1j * ft)  # nfreqs x window_len

    # Sigma controls the width of the gaussians applied to each wavelet. This
    # is adaptive for each frequency to match ncycles
    sigma = ncycles / (2 * np.pi * freq)

    # 2) Use broadcasting to make all windows at once, avoid a for f in freq loop
    gauss = np.exp((-time_vect ** 2)[:, None] / (2 * sigma ** 2)).T

    return wave * gauss
