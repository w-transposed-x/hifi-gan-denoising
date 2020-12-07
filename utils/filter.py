# Modified from PyOctaveBand by jmrplens (GPL-3.0 License)
# https://github.com/jmrplens/PyOctaveBand

import librosa
import numpy as np
from scipy import signal


def random_eq(audio, fs, fraction, order, limits_freq, distribtion):
    bands, _, _ = filterbank(audio, fs, fraction, order, limits_freq)
    bands = np.stack(bands, axis=0)
    coeffs = distribtion(size=(bands.shape[0], 1))
    eq_coeffs = librosa.db_to_amplitude(coeffs)
    bands *= eq_coeffs
    return librosa.util.normalize(np.sum(bands, axis=0))


def filterbank(audio, fs, fraction, order, limits):
    sos = []
    freq, freq_d, freq_u = _genfreqs(limits, fraction, fs)
    sos.append(_butter(freq_d[0], 'lowpass', fs, order))
    sos.extend(_butterbandpass(freq_d, freq_u, fs, order))
    sos.append(_butter(freq_u[-1], 'highpass', fs, order))
    bands = []
    for s in sos:
        bands.append(
            signal.sosfiltfilt(s, audio)
        )
    return bands, freq_d, freq_u


def _butter(freq, type, fs, order):
    return signal.butter(N=order,
                         Wn=freq,
                         btype=type,
                         analog=False,
                         output='sos',
                         fs=fs)


def _butterbandpass(freq_d, freq_u, fs, order):
    sos = []
    # Generates coefficients for each frequency band
    for lower, upper in zip(freq_d, freq_u):
        # Butterworth Filter with SOS coefficients
        sos.append(signal.butter(
            N=order,
            Wn=np.array([lower, upper]),
            btype='bandpass',
            analog=False,
            output='sos',
            fs=fs))
    return sos


def _genfreqs(limits, fraction, fs):
    # Generates frequencies
    freq, freq_d, freq_u = getansifrequencies(fraction, limits)

    # Removes outer frequency to prevent filter error (fs/2 < freq)
    freq, freq_d, freq_u = _deleteouters(freq, freq_d, freq_u, fs)

    return freq, freq_d, freq_u


def normalizedfreq(fraction):
    """
    Normalized frequencies for one-octave and third-octave band. [IEC
    61260-1-2014]
    :param fraction: Octave type, for one octave fraction=1,
    for third-octave fraction=3
    :type fraction: int
    :returns: frequencies array
    :rtype: list
    """
    predefined = {1: _oneoctave(),
                  3: _thirdoctave(),
                  }
    return predefined[fraction]


def _thirdoctave():
    # IEC 61260 - 1 - 2014 (added 12.5, 16, 20 Hz)
    return [12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
            315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
            5000, 6300, 8000, 10000, 12500, 16000, 20000]


def _oneoctave():
    # IEC 61260 - 1 - 2014 (added 16 Hz)
    return [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


def _deleteouters(freq, freq_d, freq_u, fs):
    idx = np.asarray(np.where(np.array(freq_u) > fs / 2))
    if any(idx[0]):
        freq = np.delete(freq, idx).tolist()
        freq_d = np.delete(freq_d, idx).tolist()
        freq_u = np.delete(freq_u, idx).tolist()
    return freq, freq_d, freq_u


def getansifrequencies(fraction, limits=None):
    """ ANSI s1.11-2004 && IEC 61260-1-2014
    Array of frequencies and its edges according to the ANSI and IEC standard.
    :param fraction: Bandwidth 'b'. Examples: 1/3-octave b=3, 1-octave b=1,
    2/3-octave b = 3/2
    :param limits: It is a list with the minimum and maximum frequency that
    the array should have. Example: [12,20000]
    :returns: Frequency array, lower edge array and upper edge array
    :rtype: list, list, list
    """

    if limits is None:
        limits = [12, 20000]

    # Octave ratio g (ANSI s1.11, 3.2, pg. 2)
    g = 10 ** (3 / 10)  # Or g = 2
    # Reference frequency (ANSI s1.11, 3.4, pg. 2)
    fr = 1000

    # Gets starting index 'x' and first center frequency
    x = _initindex(limits[0], fr, g, fraction)
    freq = _ratio(g, x, fraction) * fr

    # Gets each frequency until reach maximum frequency
    freq_x = 0
    while freq_x * _bandedge(g, fraction) < limits[1]:
        # Increases index
        x = x + 1
        # New frequency
        freq_x = _ratio(g, x, fraction) * fr
        # Stores new frequency
        freq = np.append(freq, freq_x)

    # Gets band-edges
    freq_d = freq / _bandedge(g, fraction)
    freq_u = freq * _bandedge(g, fraction)

    return freq.tolist(), freq_d.tolist(), freq_u.tolist()


def _initindex(f, fr, g, b):
    if b % 2:  # ODD ('x' solve from ANSI s1.11, eq. 3)
        return np.round(
            (b * np.log(f / fr) + 30 * np.log(g)) / np.log(g)
        )
    else:  # EVEN ('x' solve from ANSI s1.11, eq. 4)
        return np.round(
            (2 * b * np.log(f / fr) + 59 * np.log(g)) / (2 * np.log(g))
        )


def _ratio(g, x, b):
    if b % 2:  # ODD (ANSI s1.11, eq. 3)
        return g ** ((x - 30) / b)
    else:  # EVEN (ANSI s1.11, eq. 4)
        return g ** ((2 * x - 59) / (2 * b))


def _bandedge(g, b):
    # Band-edge ratio (ANSI s1.11, 3.7, pg. 3)
    return g ** (1 / (2 * b))
