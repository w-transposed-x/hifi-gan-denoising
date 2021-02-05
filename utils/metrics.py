import numpy as np
from pypesq import pesq
from pystoi import stoi


def pesq_score(y_true: np.array, y_pred: np.array, samplerate=16000, mode='wb') -> float:
    """Computes the Perceptual Evaluation of Speech Quality metric between `y_true` and `y_pred`.

    Args:
        y_true (np.array): The original audio signal with shape (samplerate*length).
        y_pred (np.array): The predicted audio signal with shape (samplerate*length).
        samplerate (int, optional): Either 8000 or 16000. Defaults to 16000.
        mode (str, optional): Either 'wb' or 'nb'. 'wb' is only available for 16000Hz. Defaults to 'wb'.

    Returns:
        float: The pesq score between `y_true` and `y_pred`.
    """
    return pesq(ref=y_true, deg=y_pred, fs=samplerate)

def stoi_score(y_true: np.array, y_pred: np.array, samplerate=16000, extended=False) -> float:
    """Computes the Short Term Objective Intelligibility metric between `y_true` and `y_pred`.

    Args:
        y_true (np.array): The original audio signal with shape (samplerate*length).
        y_pred (np.array): The predicted audio signal with shape (samplerate*length).
        samplerate (int, optional): Either 8000 or 16000. Defaults to 16000.
        extended (boolean, optional): Whenever to use the extended stoi metric instead.

    Returns:
        float: The stoi score between `y_true` and `y_pred`.
    """
    return stoi(y_true, y_pred, fs_sig=samplerate, extended=extended)