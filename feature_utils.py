import numpy as np

import librosa
import skimage.util

def get_melspectrogram(X, sr=48000):
    def _get_melspectrogram(x):
        mel = librosa.feature.melspectrogram(y=x, sr=sr, win_length=1920, hop_length=960)# [np.newaxis, :]
        out = librosa.power_to_db(mel, ref=np.max)

        out = skimage.util.view_as_blocks(out[:, :150], (out.shape[0], out.shape[1]//10)).squeeze()
        return out

    X_features = np.apply_along_axis(_get_melspectrogram, 1, X)
    return X_features


def get_autocorrelate(X, sr=48000):
    def _get_autocorrelate(X):
        y = librosa.core.autocorrelate(X)
        y = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        y = np.transpose(y, (1, 0))
        return y

    X_features = np.apply_along_axis(_get_autocorrelate, 1, X)
    return X_features
