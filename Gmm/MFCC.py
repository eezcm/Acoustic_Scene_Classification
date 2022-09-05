import os
import warnings
import numpy as np
import librosa


def mfcc(wav_path, delta=2):
    """
    Read .wav files and calculate MFCC
    :param wav_path: path of audio file
    :param delta: derivative order, default order is 2
    :return: mfcc
    """
    y, sr = librosa.load(wav_path)
    # MEL frequency cepstrum coefficient
    mfcc_feat = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 15)
    ans = [mfcc_feat]
    # Calculate the 1st derivative
    if delta >= 1:
        mfcc_delta1 = librosa.feature.delta(mfcc_feat, order = 1, mode ='nearest')
        ans.append(mfcc_delta1)
    # Calculate the 2nd derivative
    if delta >= 2:
        mfcc_delta2 = librosa.feature.delta(mfcc_feat, order = 2, mode ='nearest')
        ans.append(mfcc_delta2)

    return np.transpose(np.concatenate(ans, axis = 0),[1,0])


def get_mfcc_data(train_dir):
    """
    :param train_dir: training directory
    :return: mfcc_data
    """
    mfcc_data = []
    for i in range(10):
        digit_mfcc = np.array([])
        digit = str(i)
        digit_dir = os.path.join(train_dir, 'digit_' + digit)
        train_files = [x for x in os.listdir(digit_dir) if x.endswith('.wav')]
        for file_name in train_files:
            file_path = os.path.join(digit_dir, file_name)
            # get mfcc feature and ignore the warning
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                features_mfcc = mfcc(file_path)
            # append mfcc to X
            if len(digit_mfcc) == 0:
                digit_mfcc = features_mfcc
            else:
                digit_mfcc = np.append(digit_mfcc, features_mfcc, axis=0)
        mfcc_data.append(digit_mfcc)

    return mfcc_data