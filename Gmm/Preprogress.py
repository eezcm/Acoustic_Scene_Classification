# -*- coding: utf-8 -*-
import os
import wave
import numpy as np

def calEnergy(wave_data):
    """
    :param wave_data: binary data of audio file
    :return: energy
    """
    energy = []
    sum = 0
    for i in range(len(wave_data)):
        sum = sum + (int(wave_data[i]) * int(wave_data[i]))
        if (i + 1) % 256 == 0:
            energy.append(sum)
            sum = 0
        elif i == len(wave_data) - 1:
            energy.append(sum)
    return energy


def calZeroCrossingRate(wave_data):
    """
    :param wave_data: binary data of audio file
    :return: ZeroCrossingRate
    """
    zeroCrossingRate = []
    sum = 0
    for i in range(len(wave_data)):
        sum = sum + np.abs(int(wave_data[i] >= 0) - int(wave_data[i - 1] >= 0))
        if (i + 1) % 256 == 0:
            zeroCrossingRate.append(float(sum) / 255)
            sum = 0
        elif i == len(wave_data) - 1:
            zeroCrossingRate.append(float(sum) / 255)
    return zeroCrossingRate


def endPointDetect(energy, zeroCrossingRate):
    """
    :param energy: energy
    :param zeroCrossingRate: zeroCrossingRate
    :return: data after endpoint detection
    """
    sum = 0
    for en in energy:
        sum = sum + en
    avg_energy = sum / len(energy)

    sum = 0
    for en in energy[:5]:
        sum = sum + en
    ML = sum / 5
    MH = avg_energy / 5  # high energy threshold
    ML = (ML + MH) / 5  # low energy threshold

    sum = 0
    for zcr in zeroCrossingRate[:5]:
        sum = float(sum) + zcr
    Zs = sum / 5  # zero crossing rate threshold

    A = []
    B = []
    C = []

    # MH is used for preliminary detection
    flag = 0
    for i in range(len(energy)):
        if len(A) == 0 and flag == 0 and energy[i] > MH:
            A.append(i)
            flag = 1
        elif flag == 0 and energy[i] > MH and i - 21 > A[len(A) - 1]:
            A.append(i)
            flag = 1
        elif flag == 0 and energy[i] > MH and i - 21 <= A[len(A) - 1]:
            A = A[:len(A) - 1]
            flag = 1

        if flag == 1 and energy[i] < MH:
            # if frame is too short, remove it
            if i - A[len(A) - 1] <= 2:
                A = A[:len(A) - 1]
            else:
                A.append(i)
            flag = 0

    # ML is used for second detection
    for j in range(len(A)):
        i = A[j]
        if j % 2 == 1:
            while i < len(energy) and energy[i] > ML:
                i = i + 1
            B.append(i)
        else:
            while i > 0 and energy[i] > ML:
                i = i - 1
            B.append(i)

    # zero crossing rate threshold is for the last step
    for j in range(len(B)):
        i = B[j]
        if j % 2 == 1:
            while i < len(zeroCrossingRate) and zeroCrossingRate[i] >= 3 * Zs:
                i = i + 1
            C.append(i)
        else:
            while i > 0 and zeroCrossingRate[i] >= 3 * Zs:
                i = i - 1
            C.append(i)
    return C


if __name__ == '__main__':
    # make directory to save processed records, divide records into train and test 4:1
    if not os.path.exists("./processed_predict_records/"):
        os.makedirs("./processed_predict_records/")
    records_path = "./processed_predict_records/"
    f = wave.open(records_path + 'test' + ".wav", "rb")
    # get the channels, sample_width, frame_rate and frames num of wav file
    channels, sample_width, frame_rate, frames = f.getparams()[:4]
    # convert data to binary array
    wave_data = np.frombuffer(f.readframes(frames), dtype=np.short)
    f.close()
    # end point detection
    energy = calEnergy(wave_data)
    zeroCrossingRate = calZeroCrossingRate(wave_data)
    N = endPointDetect(energy, zeroCrossingRate)
    # output
    m = 0
    while m < len(N):
        save_path = "./processed_predict_records/" + 'test' + ".wav"
        # save the data to a wav file
        wf = wave.open(save_path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(frame_rate)
        wf.writeframes(b"".join(wave_data[N[m] * 256: N[m + 1] * 256]))
        wf.close()
        m = m + 2