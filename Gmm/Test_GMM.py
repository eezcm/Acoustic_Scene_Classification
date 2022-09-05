# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import librosa
import pickle
import pyaudio
import time
import threading
import wave

# 定义类
class Recorder:
    def __init__(self, chunk=1024, channels=1, rate=64000):
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self._running = True
        self._frames = []

    # 定义开始录音
    def start(self):
        threading._start_new_thread(self.__recording, ())

    # 定义录音
    def __recording(self):
        self._running = True
        self._frames = []
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        while self._running:
            data = stream.read(self.CHUNK)
            self._frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

    # 定义停止
    def stop(self):
        self._running = False

    # 定义保存
    def save(self, filename):

        p = pyaudio.PyAudio()
        if not filename.endswith(".wav"):
            filename = filename + ".wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self._frames))
        wf.close()
        #print("Saved")


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


def mfcc(wav_path, delta=2):

    y, sr = librosa.load(wav_path)
    mfcc_feat = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 16)
    ans = [mfcc_feat]
    if delta >= 1:
        mfcc_delta1 = librosa.feature.delta(mfcc_feat, order = 1, mode ='nearest')
        ans.append(mfcc_delta1)
    if delta >= 2:
        mfcc_delta2 = librosa.feature.delta(mfcc_feat, order = 2, mode ='nearest')
        ans.append(mfcc_delta2)

    return np.transpose(np.concatenate(ans, axis = 0),[1,0])


def get_mfcc_data(train_dir):

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

def log_gaussian_prob(x, means, var):

    return (-0.5 * np.log(var) - np.divide(np.square(x - means), 2 * var) - 0.5 * np.log(2 * np.pi)).sum()


class GMM:

    def __init__(self, mfcc_data, n_components, random_state=0):
        # Initialization
        self.mfcc_data = mfcc_data
        self.means = np.tile(np.mean(self.mfcc_data, axis=0), (n_components, 1))
        # randomization
        np.random.seed(random_state)
        for k in range(n_components):
            randn_k = np.random.randn()
            self.means[k] += 0.01 * randn_k * np.sqrt(np.var(self.mfcc_data, axis=0))
        self.var = np.tile(np.var(self.mfcc_data, axis=0), (n_components, 1))
        self.weights = np.ones(n_components) / n_components
        self.n_components = n_components

    def e_step(self, x):

        log_resp = np.zeros((x.shape[0], self.n_components))
        for i in range(x.shape[0]):
            log_resp_i = np.log(self.weights)
            for j in range(self.n_components):
                log_resp_i[j] += log_gaussian_prob(x[i], self.means[j], self.var[j])
            y = np.exp(log_resp_i - log_resp_i.max())
            log_resp[i] = y / y.sum()
        return log_resp

    def m_step(self, x, log_resp):

        self.weights = np.sum(log_resp, axis=0) / np.sum(log_resp)
        denominator = np.sum(log_resp, axis=0, keepdims=True).T
        means_num = np.zeros_like(self.means)
        for k in range(self.n_components):
            means_num[k] = np.sum(np.multiply(x, np.expand_dims(log_resp[:, k], axis=1)), axis=0)
        self.means = np.divide(means_num, denominator)
        var_num = np.zeros_like(self.var)
        for k in range(self.n_components):
            var_num[k] = np.sum(np.multiply(np.square(np.subtract(x, self.means[k])),
                                            np.expand_dims(log_resp[:, k], axis=1)), axis=0)
        self.var = np.divide(var_num, denominator)

    def train(self, x):

        log_resp = self.e_step(x)
        self.m_step(x, log_resp)

    def log_prob(self, x):

        sum_prob = 0
        for i in range(x.shape[0]):
            prob_i = np.array([np.log(self.weights[j]) + log_gaussian_prob(x[i], self.means[j], self.var[j])
                               for j in range(self.n_components)])
            sum_prob += np.max(prob_i)
        return sum_prob


def train_model_gmm(train_dir, n_components, max_iter=300, random_state=0):

    gmm_models = []
    mfcc_data = get_mfcc_data(train_dir)
    for i in range(10):
        gmm_models.append(GMM(mfcc_data[i], n_components=n_components, random_state=random_state))
    iter = 1
    lower_bound = -np.infty
    while iter <= max_iter:
        prev_lower_bound = lower_bound
        total_log_like = 0.0
        for i in range(10):
            gmm_models[i].train(mfcc_data[i])
            total_log_like += gmm_models[i].log_prob(mfcc_data[i])
        lower_bound = total_log_like
        if abs(lower_bound - prev_lower_bound) < 1:
            break
        print("Iteration:", iter, " log_prob:", lower_bound)
        iter += 1

    return gmm_models


def predict_gmm(gmm_models, test_dir):

    count = 0
    pred_true = 0
    # append all test records and start digit recognition
    test_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            # Make sure the suffix is correct and avoid the influence of hidden files
            if os.path.splitext(file)[1] == '.wav':
                test_files.append(os.path.join(root, file))
    for test_file in test_files:
        # get mfcc feature and ignore the warning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            features_mfcc = mfcc(test_file)
        # calculate the score and get the maximum score
        sum_prob = []
        for i in range(10):
            prob_i = gmm_models[i].log_prob(features_mfcc)
            sum_prob.append(prob_i)
        pred = str(np.argmax(np.array(sum_prob)))
        if pred == '0':
            print("pred_number:", pred, "   ", "pred_name:", "tank_move")
        elif pred == '1':
            print("pred_number:", pred, "   ", "pred_name:", "tank_shoot")
        elif pred == '2':
            print("pred_number:", pred, "   ", "pred_name:", "truck_move")
        elif pred == '3':
            print("pred_number:", pred, "   ", "pred_name:", "truck_horn")
        elif pred == '4':
            print("pred_number:", pred, "   ", "pred_name:", "train_move")
        elif pred == '5':
            print("pred_number:", pred, "   ", "pred_name:", "train_horn")
        elif pred == '6':
            print("pred_number:", pred, "   ", "pred_name:", "fighter")
        elif pred == '7':
            print("pred_number:", pred, "   ", "pred_name:", "helicopter")
        elif pred == '8':
            print("pred_number:", pred, "   ", "pred_name:", "motorcycle")
        elif pred == '9':
            print("pred_number:", pred, "   ", "pred_name:", "missile")
    return pred

if __name__ == '__main__':
    start = int(input('请输入数字1开始录音:'))
    if start == 1:
        rec = Recorder()
        begin = time.time()
        print("Start recording")
        rec.start()
        over = int(input('请输入数字2停止录音:'))
        if over == 2:
            print("Stop recording")
            rec.stop()
            fina = time.time()
            t = fina - begin
            print('录音时间为%ds' % t)
            rec.save("./processed_predict_records/test.wav")

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

    predict_dir = './processed_predict_records'
    f = open('gmm_models_16.pkl', 'rb')
    gmm_models = pickle.load(f)
    f.close()
    p = predict_gmm(gmm_models, predict_dir)
    #print(p)
