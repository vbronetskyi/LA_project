import librosa
import numpy as np
import librosa.display
import json
from sklearn.neighbors import KNeighborsClassifier
import os
from tqdm import tqdm
import numpy as np
import scipy.fftpack as fft
from scipy.signal import hamming

def mel_filterbank(sr, n_fft, n_mels):
    mel_f = 2595 * np.log10(1 + np.arange(0, sr / 2, (sr / 2) / (n_fft // 2 + 1)) / 700)
    f_bins = np.floor((n_fft + 1) * mel_f / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus = f_bins[m - 1]
        f_m = f_bins[m]
        f_m_plus = f_bins[m + 1]

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return fbank

def mfcc(y, sr=22050, n_mfcc=20, n_fft=2048, hop_length=512):
    # Pre-emphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # Short-time Fourier Transform (STFT)
    frames = np.abs(
    fft.fft(hamming(n_fft) * y[:, np.newaxis], n_fft, axis=0)[:n_fft // 2 + 1, :])


    # Mel Filterbank
    mel_basis = mel_filterbank(sr, n_fft, n_mfcc)
    melspec = np.dot(frames.T, mel_basis.T)

    # Log Mel Spectrogram
    log_melspec = np.log(melspec + 1e-16)

    # Discrete Cosine Transform (DCT)
    mfcc_coeffs = fft.dct(log_melspec, axis=0, type=2, norm="ortho")[:n_mfcc]

    return mfcc_coeffs.T


def extract_mfcc(audio_file, window_size_ms=25, step_ms=10, n_mfcc=13):
    """Extracts Mel-frequency cepstral coefficients (MFCCs) from an audio file.

    Args:
        audio_file (str): Path to the audio file.
        window_size_ms (int, optional): Size of the analysis window in milliseconds. Defaults to 25.
        step_ms (int, optional): Step size between consecutive frames in milliseconds. Defaults to 10.
        n_mfcc (int, optional): Number of MFCCs to extract. Defaults to 13.

    Returns:
        numpy.ndarray: MFCC matrix, where rows represent frames and columns represent MFCC coefficients.
    """
    y, sr = librosa.load(audio_file, sr=None)
    y = np.array([i for i in y if abs(i) > 0.005])

    n_fft = int(sr * window_size_ms / 1000)
    hop_length = int(sr * step_ms / 1000)

    mfccs = mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    return mfccs.transpose()


class UsLeVoModel:
    """
    Model for classification peoplr by voice mfc coeficients
    """

    def __init__(self, k, win_size, step) -> None:
        self.space = KNeighborsClassifier(n_neighbors=k)
        self.w = win_size
        self.s = step

    def _get_majority(self, lst):
        mj = max(set(lst), key=list(lst).count)
        percent = len([1 for i in lst if i == mj]) / len(lst)
        return mj, round(percent, 3)

    def train(self, json_path, rec_dir):
        with open(json_path, "r", encoding="utf-8") as f:
            dct = json.load(f)
        classes = []
        data = []
        for user, value in dct.items():
            for rec_path in value["train"]:
                rec_mfcc = extract_mfcc(
                    rec_dir + os.sep + rec_path, window_size_ms=self.w, step_ms=self.s
                )
                for j in rec_mfcc:
                    data.append(j)
                    classes.append(user)
        self.space.fit(data, classes)

    def test(self, json_path, rec_dir):
        with open(json_path, "r", encoding="utf-8") as f:
            dct = json.load(f)

        retults = {"right": [], "wrong": []}

        for user, value in dct.items():
            for rec_path in value["validation"]:
                rec_mfcc = extract_mfcc(
                    rec_dir + os.sep + rec_path, window_size_ms=self.w, step_ms=self.s
                )
                prediction, prc = self._get_majority(self.space.predict(rec_mfcc))
                if prediction == user:
                    retults["right"].append(prc)
                else:
                    retults["wrong"].append(prc)
        return retults


dct_m = {}
dct_f = {}
for k in tqdm([20]):
    for win_size in [40]:
        # m = UsLeVoModel(k, win_size, int(win_size / 2.5))
        # m.train("src/male.json", "dataset/clips")
        # dct_m[f"k={k}, win_size={win_size}"] = m.test("src/male.json", "dataset/clips")
        f = UsLeVoModel(k, win_size, int(win_size / 2.5))
        f.train("src/female.json", "dataset/clips")
        dct_f[f"k={k}, win_size={win_size}"] = f.test("src/female.json", "dataset/clips")
        

# with open("male_results.json", "w", encoding="utf-8") as file:
#     json.dump(dct_m, file)
with open("female_results.json", "w", encoding="utf-8") as file:
    json.dump(dct_f, file)
