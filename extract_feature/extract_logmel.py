
import soundfile 
import librosa
from scipy import signal
import numpy as np
from scipy import io
import pandas as pd
import os

def extract_logmel(wavfile: str, savefile: str, window_size: int, hop: int, ham_window, filter_bank):
    '''
    Args:
        wavfile: The absolute path to the audio file (.wav, .mp4).
        savefile: The absolute path to save the extracted feature in mat format. 
    '''
    x, fs = soundfile.read(wavfile)

    if x.ndim > 1:
        x = np.mean(x, axis=-1)

    f, t, x = signal.spectral.spectrogram(x, fs, window=ham_window, nperseg=window_size, noverlap=window_size - hop, detrend=False, mode='magnitude')
    x = np.dot(x.T, filter_bank.T).T   # Hz -> mel
    x = np.log(x + 1e-8)     # mel -> log_mel

    x = x.T   # -> (t, d)
    print(savefile, x.shape)

    dict = {'logmel': x}
    io.savemat(savefile, dict)

if __name__ == '__main__':
    frame = 0.025   # second
    hop = 0.01      # second
    sr = 16000      # sample rate  16000 (iemocap, daiz_woc) or 44100 (meld, pitt)
    window_size = int(sr * frame)
    hop = int(sr * hop)
    n_mels = 128
    ham_window = np.hamming(window_size)
    filter_bank = librosa.filters.mel(sr=sr, n_fft=window_size, n_mels=n_mels)

    #### use extract_logmel
    # wavfile = xxx
    # savefile = xxx
    # extract_logmel(wavfile, savefile, window_size, hop, ham_window, filter_bank)

    wav_list = pd.read_csv("~/SpeechFormer-master/metadata/metadata_daicwoz_crop_resample.csv")["name"].to_numpy()
    wav_dir = '/hy-tmp/DAIC_utterances'
    save_dir = '/hy-tmp/DAIC_feature/logmel_25ms_mat'

    for wav in wav_list:
        wav_file = os.path.join(wav_dir, wav)
        savefile = os.path.join(save_dir, wav[:-10] + ".mat")
        extract_logmel(wav_file, savefile, window_size, hop, ham_window, filter_bank)

    