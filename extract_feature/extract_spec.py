
import numpy as np
from scipy import io
import librosa
import pandas as pd
import os


def extract_spec(wavfile, savefile, nfft, hop):
    '''Calculate magnitude spectrogram from raw audio file by librosa.
    
    Args:
        wavfile: The absolute path to the audio file (.wav, .mp4).
        savefile: The absolute path to save the extracted feature in mat format. 
    '''
    y, _sr = librosa.load(wavfile, sr=None)
    M = np.abs(librosa.stft(y, n_fft=nfft, hop_length=hop, window='hamming'))

    data = {'spec': M}
    io.savemat(savefile, data)

    print(savefile, M.shape)


if __name__ == '__main__':
    sr = 16000     # sample rate  16000 (iemocap, daiz_woc) or 44100 (meld, pitt)
    frame = 0.02   # 20ms
    nfft = int(sr*frame)
    hop = nfft//2

    #### use extract_spec
    wav_list = pd.read_csv("~/SpeechFormer-master/metadata/metadata_daicwoz_crop_resample.csv")["name"].to_numpy()
    wav_dir = '/hy-tmp/DAIC_utterances'
    save_dir = '/hy-tmp/DAIC_feature/wav_spec_20ms_mat'

    for wav in wav_list:
        wav_file = os.path.join(wav_dir, wav)
        savefile = os.path.join(save_dir, wav[:-10] + ".mat")
        extract_spec(wav_file, savefile, nfft, hop)
