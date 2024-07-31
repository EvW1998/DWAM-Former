
import soundfile 
import librosa
from scipy import signal
import numpy as np
from scipy import io
from scipy.io import wavfile
import pandas as pd
import os

def extract_fbank(inputfile: str, savefile: str):
    fs, wavdata = wavfile.read(inputfile)  # read a wav file into numpy array

    # Load weight
    frame_len = 25  # each frame length (ms)
    frame_shift = 10  # frame shift length (ms)
    frame_len_samples = frame_len * fs // 1000  # each frame length (samples)
    frame_shift_samples = frame_shift * fs // 1000  # frame shifte length (samples)
    total_frames = int(
        np.ceil((len(wavdata) - frame_len_samples) / float(frame_shift_samples)) + 1)  # total frames will get
    padding_length = int((total_frames - 1) * frame_shift_samples + frame_len_samples - len(
        wavdata))  # how many samples last frame need to pad

    pad_data = np.pad(wavdata, (0, padding_length), mode='constant')  # pad last frame with zeros
    frame_data = np.zeros((total_frames, frame_len_samples))  # where we save the frame results
    pre_emphasis_coeff = 0.97  # Pre-emphasis coefficient
    pad_data = np.append(pad_data[0], pad_data[1:] - pre_emphasis_coeff * pad_data[:-1])  # Pre-emphasis

    # hamming window
    window_func = np.hamming(frame_len_samples)

    for i in range(total_frames):
        single_frame = pad_data[
                       i * frame_shift_samples:i * frame_shift_samples + frame_len_samples]  # original frame data
        single_frame = single_frame * window_func  # add window function
        frame_data[i, :] = single_frame

    # FFT
    K = 512  # length of DFT
    freq_domain_data = np.fft.rfft(frame_data, K)  # DFT

    # calcualte power spectrum
    power_spec = np.absolute(freq_domain_data) ** 2 * (1 / K)  # power spectrum

    # Mel
    low_frequency = 20  # We don't use start from 0 Hz because human ear is not able to perceive low frequency signal.
    high_frequency = fs // 2  # if the speech is sampled at f Hz then our upper frequency is limited to 2/f Hz.
    low_frequency_mel = 2595 * np.log10(1 + low_frequency / 700)
    high_frequency_mel = 2595 * np.log10(1 + high_frequency / 700)
    n_filt = 40  # number of mel-filters (usually between 22-40)
    mel_points = np.linspace(low_frequency_mel, high_frequency_mel, n_filt + 2)  # Make the Mel scale spacing equal.
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # convert back to Hz scale.
    bins = np.floor((K + 1) * hz_points / fs)  # round those frequencies to the nearest FFT bin.

    fbank = np.zeros((n_filt, int(np.floor(K / 2 + 1))))
    for m in range(1, n_filt + 1):
        f_m_minus = int(bins[m - 1])  # left point
        f_m = int(bins[m])  # peak point
        f_m_plus = int(bins[m + 1])  # right point

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    filter_bank = np.matmul(power_spec, fbank.T)  # This is known as fbank feature.
    filter_bank = np.where(filter_bank == 0, np.finfo(float).eps,
                           filter_bank)  # Repalce 0 to a small constant or it will cause problem to log.

    # taking log
    log_fbank = np.log(filter_bank)
    # calcualtion is done with FBank
    print(savefile, log_fbank.shape)

    dict = {'fbank': log_fbank}
    io.savemat(savefile, dict)


if __name__ == '__main__':

    wav_list = pd.read_csv("~/SpeechFormer-master/metadata/metadata_daicwoz_crop_resample.csv")["name"].to_numpy()
    wav_dir = '/hy-tmp/DAIC_utterances'
    save_dir = '/hy-tmp/DAIC_feature/fbank_mat'

    for wav in wav_list:
        wav_file = os.path.join(wav_dir, wav)
        savefile = os.path.join(save_dir, wav[:-10] + ".mat")
        extract_fbank(wav_file, savefile)
