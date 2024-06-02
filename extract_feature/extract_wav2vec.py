
from scipy import io
import soundfile
import torch
import numpy as np
import scipy.signal as signal
from fairseq.models.wav2vec import Wav2VecModel
import pandas as pd
import os

def extract_wav2vec(wavfile, savefile):
    '''
    Args:
        wavfile: The absolute path to the audio file (.wav, .mp4).
        savefile: The absolute path to save the extracted feature in mat format. 
    '''
    wavs, fs = soundfile.read(wavfile)
    
    if fs != sample_rate:
        result = int((wavs.shape[0]) / fs * sample_rate)
        wavs = signal.resample(wavs, result)

    if wavs.ndim > 1:
        wavs = np.mean(wavs, axis=1)

    wavs = torch.from_numpy(np.float32(wavs)).unsqueeze(0)    # (B, S)
    
    z = wav2vec.feature_extractor(wavs)
    # z = wav2vec.vector_quantizer(z)['x']    # vq-wav2vec
    feature_wav = wav2vec.feature_aggregator(z)
    feature_wav = feature_wav.transpose(1,2).squeeze().detach().numpy()   # (t, 512)
    dict = {'wav': feature_wav}
    io.savemat(savefile, dict)
    
    print(savefile, feature_wav.shape)

if __name__ == '__main__':
    '''
    Pre-trained wav2vec model is available at https://github.com/pytorch/fairseq/blob/main/examples/wav2vec.
    Download model and save at model_path.
    '''
    path = './pre_trained_model/wav2vec'
    model_path = os.path.join(path, "wav2vec_large.pt")

    # model_path = '~/SpeechFormer-master/pre_trained_model/wav2vec/wav2vec_large.pt'
    sample_rate = 16000    # input should be resampled to 16kHz!
    cp = torch.load(model_path, map_location='cpu')
    wav2vec = Wav2VecModel.build_model(cp['args'], task=None)
    wav2vec.load_state_dict(cp['model'])
    wav2vec.eval()

    #### use extract_wav2vec
    # wavfile = xxx
    # savefile = xxx
    # extract_wav2vec(wavfile, savefile)

    wav_list = pd.read_csv("~/SpeechFormer-master/metadata/metadata_daicwoz_crop_resample.csv")["name"].to_numpy()
    wav_dir = '/hy-tmp/DAIC_utterances'
    save_dir = '/hy-tmp/DAIC_feature/wav_wav2vec_mat'

    for wav in wav_list:
        wav_file = os.path.join(wav_dir, wav)
        savefile = os.path.join(save_dir, wav[:-10] + ".mat")
        extract_wav2vec(wav_file, savefile)
    
