import sys
import os
import subprocess
import json
import math
import scipy
import vosk  # tested with VOSK 0.3.15
import nltk
import librosa
import numpy
import pandas
from functools import lru_cache
from itertools import product as iterprod


vosk.SetLogLevel(-1)
transcribe_model_path = '/hy-tmp/vosk-model-en-us-0.22'  # https://alphacephei.com/vosk/models
sample_rate = 16000

if not os.path.exists(transcribe_model_path):
    raise ValueError(f"Could not find VOSK model at {transcribe_model_path}")

transcribe_model = vosk.Model(transcribe_model_path)
# transcribe_recognizer = vosk.KaldiRecognizer(transcribe_model, sample_rate)

try:
    arpabet = nltk.corpus.cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    arpabet = nltk.corpus.cmudict.dict()


def extract_words(res):
    jres = json.loads(res)
    if not 'result' in jres:
        return []
    words = jres['result']
    return words


def transcribe_words(recognizer, bytes):
    results = []

    chunk_size = 4000
    for chunk_no in range(math.ceil(len(bytes) / chunk_size)):
        start = chunk_no * chunk_size
        end = min(len(bytes), (chunk_no + 1) * chunk_size)
        data = bytes[start:end]

        if recognizer.AcceptWaveform(data):
            words = extract_words(recognizer.Result())
            results += words
    results += extract_words(recognizer.FinalResult())

    return results


def get_transcribe_words(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    int16 = numpy.int16(audio * 32768).tobytes()

    transcribe_recognizer = vosk.KaldiRecognizer(transcribe_model, sample_rate)
    res = transcribe_words(transcribe_recognizer, int16)
    df = pandas.DataFrame.from_records(res)

    if df.empty:
        return None
    else:
        return df.sort_values('start').drop('conf', axis=1)


@lru_cache()
def wordbreak(s):
    s = s.lower()

    if s in arpabet:
        return arpabet[s]
    middle = len(s) / 2
    partition = sorted(list(range(len(s))), key=lambda x: (x - middle) ** 2 - x)
    for i in partition:
        pre, suf = (s[:i], s[i:])
        if pre in arpabet and wordbreak(suf) is not None:
            return [x + y for x, y in iterprod(arpabet[pre], wordbreak(suf))]
    return None


def get_phoneme(transcribes):
    transcribes = transcribes.to_numpy()
    phonemes = []

    for t in transcribes:
        word = t[2]
        r = wordbreak(word)

        if r is not None:
            r = r[0]
        else:
            r = []

        phonemes.append([r, len(r)])

    return pandas.DataFrame(phonemes, columns=['phonemes', 'number'])


def main():
    audio_dir = '../DAIC_utterances'
    audio_paths = [f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]

    for audio_path in audio_paths:
        out_path = audio_path[:-9] + 'TRANSCRIBE.csv'
        mat_path = audio_path[:-9] + 'TRANSCRIBE.mat'
        out_path = os.path.join('/hy-tmp/DAIC_utterances_info', out_path)
        mat_path = os.path.join('/hy-tmp/DAIC_utterances_info', mat_path)

        audio_path = os.path.join(audio_dir, audio_path)
        transcribe_infos = get_transcribe_words(audio_path)
        
        if transcribe_infos is None:
            full_infos = pandas.DataFrame(columns=['end', 'start', 'word', 'phonemes', 'number'])
        else:
            phoneme_infos = get_phoneme(transcribe_infos)
            full_infos = pandas.concat([transcribe_infos, phoneme_infos], axis=1, join='inner')

        full_infos.to_csv(out_path, index=False, encoding="utf_8_sig")
        scipy.io.savemat(mat_path, {'transcribe': full_infos.to_numpy()})
        print('Word segments saved to', out_path)


if __name__ == '__main__':
    main()
