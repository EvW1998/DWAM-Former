import pandas as pd
import numpy as np
import os
from pydub import AudioSegment

directory_path = "/hy-tmp/wwwdaicwoz/"
save_path = "/hy-tmp/DAIC_utterances/"

train_split = pd.read_csv(
    os.path.join(directory_path, "train_split_Depression_AVEC2017.csv"))[['Participant_ID', 'PHQ8_Binary']].to_numpy()
dev_split = pd.read_csv(
    os.path.join(directory_path, "dev_split_Depression_AVEC2017.csv"))[['Participant_ID', 'PHQ8_Binary']].to_numpy()
split_list = np.concatenate((train_split, dev_split), axis=0)

wav_list = []
count = 1
for participant in split_list:

    df = pd.read_csv(os.path.join(directory_path, str(participant[0]) + "_P", str(participant[0]) + "_TRANSCRIPT.csv"))
    df = pd.DataFrame(df['start_time\tstop_time\tspeaker\tvalue'].str.split('\t').tolist())
    df = df[df[2] == 'Participant'].reset_index()
    df['4'] = range(1, len(df) + 1)
    df['5'] = df.apply(lambda row: float(row[1]) - float(row[0]), axis=1)
    df = df.sort_values(by=['5', '4'], ascending=[False, True]).head(18 if int(participant[1]) == 0 else 46)
    df = df.drop('index', axis=1)

    for index, row in df.iterrows():
        t0 = float(row[0]) * 1000
        t1 = float(row[1]) * 1000
        newAudio = AudioSegment.from_wav(os.path.join(directory_path, str(participant[0]) + "_P", str(participant[0]) + "_AUDIO.wav"))
        newAudio = newAudio[t0:t1]
        newAudio.export(os.path.join(save_path, str(participant[0]) + "_s" + str(row['4']) + "_AUDIO.wav"), format="wav")
        wav_list.append(str(participant[0]) + "_s" + str(row['4']) + "_AUDIO.wav")

    print(count)
    count += 1

print(len(wav_list))
wav_df = pd.DataFrame(wav_list, columns=['name'])
print(wav_df)
wav_df.to_csv("/hy-tmp/wav_file.csv")
