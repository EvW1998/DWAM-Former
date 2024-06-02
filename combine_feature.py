import scipy.io as scio
import os


def combine_f(feature_dir, save_dir):
    transcribe_dir = '/hy-tmp/DAIC_utterances_info'
    feature_paths = [f for f in os.listdir(feature_dir) if os.path.isfile(os.path.join(feature_dir, f))]

    for feature_path in feature_paths:
        transcribe_mat_path = os.path.join(transcribe_dir, feature_path[:-4] + '_TRANSCRIBE.mat')
        feature_mat_path = os.path.join(feature_dir, feature_path)
        save_mat_path = os.path.join(save_dir, feature_path)

        print('Process ', transcribe_mat_path)
        transcribe_data = scio.loadmat(transcribe_mat_path)
        feature_data = scio.loadmat(feature_mat_path)

        sentence_info = transcribe_data[list(transcribe_data.keys())[-1]].tolist()
        end_time = []
        start_time = []
        word = []
        phonemes = []
        phonemes_number = []

        for word_info in sentence_info:
            end_time.append(float(word_info[0][0]))
            start_time.append(float(word_info[1][0]))
            word.append(list(word_info[2])[0])
            phonemes.append(list(word_info[3]))
            phonemes_number.append(int(word_info[4][0]))

        feature_data['end_time'] = end_time
        feature_data['start_time'] = start_time
        feature_data['word'] = word
        feature_data['phonemes'] = phonemes
        feature_data['phonemes_number'] = phonemes_number

        scio.savemat(save_mat_path, feature_data)
        print('Feature updated to', save_mat_path)


if __name__ == '__main__':
    features_dir = '/hy-tmp/DAIC_learnable_features'
    features_dir_list = [os.path.join(features_dir, f) for f in os.listdir(features_dir)]
    print(features_dir_list)

    for d in features_dir_list:
        # save_dir = '/hy-tmp/DAIC_hc_features/wav_spec_20ms_mat_complete'
        save_dir = d + '_complete'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        combine_f(d, save_dir)
