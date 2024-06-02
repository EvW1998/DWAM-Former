import os
from scipy import io
from gensim.models import Word2Vec

if __name__ == '__main__':
    features_dir = '../DAIC_feature/hubert_large_L24_mat_complete'
    features_dir_list = [os.path.join(features_dir, f) for f in os.listdir(features_dir)]
    p_list = []

    for d in features_dir_list:
        try:
            mat_data = io.loadmat(d)
            phonemes = mat_data['phonemes'][0]

            for p in phonemes:
                for i in p:
                    if i not in p_list:
                        p_list.append(i)
        except:
            pass

    model = Word2Vec(sentences=[p_list], vector_size=100, window=5, min_count=1, workers=4)
    word_vectors = model.wv
    word_vectors.save("./pre_trained_model/word2vec.wordvectors")
