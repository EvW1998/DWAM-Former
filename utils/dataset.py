import pandas as pd
import numpy as np
import os
import math
import torch
import re
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from scipy import io
from utils.speech_kit import Speech_Kit, get_D_P
import multiprocessing as mp
from sklearn.model_selection import StratifiedShuffleSplit
from gensim.models import KeyedVectors

wv = KeyedVectors.load("./pre_trained_model/word2vec.wordvectors", mmap='r')


def identity(x):
    return x


class DistributedDalaloaderWrapper():
    def __init__(self, dataloader: DataLoader, collate_fn):
        self.dataloader = dataloader
        self.collate_fn = collate_fn

    def _epoch_iterator(self, it):
        for batch in it:
            yield self.collate_fn(batch)

    def __iter__(self):
        it = iter(self.dataloader)
        return self._epoch_iterator(it)

    def __len__(self):
        return len(self.dataloader)

    @property
    def dataset(self):
        return self.dataloader.dataset

    def set_epoch(self, epoch: int):
        self.dataloader.sampler.set_epoch(epoch)


def universal_collater(batch):
    all_data = [[] for _ in range(len(batch[0]))]
    for one_batch in batch:
        for i, (data) in enumerate(one_batch):
            all_data[i].append(data)
    return all_data


class Base_database():
    def __init__(self, names, labels, matdir=None, matkey=None, state=None, label_conveter=None):
        self.names = names
        self.labels = labels
        self.state = state
        self.matdir = matdir
        self.matkey = matkey
        self.conveter = label_conveter

    def get_wavfile_label(self, name):
        idx = self.names.index(name)
        label = self.labels[idx]
        return label

    def load_a_sample(self, idx=0):
        label = self.labels[idx]

        mat_data = io.loadmat(os.path.join(self.matdir, self.names[idx]))

        end_time = mat_data['end_time'][0]
        start_time = mat_data['start_time'][0]
        word = mat_data['word']
        phonemes = mat_data['phonemes'][0]
        phonemes_num = []

        phonemes_numpy = wv[phonemes[0]]
        phonemes_num.append(len(phonemes[0]))
        for i in range(1, len(phonemes)):
            phonemes_numpy = np.vstack((phonemes_numpy, wv[phonemes[i]]))
            phonemes_num.append(len(phonemes[i]))

        phonemes = torch.tensor(phonemes_numpy)

        del phonemes_numpy

        x = np.float32(mat_data[self.matkey])
        y = torch.tensor(self.label_2_index(label))

        return x, y, [end_time, start_time, word, phonemes, phonemes_num]

    def get_sample_name(self, idx):
        return self.names[idx]

    def label_2_index(self, label):
        index = self.conveter[label]
        return index


class Base_dataset(Dataset):
    def __init__(self, database: Base_database, mode, length, feature_dim, pad_value, load_name=False):
        super().__init__()
        self.database = database
        self.kit = Speech_Kit(mode, length, feature_dim, pad_value)
        self.load_name = load_name

    def __len__(self):
        return len(self.database.names)

    def __getitem__(self, idx):
        return _getitem(idx, self.database, self.kit, self.load_name)


class DAIC_WOZ(Base_database):
    def __init__(self, matdir=None, matkey=None, state=None, meta_csv_file=None):
        assert state in ['train', 'test'], print(
            f'Wrong state: {state}')  # test represents the development set in this database.
        self.set_state(state)
        names, labels = self.load_state_data(meta_csv_file)

        label_conveter = {'not-depressed': 0, 'depressed': 1}
        super().__init__(names, labels, matdir, matkey, state, label_conveter)

    def set_state(self, state):
        if state == 'test':
            state = 'dev'
        self.state = state

    def load_state_data(self, meta_csv_file):
        df = pd.read_csv(meta_csv_file)

        df = df[df.state == self.state]

        names, indexes = [], []
        index_2_label = {0: 'not-depressed', 1: 'depressed'}
        for row in df.iterrows():
            names.append(row[1]['name'])
            indexes.append(row[1]['label'])
        labels = [index_2_label[idx] for idx in indexes]

        mat_names = [n[:-10] for n in names]  # (Participant_ID)_(segment)

        return mat_names, labels


class DAIC_WOZ_dataset(Base_dataset):
    def __init__(self, matdir, matkey, state, meta_csv_file, length=0, feature_dim=0,
                 pad_value=0, mode='constant', **kwargs):
        database = DAIC_WOZ(matdir, matkey, state, meta_csv_file)
        super().__init__(database, mode, length, feature_dim, pad_value, load_name=True)

        # if state == 'train':
        # self.resample_up()
        # self.resample_down()

    def resample_up(self):
        names = self.database.names
        labels = self.database.labels
        num_d = labels.count('depressed')
        num_nd = labels.count('not-depressed')

        r = num_nd // num_d
        s = num_nd % num_d
        index_d = np.arange(num_d)
        random.shuffle(index_d)

        d_index = [index for (index, value) in enumerate(labels) if value == 'depressed']
        d_names = [names[index] for index in d_index]
        d_labels = [labels[index] for index in d_index]

        d_names_s = [d_names[index_d[i]] for i in range(s)]
        d_labels_s = [d_labels[index_d[i]] for i in range(s)]

        d_names = d_names * (r - 1)
        d_labels = d_labels * (r - 1)

        d_names.extend(d_names_s)
        d_labels.extend(d_labels_s)

        names.extend(d_names)
        labels.extend(d_labels)

        self.database.names = names
        self.database.labels = labels

    def resample_down(self):
        names = self.database.names
        labels = self.database.labels
        num_d = labels.count('depressed')
        num_nd = labels.count('not-depressed')

        index_nd = np.arange(num_nd)
        random.shuffle(index_nd)

        nd_index = [index for (index, value) in enumerate(labels) if value == 'not-depressed']
        nd_names = [names[index] for index in nd_index]
        nd_labels = [labels[index] for index in nd_index]

        nd_names = [nd_names[index_nd[i]] for i in range(num_d)]
        nd_labels = [nd_labels[index_nd[i]] for i in range(num_d)]

        d_index = [index for (index, value) in enumerate(labels) if value == 'depressed']
        d_names = [names[index] for index in d_index]
        d_labels = [labels[index] for index in d_index]

        names = d_names + nd_names
        labels = d_labels + nd_labels

        self.database.names = names
        self.database.labels = labels


class DataloaderFactory():
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, state, **kwargs):
        """
        data_json {'feature_dim': 1024,
                   'lmdb_root': '/hy-tmp/DAIC_feature//daic_woz_hubert_L12_v6',
                   'matdir': '../DAIC_feature/hubert_large_L12_mat_complete',
                   'matkey': 'hubert',
                   'length': 426,
                   'pad_value': 0,
                   'frame': 0.025,
                   'hop': 0.02,
                   'meta_csv_file': '~/DWAM-Former/metadata/metadata_daicwoz_crop_resample.csv'}
        """
        if self.cfg.dataset.database == 'daic_woz':
            dataset = DAIC_WOZ_dataset(
                mode=self.cfg.dataset.padmode,  # constant
                state=state,  # train
                **kwargs
            )
        else:
            raise KeyError(f'Unsupported database: {self.cfg.dataset.database}')

        collate_fn = universal_collater
        sampler = DistributedSampler(dataset, shuffle=state == 'train')
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.train.batch_size,
            drop_last=False,
            num_workers=self.cfg.train.num_workers,
            collate_fn=identity,
            sampler=sampler,
            pin_memory=True,
            multiprocessing_context=mp.get_context('fork'),
            # quicker! Used with multi-process loading (num_workers > 0)
        )

        return DistributedDalaloaderWrapper(dataloader, collate_fn)


def _getitem(idx: int, database: Base_database, kit: Speech_Kit, load_name: bool):
    x, y, phonemes_info = database.load_a_sample(idx)

    if database.matkey == 'spec':
        x, _ = get_D_P(x)  # ndarray
        x = x.transpose(1, 0)
    new_x, mask = kit.pad_spec(x)  # ndarray -> Tensor

    name = database.get_sample_name(idx) if load_name else None

    return new_x, y, name, phonemes_info, mask
