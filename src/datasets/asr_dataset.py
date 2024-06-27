import numpy as np
import pandas as pd
from unidecode import unidecode

import torch
import torchaudio
from torch.utils.data import Dataset

class ASRDataset(Dataset):
    """Dataset to load the BBS-S2TC data.
    """

    def __init__(self, config, dataset_path, filter_spkr_ids=['all-spkrs'], filter_by_language=['all-langs']):
        # -- config
        self.config = config

        # -- reading dataset
        self.dataset = pd.read_csv(dataset_path, delimiter=',', dtype={"speaker_id": "string"})
        self.dataset['sample_id'] = self.dataset['path'].map(lambda x: x.split('/')[-1])

        # prctg = int(len(self.dataset) * 0.1)
        # self.dataset = self.dataset[:prctg]

        # -- filtering by duration
        self.dataset = self.dataset[self.dataset['length'] <= 13]

        # -- filtering targeted speakers
        if 'all-spkrs' not in filter_spkr_ids:
            self.dataset = self.dataset[self.dataset['speaker_id'].isin(filter_spkr_ids)]

        # -- filtering targeted language
        if 'all-langs' not in filter_by_language:
            self.dataset = self.dataset[self.dataset['language'].isin(filter_by_language)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = {}

        # -- sample metadata
        sample['sample_id'] = self.dataset.iloc[index]['sample_id']
        sample['speaker_id'] = self.dataset.iloc[index]['speaker_id']
        sample['language'] = f'<{self.dataset.iloc[index]["language"].upper()}>'

        # -- input and output model data
        sample['speech'] = self.__get_speech_sample__(index)
        sample['text'] = self.__get_text_sample__(index)

        return sample

    def __get_speech_sample__(self, index):
        wav_path = self.dataset.iloc[index]['path']
        waveform, sample_rate = torchaudio.load(wav_path, normalize=True)

        return waveform # -- (T,)

    def __get_text_sample__(self, index):
        return self.dataset.iloc[index]['sentence'].strip().lower() # -- (L,)
