import os
import math
import random
import argparse
import pandas as pd
from tqdm import tqdm

import audeer
import audonnx
import librosa
import numpy as np

if __name__ == "__main__":
    """Please refer to:
        · https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender
        · https://github.com/audeering/w2v2-age-gender-how-to?tab=readme-ov-file
    """
    # -- command line arguments
    data_root = '../../data/LIP-RTVE/WAVs/'
    df_path = './liprtve_gender_metadata.csv'
    df = pd.read_csv(df_path)
    output_path = './liprtve_speaker_metadata.csv'


    url = 'https://zenodo.org/record/7761387/files/w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip'
    cache_root = audeer.mkdir('cache')
    model_root = audeer.mkdir('model')

    archive_path = audeer.download_url(url, cache_root, verbose=True)
    audeer.extract_archive(archive_path, model_root)
    model = audonnx.load(model_root)
    id2gender = {0: 'female', 1: 'male', 2: 'child'}

    age_per_spkr = []
    gender_per_spkr = []
    for spkr_id in tqdm(df['speaker_id'].tolist()):
        spkr_dir = os.path.join(data_root, spkr_id)
        spkr_samples = os.listdir(spkr_dir)

        random.shuffle(spkr_samples)
        samples_to_process = spkr_samples[:5]

        spkr_ages = []
        spkr_genders = []
        for sample in samples_to_process:
            sample_path = os.path.join(spkr_dir, sample)
            signal, sr = librosa.load(sample_path, sr=16000)

            output_dict = model(signal, 16000)

            spkr_age = output_dict['logits_age'][0][0]; spkr_ages.append(spkr_age)
            spkr_gender = id2gender[np.argmax( output_dict['logits_gender'] )]; spkr_genders.append(spkr_gender)

        age = math.ceil( np.array(spkr_ages).mean() * 100 ); age_per_spkr.append(age)
        gender = max(set(spkr_genders), key=spkr_genders.count); gender_per_spkr.append(gender)

    df['gender'] = gender_per_spkr
    df['age'] = age_per_spkr
    df.to_csv(output_path)
