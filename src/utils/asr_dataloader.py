import os
import torch
import torch.nn as nn
import torch.utils.data as data
from src.datasets import ASRDataset

def get_dataloader(config, dataset_path, audio_transforms, tokenizer, converter, filter_spkr_ids=['all-spkrs'], is_training=True):

    # -- defining dataset
    dataset = ASRDataset(
        config,
        dataset_path=dataset_path,
        filter_spkr_ids=filter_spkr_ids,
    )

    # -- defining dataloader
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=config.training_settings['batch_size'] if is_training else 1,
        shuffle=is_training,
        collate_fn=lambda x: asr_data_processing(x, audio_transforms, tokenizer, converter, config),
        num_workers=config.training_settings['num_workers'],
        pin_memory=True,
    )

    return dataloader

def asr_data_processing(data, audio_transforms, tokenizer, converter, config):
    # -- create empty batch
    batch_keys = list(data[0].keys()) + ['speech_lengths', 'text_lengths', 'ref']
    if config.aux_ctc_tasks is not None:
        for task_id in config.aux_ctc_tasks:
            batch_keys += [task_id, f'{task_id}_lengths']

    batch = {data_key:[] for data_key in batch_keys}

    for sample in data:
        # -- sample metadata
        batch['sample_id'].append(sample['sample_id'])
        batch['speaker_id'].append(sample['speaker_id'])
        batch['language'].append(sample['language'])

        # -- audio preprocessing
        audio = audio_transforms(sample['speech']) if audio_transforms else sample['speech']
        audio = audio.transpose(1,0)
        audio_lengths = audio.shape[0]
        audio = audio[:audio_lengths // 640 * 640, :]

        batch['speech'].append(audio)
        batch['speech_lengths'].append(audio.shape[0])

        # -- auxiliary 'lid_utt' CTC task
        if config.aux_ctc_tasks is not None:
            for task_id in config.aux_ctc_tasks:
                if task_id == 'lid_utt':
                    # we remove an aritifical space symbol provided by the tokenizer
                    language_id = torch.Tensor(converter.tokens2ids(tokenizer.text2tokens(sample['language']))[1:])
                    batch[task_id].append(language_id)
                    batch[f'{task_id}_lengths'].append(1)
                else:
                    raise ValueError('unknown auxiliary CTC task ID: {task_id}')

        # -- transcription preprocessing
        text = torch.Tensor(converter.tokens2ids(tokenizer.text2tokens(sample['text'])))

        batch['text'].append(text)
        batch['text_lengths'].append(text.shape[0])

        # -- reference for evaluation
        batch['ref'].append(sample['text'])

    # -- speech sequence padding
    batch['speech'] = nn.utils.rnn.pad_sequence(batch['speech'], padding_value=config.model_conf['ignore_id'], batch_first=True).type(torch.float32) # -- (#batch, time, channel)
    batch['speech_lengths'] = torch.Tensor(batch['speech_lengths']).type(torch.int64) # -- (#batch,)

    if config.aux_ctc_tasks is not None:
        for task_id in config.aux_ctc_tasks:
            batch[task_id] = nn.utils.rnn.pad_sequence(batch[task_id], padding_value=config.model_conf['ignore_id'], batch_first=True).type(torch.int64) # -- (#batch, L)
            batch[f'{task_id}_lengths'] = torch.Tensor(batch[f'{task_id}_lengths']).type(torch.int64) # -- (#batch, )

    batch['text'] = nn.utils.rnn.pad_sequence(batch['text'], padding_value=config.model_conf['ignore_id'], batch_first=True).type(torch.int64) # -- (#batch, L)
    batch['text_lengths'] = torch.Tensor(batch['text_lengths']).type(torch.int64) # -- (#batch,)

    return batch
