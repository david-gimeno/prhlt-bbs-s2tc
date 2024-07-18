#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.tasks import ASRTask
from src.evaluation import compute_bootstrap_wer
from espnet2.torch_utils.model_summary import model_summary

import os
import sys
import yaml
import random
import argparse
from tqdm import tqdm
from colorama import Fore
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from src.utils import *
from src.transforms import *
from torchvision import transforms

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def training(e2e, train_loader, optimizer, scheduler, accum_grad, scaler=None):
    e2e.train()

    # -- training
    train_loss = 0.0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(tqdm(train_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.GREEN, Fore.RESET))):
        batch = {k: v.to(device=config.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batch.items()}

        # -- forward
        loss = e2e(**batch)[0] / config.training_settings['accum_grad']

        # -- backward
        loss.backward()

        # -- update
        if ((batch_idx+1) % accum_grad == 0) or (batch_idx+1 == len(train_loader)):
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        train_loss += loss.item()

    return train_loss / (len(train_loader) / accum_grad)

def validation(e2e, data_loader):
    e2e.eval()
    data_loss = 0.0
    data_cer = 0.0

    # -- validation
    with torch.no_grad():
        for batch in tqdm(data_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
            batch = {k: v.to(device=config.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batch.items()}

            # -- forward
            loss, stats, weight = e2e(**batch)

            data_loss += loss.item()
            data_cer += stats["cer_ctc"].item() * 100.0

    return round(data_loss / len(data_loader), 3), round(data_cer / len(data_loader), 3)

def inference(output_dir, speech2text, eval_loader, dataset):
    print(f"Decoding {dataset.upper()} dataset:")

    # -- obtaining hypothesis
    lang_preds = []
    lang_refs = []
    lang_mapping = {'<EU>': 0, '<ES>': 1, '<BI>': 2}

    dst_dir = os.path.join(output_dir, "inference/")
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, dataset+".inf")
    with open(dst_path, "w") as f:
        with torch.no_grad():
            for batch in tqdm(eval_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.YELLOW, Fore.RESET)):
                result = speech2text(torch.squeeze(batch['speech'], 0))

                hyp = result[0][0]

                if args.output_for_submission:
                    sample_id = os.path.basename(batch['path'])
                    with open(args.output_for_submission; 'w', encoding='utf-8') as f:
                        f.write(f'{sample_id} {hyp.strip()}\n')

                else:
                    # -- dumping results
                    f.write(batch['ref'][0].strip() + "#" + hyp.strip() + "\n")

                    # -- language identification
                    lang_hyp = result[0][-1]
                    lang_ref = batch['language'][0]
                    if lang_hyp is not None:
                        lang_preds.append( lang_mapping[lang_hyp] )
                    else:
                        lang_choices = list(set(lang_mapping.values()) - set([lang_mapping[lang_ref]]))
                        lang_preds.append( random.sample(lang_choices, 1)[0] )

                    lang_refs.append( lang_mapping[lang_ref] )

    if args.output_for_submssion:
        print("You can check the output for the challenge submission in {args.output_for_submission}!")
    else:
        # -- computing WER
        lang_acc = accuracy_score(lang_refs, lang_preds)
        wer, cer, ci_wer, ci_cer = compute_bootstrap_wer(dst_path)
        report_wer = "%WER: " + str(wer) + " ± " + str(ci_wer); print(f"\n{report_wer}")
        report_cer = "%CER: " + str(cer) + " ± " + str(ci_cer); print(report_cer)
        report_lid = "%LID: " + str(lang_acc); print(report_lid)
        with open(dst_path.replace(".inf", ".wer"), "w", encoding="utf-8") as f:
            f.write(report_wer + "\n")
            f.write(report_cer + "\n")
            f.write(report_lid + "\n")

    # -- computing confusion matrix
    cm = confusion_matrix(lang_refs, lang_preds)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(lang_mapping.keys()))
    cm_display.plot().figure_.savefig(dst_path.replace(".inf", ".png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Audio-Visual Speech Recognition System.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--training-dataset", default="", type=str, help="Path to where the training dataset split is")
    parser.add_argument("--validation-dataset", default="", type=str, help="Path to where the validation dataset split is")
    parser.add_argument("--test-dataset", default="", type=str, help="Path to where the test dataset split is")
    parser.add_argument("--filter-spkr-ids", nargs='+', default=["all-spkrs"], type=str, help="Choose the speaker's data you want to use")
    parser.add_argument("--filter-by-language", nargs='+', default=["all-langs"], type=str, help="Choose the speaker's data you want to use based on the language ('es', 'eu', or 'bi')")

    parser.add_argument("--mode", default="both", type=str, help="Choose: 'training', 'inference' or 'both'")
    parser.add_argument("--mask", default="none", type=str, help="Choose: 'audio', 'video' or 'none'")
    parser.add_argument("--snr-target", default=9999, type=int, help="A specific signal-to-noise rate when adding noise to the audio waveform.")
    parser.add_argument("--noise", default="./src/noise/babble_noise.wav", type=str, help="Path to .wav file of noise")

    parser.add_argument("--config-file", required=True, type=str, help="Path to a config file that specifies the AVSR model architecture")
    parser.add_argument("--load-checkpoint", default="", type=str, help="Path to load a pretrained AVSR model")

    parser.add_argument("--load-modules", nargs='+', default=["entire-e2e"], type=str, help="Choose which parts of the model you want to load: 'entire-e2e', 'frontend', 'encoder' or 'decoder'")
    parser.add_argument("--freeze-modules", nargs='+', default=["no-frozen"], type=str, help="Choose which parts of the model you want to freeze: 'no-frozen', 'frontend', 'encoder' or 'decoder'")

    parser.add_argument("--yaml-overrides", metavar="CONF:KEY:VALUE", nargs='*', help="Set a number of conf-key-value pairs for modifying the yaml config file on the fly.")

    parser.add_argument("--output-dir", required=True, type=str, help="Path to save the fine-tuned model and its inference hypothesis")
    parser.add_argument("--output-name", required=True, type=str, help="Name of the file where the hypothesis and results will be write down.")
    parser.add_argument("--output-for-submission", default='', type=str, help='Specified output path if you want the expected output for the challenge submission')

    args = parser.parse_args()

    # -- configuration architecture details
    config_file = Path(args.config_file)
    with config_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    override_yaml(config, args.yaml_overrides)
    config = argparse.Namespace(**config)

    # -- security checks
    security_checks(config)

    # -- building tokenizer and converter
    tokenizer, converter = get_tokenizer_converter(config)

    # -- audio preprocessing
    train_audio_transforms = Compose([
        SpeedRate(sample_rate=16000),
    ])
    eval_audio_transforms = Compose([
        AddNoise(noise_path=args.noise, sample_rate=16000, snr_target=args.snr_target),
    ])

    # -- training
    if args.mode in ["training", "both"]:

        # -- -- building ASR end-to-end system
        e2e = ASRTask.build_model(config).to(
            dtype=getattr(torch, config.dtype),
            device=config.device,
        )
        print(model_summary(e2e))

        # -- -- loading the AVSR end-to-end system from a checkpoint
        load_e2e(e2e, args.load_modules, args.load_checkpoint, config)

        # -- -- freezing modules of the AVSR end-to-end system
        freeze_e2e(e2e, args.freeze_modules, config)

        # -- -- creating dataloaders
        train_loader = get_dataloader(config, dataset_path=args.training_dataset, audio_transforms=train_audio_transforms, tokenizer=tokenizer, converter=converter, filter_spkr_ids=args.filter_spkr_ids, filter_by_language=args.filter_by_language, is_training=True)
        val_loader = get_dataloader(config, dataset_path=args.validation_dataset, audio_transforms=eval_audio_transforms, tokenizer=tokenizer, converter=converter, filter_spkr_ids=args.filter_spkr_ids, filter_by_language=args.filter_by_language, is_training=False)
        test_loader = get_dataloader(config, dataset_path=args.test_dataset, audio_transforms=eval_audio_transforms, tokenizer=tokenizer, converter=converter, filter_spkr_ids=args.filter_spkr_ids, filter_by_language=args.filter_by_language, is_training=False)

        # -- -- optimizer and scheduler
        optimizer, scheduler = set_optimizer(config, e2e, train_loader)

        # -- -- training process

        # e = 0
        # spe = 20973
        # for e in range(e*spe):
        #     optimizer.step()

        val_stats = []
        print("\nTRAINING PHASE\n")
        scaler = GradScaler if config.training_settings['use_amp'] else None
        for epoch in range(1, config.training_settings['epochs']+1):
            train_loss = training(e2e, train_loader, optimizer, scheduler, config.training_settings['accum_grad'], scaler)
            val_loss, val_cer = validation(e2e, val_loader)

            print(f"Epoch {epoch}: TRAIN LOSS={train_loss} || VAL LOSS={val_loss} | VAL CER={val_cer}%")
            dst_check_path = save_model(args.output_dir, e2e, str(epoch).zfill(3))
            val_stats.append( (dst_check_path, val_cer) )

        # -- -- computing average model
        save_val_stats(args.output_dir, val_stats)
        sorted_val_stats = sorted(val_stats, key=lambda x: x[1])
        # check_paths = [check_path for check_path, cer in sorted_val_stats[:config.training_settings['average_epochs']]]
        check_paths = [check_path for check_path, val_cer in val_stats[-config.training_settings['average_epochs']:]]
        average_model(e2e, check_paths)
        save_model(args.output_dir, e2e, "average")

    # -- inference
    if args.mode in ["inference", "both"]:
        print("\nINFERENCE PHASE\n")

        # -- -- building speech-to-text recoginiser
        speech2text = build_speech2text(args, config)

        # -- -- creating validation & test dataloaders
        eval_loader = get_dataloader(config, dataset_path=args.test_dataset, audio_transforms=eval_audio_transforms, tokenizer=tokenizer, converter=converter, filter_spkr_ids=args.filter_spkr_ids, filter_by_language=args.filter_by_language, is_training=False)
        inference(args.output_dir, speech2text, eval_loader, args.output_name)

