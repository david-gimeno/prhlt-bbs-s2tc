#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import torch
from src.utils import *

import os
import yaml
import argparse
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from colorama import Fore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to study the influence of each branch in each layer of the Branchformer Encoder.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config-file", required=True, type=str, help="Path to a config file that specifies the VSR model architecture.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of training epochs.")
    parser.add_argument("--average-epochs", default=10, type=int, help="Number of epochs which will be considered to compute the average model.")
    parser.add_argument("--load-checkpoint", required=True, type=str, help="Choose any model checkpoint, so it is going to be replaced by the specified average")
    parser.add_argument("--output-dir", required=True, type=str, help="Path where the average model will be store. It should be match with the output directory of the experiments")
    parser.add_argument("--output-name", required=True, type=str, help="Name of the output model checkpoint")

    args = parser.parse_args()

    # -- configuration architecture details
    model_config_file = Path(args.config_file)
    with model_config_file.open("r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
    model_config = argparse.Namespace(**model_config)

    # -- building end-to-end speech recogniser
    speech2text = build_speech2text(args, model_config)

    # -- finding model checkpoints to average
    checkpoint_paths = []
    model_checks_dir = os.path.join(args.output_dir, 'models')
    for i in range(args.epochs, args.epochs-args.average_epochs, -1):
        checkpoint_path = os.path.join(model_checks_dir, f'model_{str(i).zfill(3)}.pth')
        checkpoint_paths.append( checkpoint_path )
    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(i, checkpoint_path)
    average_model(speech2text.asr_model, checkpoint_paths)
    save_model(args.output_dir, speech2text.asr_model, args.output_name)
