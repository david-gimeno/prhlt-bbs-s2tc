import os

from src.inference import ASR2Text, ASR2TextMaskCTC

def build_speech2text(args, config):
    if config.model == "espnet":
        speech2text = ASR2Text(
            asr_train_config= args.config_file,
            asr_model_file=args.load_checkpoint if args.load_checkpoint != "" else os.path.join(args.output_dir, "models/model_average.pth"),
            lm_train_config=args.lm_config_file if args.lm_config_file != "" else None,
            lm_file=args.load_lm if args.load_lm != "" else None,
            **config.inference_conf,
        )
    elif config.model == "maskctc":
        speech2text = ASR2TextMaskCTC(
            asr_train_config=args.config_file,
            asr_model_file=args.load_checkpoint if args.load_checkpoint != "" else os.path.join(args.output_dir, "models/model_average.pth"),
            token_type=config.token_type,
            **config.inference_conf,
        )
    else:
        raise ValueError(f'unknown model architecture {config.model}')

    return speech2text
