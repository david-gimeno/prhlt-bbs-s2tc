<h1 align="center"><span style="font-weight:normal">PRHTL @ Bilingual Basque-Spanish<br>Speech to Text Challenge</h1>  

  <div align="center">
    
[David Gimeno-Gómez](https://scholar.google.es/citations?user=DVRSla8AAAAJ&hl=en), [Carlos-D. Martínez-Hinarejos](https://scholar.google.es/citations?user=M_EmUoIAAAAJ&hl=en)
</div>

<div align="center">
  
[📘 Introduction](#intro) |
[🛠️ Preparation](#preparation) |
[💪 Training](#training) |
[🦒 Model Zoo](#modelzoo) |
[🔮 Inference](#inference) |
[📊 Results](#results) |
[📖 Citation](#citation) |
[📝 License](#license)
</div>

## <a name="intro"></a> 📘 Introduction

Bilingual Basque-Spanish Automatic Speech Recognition. Official source code for the participation of the PRHLT team at the Bilingual Basque-Spanish Speech to Text Challenge within the framework of the Albayzín Evaluations 2024. 

## <a name="preparation"></a> 🛠️ Preparation

- Prepare the **conda environment** to run the experiments:

```
conda create -n bbs-s2tc python=3.8
conda activate bbs-s2tc
pip install -r requirements.txt
```

- By inspecting our [dataset](https://github.com/david-gimeno/prhlt-bbs-s2tc/blob/main/src/datasets/asr_dataset.py), you can get an idea about how our data CSV splits are organized. Specifically, here you can imagine how they look:

```
path,language,speaker_id,PRR,length,sentence
../basque_parliament_1/audio/12-079_20220217_01/12-079_20220217_01_196.24_201.88.mp3,eu,416,100.0,5.64,egun on guztioi osoko bilkura hasiko dugu
```

## <a name="training"></a> 💪 Training

The following command represents the training and inference of a non-autoregressive Conformer-based Mask-CTC model with Hierarchical Language Identification on the official traninig set of the challenge:

```
python asr_main.py \
  --training-dataset ./splits/bbs-s2tc/all_clean_data.csv \
  --validation-dataset ./splits/bbs-s2tc/dev.csv \
  --test-dataset ./splits/bbs-s2tc/test.csv \
  --config-file ./configs/asr/conformer+hierlidutt_maskctc.yaml \
  --mode both \
  --output-dir ./exps/bbs-s2tc/ \
  --output-name test-liprtve-si-finetuned-from-english \
  --yaml-overrides training_settins:batch_size:16
```

## <a name="modelzoo"></a> 🦒 Model Zoo

Our best-performing model checkpoint for the challenge is publicly available in our official Zenodo repository. Please, click [here](https://zenodo.org/records/12772215) to download the checkpoint along with their corresponding configuration file. By following the instructions indicated above for both training and inference, you will be able to evaluate our models and also fine-tune them to your dataset of interest.

## <a name="inference"></a> 🔮 Inference

You can check the performance achieved by already estimated models by running the following command:

```
python asr_main.py \
  --test-dataset ./splits/test/speaker-independent/liprtve.csv \
  --config-file ./configs/asr/conformer+hierlidutt_maskctc.yaml \
  --load-checkpoint ./exps/bbs-s2tc/models/model_average.pth \
  --mode inference \
  --filter-by-language all-langs \
  --output-dir ./exps/bbs-s2tc/ \
  --output-name test_all-langs \
```

## <a name="results"></a> 📊 Results

Detailed discussion on these results both for Spanish and Basque languages can be found in our [paper]()!

## <a name="citation"></a> 📖 Citation

The paper is currently under review for the [IberSpeech 2024](https://iberspeech.tech/) conference.

```
@inproceedings{gimeno2025tailored,
  author={Gimeno-G{\'o}mez, David and Carlos-D. Martínez-Hinarejos},
  title={{The PRHLT Speech Recognition System the Bilingual Basque-Spanish Speech to Text Albayzín Challenge}},
  booktitle={},
  volume={},
  pages={}
  year={},
}
```

## <a name="license"></a> 📝 License

This work is protected by [CC BY-NC-ND 4.0 License (Non-Commercial & No Derivatives)](LICENSE)
