# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a template for solving ASR task with PyTorch. This template branch is a part of the [HSE DLA course](https://github.com/markovka17/dla) ASR homework. Some parts of the code are missing (or do not follow the most optimal design choices...) and students are required to fill these parts themselves (as well as writing their own models, etc.).

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw1_asr).

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

### Training

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

### Inference

To run inference on a dataset and save predictions:

```bash
# On the default evaluation dataset (LibriSpeech test-clean)
python3 inference.py -cn=inference \
  inferencer.from_pretrained=path/to/checkpoint.pth

# On a custom directory dataset
python3 inference.py -cn=inference \
  datasets=custom_dir \
  custom_audio_dir=path/to/audio/directory \
  custom_transcription_dir=path/to/transcriptions/directory \
  inferencer.from_pretrained=path/to/checkpoint.pth \
  inferencer.save_path=output_dir_name
```

**Custom Directory Format:**
Your custom dataset should be organized as follows:

```
my_dataset/
├── audio/
│   ├── utterance_001.wav    (or .mp3, .flac, .m4a)
│   ├── utterance_002.wav
│   └── ...
└── transcriptions/  (optional, only needed for metric calculation)
    ├── utterance_001.txt
    ├── utterance_002.txt
    └── ...
```

Each transcription file should contain a single line with the ground truth text.

## Evaluation and Metrics

To calculate WER (Word Error Rate) and CER (Character Error Rate) metrics:

```bash
python3 calc_metrics.py \
  --ground_truth_dir path/to/ground_truth_transcriptions \
  --predicted_dir path/to/predicted_transcriptions \
  --verbose
```

**Example with inference output:**

```bash
# First run inference and extract predictions
python3 inference.py -cn=inference \
  inferencer.from_pretrained=saved/testing/model_best.pth \
  inferencer.save_path=demo_results

# Then calculate metrics
python3 calc_metrics.py \
  --ground_truth_dir data/saved/demo_results/test \
  --predicted_dir data/saved/demo_results/predictions \
  --verbose
```

The script expects `.txt` files in both directories with matching IDs (utterance IDs without the extension).

## Demo Notebook

A comprehensive Colab-ready demo notebook is provided: `demo_asr.ipynb`.

This notebook demonstrates the complete workflow:

1. **Installation**: Clone the repository and install dependencies
2. **Setup**: Download pretrained model checkpoints
3. **Inference Demo**: Run inference on the LibriSpeech test-clean evaluation dataset
4. **Metrics Calculation**: Calculate WER and CER metrics on inference results
5. **Custom Dataset**: Run inference on your own custom dataset and evaluate it

The notebook includes:
- Detailed comments explaining each step
- Instructions for using Google Drive datasets (Colab-ready)
- Example usage of both `inference.py` and `calc_metrics.py` scripts
- Troubleshooting tips

**To run the demo:**
- In Jupyter/JupyterLab: `jupyter notebook demo_asr.ipynb`
- In Google Colab: Upload the notebook or open directly from GitHub

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
