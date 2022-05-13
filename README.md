# Bottleneck-Anchor-Disentangle: A VAE framework

BAD-VAE is a developing framework for unsupervised disentanglement of sequential data.
The current repo has been tested on NVIDIA RTX A5000 with CUDA 11.4.

Audio samples can be found in https://yjlolo.github.io/dSEQ-VAE.

## News
- (22/5/1) Publish code for the paper [*Towards Robust Unsupervised Disentanglement of Sequential Data —
A Case Study Using Music Audio*](https://arxiv.org/abs/2205.05871) accepted to [IJCAI-22](https://ijcai-22.org/).

## Additional results
We note that reproducing exactly the numbers reported in the paper is tricky due to discrepancies in both software and hardware, especially for unsupervised models.
To mitigate the issue, we provide additional results (by running `./scripts/benchmark/*/run_*.sh`) which trains and evaluates each combination of configurations with six random seeds.
The parameters include the three sizes of the local latent space, and the ADAM optimiser with or without the `amsgrad` variant.
This amounts to 288 (2 datasets * 4 models * 6 seeds * 3 latent sizes * 2 optimisers) checkpoints whose LDA F1 scores are summarised below.
The evaluation is performed using the code defined under `test_epoch_end()` in `src/models/base.py`.

- DMelodies

<img src="misc/dmel_lda_amsgrad=F.png" width="450" height="160">
<img src="misc/dmel_lda_amsgrad=T.png" width="450" height="140">

- URMP

<img src="misc/urmp_lda_amsgrad=F.png" width="450" height="160">
<img src="misc/urmp_lda_amsgrad=T.png" width="450" height="140">

For each dataset, the top and bottom panels correspond to `amsgrad` being `False` and `True`, respectively, and the variance is due to the six random seeds.
It shows that the proposed TS-DSAE performs the best in terms of disentanglement, and is robust against the configurations.

## Installation

Clone the project and the submodule:

```
git clone https://github.com/yjlolo/dSEQ-VAE.git --recurse-submodules
```

`cd dSEQ-VAE` and install the dependencies (virtual environment recommended):

```
pip install -r requirements.txt
```

`source env.sh` to include the necessary paths.

## Data

- **DMelodies**:
Download the audio files from [Zenodo](https://zenodo.org/record/6540603), unzip and put `wav_datasets` under `data/dmelodies_dataset/`.

- **URMP (TODO)**:
Follow the script provided from [Hayes et al.](https://github.com/ben-hayes/neural-waveshaping-synthesis#data) to process the data.
Set the `--output-directory` as `data/{dir_name}` when `create_dataset.py`, with `{dir_name}` replaced.
In this repo, modify both `datasets.train.path_to_data` and `datasets.val.path_to_data` under `conf/data/urmp.yaml` as the `dir_name`.

## Usage
To train a model, run

```
./scripts/run_{model}.sh
```

with `{model}` replaced with one of the four model options `dsae`, `freeze`, `tsdsae_woReg` and `tsdsae`.
See `conf/` for full configurations.

For debugging purpose, run the following instead for a sanity check.

```
python train.py train.pl_trainer.fast_dev_run=True
```

> **WARNING**: The scripts for generating the additional results above `./scripts/benchmark/run_*.sh` are computationally expensive, and execute multiple experiment sets sequentially.

## TODO
- [x] Upload benchmark results from running `./scripts/benchmark/run_*.sh`
- [x] Upload audio samples for melody swapping
- [ ] Test and streamline URMP process
- [x] Elaborate `README.md`
- [ ] Publish evaluation code for the local latent using Crepe
