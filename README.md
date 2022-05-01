# Disentangled SEQuential AutoEncoders

This project aims to provide a framework for unsupervised disentanglement of sequential data,
and is under active development.

## News
- (22/5/1) Publish code for IJCAI-22. Provide the DMelodies dataset and evaluation of the global latent space using LDA.

## Usage
`source env.sh` to include necessary paths, and `./scripts/run_{model}.sh` with `{model}` replaced with one of the four models `dsae`, `freeze`, `tsdsae_woReg` and `tsdsae`.
Possible configurations can be found under `conf/`.

## TODO
- [ ] Upload benchmark results from running `./scripts/benchmark/run_*.sh`
- [ ] Upload audio samples
- [ ] Elaborate `README.md`
- [ ] Publish evaluation code for the local latent using Crepe