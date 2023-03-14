# EntropyBasedCLDMetrics
Github repository accompanying the paper "Measuring language development from child-centered recordings".

- `checkpoints/` contains all the trained model (`.pt` for pytorch models and `.arpa` for [KenLM](https://github.com/kpu/kenlm) language models)
- `data/` contains all the raw data used in the different experiments.
- `configs/` contains all the necessary options to train/test the models.
- `results/` contains all the csvs storing the results.
- `src/` contains all the source code.

For reproducing all the experiments, you will need to:

1) Prepare the data to train and test the models
2) Run the training
3) Run the testing
4) Reproduce the anlaysis (plots and statistical analysis of the results)

# Python environment

`TODO: recreate a clean version conda env`

# Data preparation

## Prepare the librispeech data for phones _n_-gram language model

```shell
python src/librispeech_for_ngram_lm.py -o data/training
```

## _n_-gram language model training

We need to first train the _n_-gram language model in order to prepare the data for the other experiments.

For training the _n_-gram language model, you will need to install [KenLM](https://github.com/kpu/kenlm). Once installed in the current directory, you can run the training:


`TODO`

# Run the training

`TODO`

# Run the testing
`TODO`

# Analysis

`TODO`
