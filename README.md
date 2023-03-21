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

# Working environment

Clone this github repos and move to it:

```bash
git clone https://github.com/yaya-sy/EntropyBasedCLDMetrics.git
cd EntropyBasedCLDMetrics
```

Create the Python environment:

```shell
conda env create -f environment.yml
```

and activate it:

```shell
conda activate ent_cldm
```

# Data preparation

## Prepare the librispeech data for phones _n_-gram language model

The phonemizer requires the espeak backend, it can be installed with: `apt-get install espeak-ng`

```shell
python src/librispeech_for_ngram_lm.py -i [LIBRISPEECH_TRAIN-CLEAN-360_FOLDER] -o data/ngram_lm/
```

## _n_-gram language model training

We need to first train the _n_-gram language model in order to prepare the data for the other experiments.

For training the _n_-gram language model, you will need to install [KenLM](https://github.com/kpu/kenlm):

```bash
conda install -c anaconda cmake
git clone https://github.com/kpu/kenlm.git
cd kenlm
python setup.py develop
mkdir -p build
cd build
cmake ..
make -j 4
```

Once installed in the current directory, you can run the training:

```shell
kenlm/build/bin/lmplz --discount_fallback -o 5 < data/ngram_lm/librispeech.phonemized > checkpoints/librispeech_360.arpa
```

## Pepare the data for Thomas regression model (Experiment 2A)

### Prepare utterances

You will need to install the Thomas corpus as imported by William N. Havard: https://gin.g-node.org/LAAC-LSCP/thomas

Once installed, you can run this command to extract utterances, their cleaned version and the timemarks:

```shell
python src/create_thomas_corpus.py -c [CHILDES_PATH_THOMAS] -o data/Thomas
```

Where `<CHILDES_PATH_THOMAS>` is the path to the installed Thomas data.
 
This will create a hierarchical folder in the folder `data`:

```
Thomas\
    cleaned\
        child_1\
            age_1\
                filename.txt
                months.txt
                Mother.cleaned
                Target_Child.cleaned
            age_2\
            ...
            age_n
        child_2
        ...
        child_n
    orthographic\
        child_1\
            age_1\
                filename.txt
                months.txt
                Mother.orthographic
                Target_Child.orthographic
            age_2\
            ...
            age_n
        child_2\
        ...
        child_n\
    timemarks\
        child_1\
            age_1\
                filename.txt
                months.txt
                Mother.timemarks
                Target_Child.timemarks
            age_2\
            ...
            age_n
        child_2\
        ...
        child_n\
```

Where `orthographic` contains the raw annotations without cleaning. The `cleaned` contains the cleaned version of the annotations. And `timemarks` contains the onsets and offsets of each utterance in the audio. All of these are aligned, meaning that the _i<sup>th</sup>_ line of each file corresponds to the _i<sup>th</sup>_ line of the other files.

The `filename.txt` files contains the raw filename and `months.txt` contains the age in months.

### Prepare inputs for the regression model

```shell
> python src/prepare_childes_corpus.py -i data/Thomas/
> python src/prepare_input_files.py -c data/Thomas/ -a [AUDIO_FOLDER] -m checkpoints/librispeech_360.arpa
```

Where `<AUDIO_FOLDER>` is the path to the audio folder. In the case of Thomas childes data, the audio folder is `recordings/raw/`

## Pepare the data for Librispeech regression model (Experiment 2B)

Create the inputs for the regression model:

```bash
> python src/prepare_librispeech_corpus.py -i [LIBRISPEECH_TRAIN-CLEAN-100_FOLDER] -o data/Librispeech/model_inputs
> python src/prepare_input_files.py -c data/Librispeech/ -a [LIBRISPEECH_TRAIN-CLEAN-100_FOLDER] -m checkpoints/librispeech_360.arpa
```

where `<LIBRISPEECH_TRAIN-CLEAN-100_FOLDER>` is the path to the folder containing the librispeech train-clean-100
## Prepare the Providence test data

### Prepare utterances

Create the hierarchical data organization for Providence:

```bash
python src/create_providence_corpus_new.py -i [PREPARED_CSV] -c [CHILDES_PATH_PROVIDENCE] -o data/Providence/
```

### Prepare inputs for the regression model

Create the inputs for the model:

```bash
> python src/prepare_childes_corpus.py -i data/Providence/
> python src/prepare_input_files.py -c data/Providence/ -a [AUDIO_FOLDER] -m checkpoints/librispeech_360.arpa
```

Where `<AUDIO_FOLDER>` is the path to the audio folder. In the case of Providence childes data, the audio folder is `recordings/raw/`

# Run the trainings

## Experiment 1A: Training on Librispeech-360

The model is already trained during the data prepration and is save on `checkpoints/librispeech_360.arpa`.
So we will not retrain it again.

## Experiment 2A: Training on Thomas

Run the regression model training on Thomas (Experiment 2A):

```bash
python src/train.py -c configs/thomas.yaml
```

## Experiment 2B: Training on Librispeech

Run the regression model training on librispeech-100 (Experiment 2B):

```bash
python src/train.py -c configs/librispeech.yaml
```

# Run the testing

## Experiment 1A: Text entropies

```bash
python src/compute_entropies_ngram_lm.py
```

This will create a csv file named `Librispeech_360h.csv` in the folder `results`.

## Experiment 2A: Entropy predictions using Thomas data

```bash
python src/compute_entropies_whisper.py -c configs/test.yaml -m checkpoints/Thomas_30h_Librispeech360_en.pt
```

This will create a csv file named `Thomas_30h_Librispeech360_en.csv` in the folder `results`.

## Experiment 2B: Entropy predictions using Librispeech data

```bash
python src/compute_entropies_whisper.py -c configs/test.yaml -m checkpoints/Librispeech_100h_Librispeech360_en.pt
```

This will create a csv file named `Librispeech_100h_Librispeech360_en.csv` in the folder `results`.

# Analysis

## Prepare the CSVs for analysis
### Experiment 1A

```bash
python src/prepare_for_analysis.py -i results/Librispeech_360h.csv
```

This will create a csv file named `Librispeech_360h_analysis.csv` in the folder `results`.

## Experiment 2A

```bash
python src/prepare_for_analysis.py -i results/Thomas_30h_Librispeech_en.csv
```

This will create a csv file named `Thomas_30h_Librispeech_en_analysis.csv` in the folder `results`.

## Experiment 2B

```bash
python src/prepare_for_analysis.py -i results/Librispeech_100h_Librispeech360_en.csv
```

This will create a csv file named `TLibrispeech_100h_Librispeech360_en_analysis.csv` in the folder `results`.

## Plottings

You can reproduce the figures of the paper with this `.Rmd` script: `analysis/plots.Rmd`

