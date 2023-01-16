# 1. Data Cartography \& logits generation
This folder contains the necessary steps for the generation of logits to be used with the `Dataset Cartography` package ([allenai/cartography](https://github.com/allenai/cartography)), which was used for the paper [Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics](https://aclanthology.org/2020.emnlp-main.746) at [EMNLP 2020](https://2020.emnlp.org/) (citation information can be found on the cartography GitHub page).

The current setup is contained within the `cartography_edits.ipynb` notebook file.

## 1.1. Table of Contents
- [1. Data Cartography \& logits generation](#1-data-cartography--logits-generation)
  - [1.1. Table of Contents](#11-table-of-contents)
  - [1.2. Generating logits](#12-generating-logits)
    - [1.2.1. Folder structure](#121-folder-structure)
    - [1.2.2. Running cartography\_edits.ipynb](#122-running-cartography_editsipynb)
  - [1.3. Data Cartography](#13-data-cartography)
    - [1.3.1. Requirements](#131-requirements)
    - [1.3.2. Running](#132-running)
  - [1.4. Some info about the concepts used for Data Cartography](#14-some-info-about-the-concepts-used-for-data-cartography)

## 1.2. Generating logits
### 1.2.1. Folder structure
1. Start with a clone or download of the folder from GitHub, which should come with the `notebook files`, this `README.md`, and a `requirements.txt` for Python.
2. Make sure to add a `data/` folder containing the `train_all_tasks.csv` and `EXIST2021_merged.csv`, these will be used for the generation of logits.
3. Make sure there is an empty `logits/` folder, for the logits output.
4. Prepare an empty `saved_models/` folder as well, in case you want to store the latest model for future retrieval.

This should lead you to the following folder structure\*:
```
./
├── cartography_edits.ipynb
├── data/
│   ├── EXIST2021_merged.csv
│   └── train_all_tasks.csv
├── logits/
├── README.md
├── requirements.txt
└── saved_models/
```
*\* Note. the two .csv files here consist of a merged train and test taken from the EXIST 2021 dataset and the train data for all semeval 2023 tasks.*

### 1.2.2. Running cartography_edits.ipynb
***`WARNING: for subsequent runs, make sure to copy your logits to another folder; they will be overwritten!`***

Once the folder structure is in order, you can run the Jupyter notebook locally (`cartography_edits.ipynb`).

To run the file in another way, such as through Google Colab, make sure to add the necessary code to install the needed packages, such as `transformers` and `jsonlines`.

Most of the steps are self-explanatory: follow the instructions and information comments found in the notebook to run the code.
If errors are found, check the following:
1. Are all steps outputting what is expected?
2. Do I have all required packages and are there any issues with the versions I'm using?
3. Is my device able to handle the parameters that have been set for the model?

## 1.3. Data Cartography
After retrieving the logit files for the different epochs, it will be necessary to git clone [allenai/cartography](https://github.com/allenai/cartography) to run the `cartography.selection.train_dy_filtering` script.

### 1.3.1. Requirements

First, make sure that all requirments are fulfilled, as I ran into some issues with virtual environment installs, you may or may not run into issues. These are the requirements I used that did work:

```python
transformers==4.24.0
tqdm
seaborn
pandas
matplotlib
numpy
jsonnet
tensorboardx
torch
spacy
scikit-learn
```

### 1.3.2. Running

After cloning the repo and preparing the requirements, it would be smart to have an ordered storage for all the data input and outputs of our data cartography. To this end, a folder structure has been defined, which can be found in `workfolder.zip`.

An overview of this folder can be found below:

```
workfolder/
├── cartography.sh
├── datafiles_for_cartography/
│   ├── exist/
│   │   ├── deberta/
│   │   │   └── WINOGRANDE/
│   │   └── hatebert/
│   │       └── WINOGRANDE/
│   ├── semeval/
│   │   ├── deberta/
│   │   │   └── WINOGRANDE/
│   │   └── hatebert/
│   │       └── WINOGRANDE/
│   └── shuffle_semeval-exist/
│       ├── deberta/
│       │   └── WINOGRANDE/
│       └── hatebert/
│           └── WINOGRANDE/
├── logits/
│   ├── exist/
│   │   ├── deberta/
│   │   │   └── training_dynamics/
│   │   └── hatebert/
│   │       └── training_dynamics/
│   ├── semeval/
│   │   ├── deberta/
│   │   │   └── training_dynamics/
│   │   └── hatebert/
│   │       └── training_dynamics/
│   └── shuffle_semeval-exist/
│       ├── deberta/
│       │   └── training_dynamics/
│       └── hatebert/
│           └── training_dynamics/
└── outputs/
    ├── data/
    │   ├── exist/
    │   │   ├── deberta/
    │   │   │   ├── ambiguous/
    │   │   │   ├── easy-to-learn/
    │   │   │   └── hard-to-learn/
    │   │   └── hatebert/
    │   │       ├── ambiguous/
    │   │       ├── easy-to-learn/
    │   │       └── hard-to-learn/
    │   ├── semeval/
    │   │   ├── deberta/
    │   │   │   ├── ambiguous/
    │   │   │   ├── easy-to-learn/
    │   │   │   └── hard-to-learn/
    │   │   └── hatebert/
    │   │       ├── ambiguous/
    │   │       ├── easy-to-learn/
    │   │       └── hard-to-learn/
    │   └── shuffle_semeval-exist/
    │       ├── deberta/
    │       │   ├── ambiguous/
    │       │   ├── easy-to-learn/
    │       │   └── hard-to-learn/
    │       └── hatebert/
    │           ├── ambiguous/
    │           ├── easy-to-learn/
    │           └── hard-to-learn/
    └── maps/
        ├── exist/
        ├── semeval/
        └── shuffle_semeval-exist/
```

This folder should be either linked (symlink) or copied inside of the `allenai/cartography` cloned repository folder.

With the folder structure in place, do the following:
1. Note the two input folders, `datafiles_for_cartography/` and `logits/`. The subfolders for these two folders should be filled with the following,
  - `datafiles_for_cartography/` - the `train.tsv`, `dev.tsv`, and `test.tsv` (dev and test can be the same if you only used a test set during training). Make sure they are put into the `WINOGRANDE` folder
  - `logits/` - the output logits from your trained model go here, into the `training_dynamics` folders
2. cd or move into the `root` of `allenai/cartography` (`$SOMEPLACE` stands for the folder structure above your cloned cartography folder)
```bash
cd $SOMEPLACE/cartography
```
3. Make sure the `workfolder` is inside of this folder (and, if using a virtual environment, make sure to activate it)
4. From there, run the following
```bash
sh workfolder/cartography.sh
```

The script should now run and it will generate all of the files into the `outputs/` folders.

## 1.4. Some info about the concepts used for Data Cartography
We train a model (I opted for HateBERT here because we are using that as one of our models, and it is easier to run than DeBERTa-base-v3) and during the training, the following aspects are extracted for each epoch:
> - `guid` : instance ID matching that in the original data file, for filtering,
> - `logits_epoch_$X` : logits for the training instance under epoch `$X`,
> - `gold` : index of the gold label, must match the logits array.

*Quoted from allenai/cartography*

These aspects are then used by the data cartography code to generalize which items were hard-to-learn, easy-to-learn and ambiguous:

- confidence == hard-to-learn
- variability == ambiguous
- confidence --worst == easy-to-learn







