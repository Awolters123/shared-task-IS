# 1. Model testing - Baselines, Data, and Main Models

## 1.1. Table of Contents
- [1. Model testing - Baselines, Data, and Main Models](#1-model-testing---baselines-data-and-main-models)
  - [1.1. Table of Contents](#11-table-of-contents)
  - [1.2. General information](#12-general-information)
  - [1.3. Data](#13-data)
  - [1.4. Baselines](#14-baselines)
  - [1.5. Main models](#15-main-models)
    - [1.5.1. Task A](#151-task-a)
    - [1.5.2. Task B](#152-task-b)
  - [1.6. Running the models](#16-running-the-models)


## 1.2. General information
This folder contains the code for the HateBERT and DeBERTa-v3-base models that were used for the project.
All code makes use of Huggingface models and the Transformers package to import these models.

***A requirements.txt file is provided within this folder for the purpose of running all models found within this folder.***

## 1.3. Data
All code that was used for the testing can be acquired online.

The following data was used for training and testing:
- `The EDOS dataset`, taken from the shared task.
- `The EXIST2021 dataset` was received by email by asking for it through the required form.
  - For our purposes, we `merged the train and test data` for this dataset and we only kept the `English examples`.
  - For Task B, we made use of the `data/TaskB_cosine_similarity.ipynb` to map the items to their corresponding variants in the EDOS dataset using `cosine similarity`.

  The EDOS data was split into a train and dev set on runtime using an `80/20 split` with a random seed of `1234`.

  Whenever EXIST2021 data was applied, it was applied on top of this 80% train data.

## 1.4. Baselines
The folder `baselines` contains two python notebooks:
- `best_model.ipynb` which contains the best baseline models that were used as a starting point for the tasks.
- `Shared_task_IS.ipynb` which contains the different baseline models that were tested to find the best starting point.

## 1.5. Main models
The main models consist of DeBERTa-v3-base and HateBERT, which we finetuned and tested using the code found within this root folder.

### 1.5.1. Task A
For Task A of EDOS, We tested three conditions for our tasks:
- The use of only EDOS data.
- The use of EDOS data extended with EXIST2021 data through concatenation.
- The use of EDOS data extended with EXIST2021 data through shuffling.

These conditions were split over two versions of the same code:

- `models.py` - Which was used mostly in conjunction with DeBERTa and the HPC cluster of the University of Groningen - Peregrine
- `...` - ...

These models have the same origin but differ in their execution, related to the fact that both variants were run by two different people on two different systems - agility and easy change were paramount.


### 1.5.2. Task B
For Task B of EDOS, We tested three conditions for our tasks:
- The use of only EDOS data.
- The use of EDOS data extended with EXIST2021 data through concatenation (the EXIST data was filtered using cosine similarity).
- The use of EDOS data extended with EXIST2021 data through shuffling (the EXIST data was filtered using cosine similarity).

These conditions were split over two versions of the same code:

- `models.py` - Which was used mostly in conjunction with DeBERTa and the HPC cluster of the University of Groningen - Peregrine
- `hatebert_taskA_AW.py` & `hatebert_taskB_AW.py`- These were used for HateBERT to run the models.

These models have the same origin but differ in their execution, related to the fact that both variants were run by two different people on two different systems - agility and easy change were paramount.


## 1.6. Running the models
The files found here can be split into two variants:
- *.ipynb
- *.py

The *.ipynb files can simply be run by opening them using an appropriate editor (for example, through the Jupyter package).

For the *.py files, the appropriate course of action is as follows:
```Bash
python $PROGRAM.py -h
```

Which will show you the large number of options that can be used to finetune and choose the different models for training and testing.
