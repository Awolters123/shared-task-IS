# SemEval 2023 - Task 10 - EDOS - Team Daunting

This repository contains the code for `Team Daunting's` submission for [SemEval 2023 - Task 10 - Explainable Detection of Online Sexism (EDOS)](https://codalab.lisn.upsaclay.fr/competitions/7124).
We took part in Tasks A and B for this shared task as part of a course `Shared Task Information Science` at the [University of Groningen](https://www.rug.nl/).

The project is split up into three parts, some of which contain certain subparts.
The systems are as follows:
- `data cartography` is where our code pertaining the data cartography method for data analysis and filtering is stored.
- `jsd` is where the code for data analysis pertaining the Jensen–Shannon divergence is stored.
- `model testing` is where the code for our different models is stored and the supplementary code can be found for the data.
  - `main models` The code for the main models (HateBERT and DeBERTa-v3-base) can be found in the root of this folder.
  - `Baselines` The code for the baselines can be found in this subfolder.

[`Data Cartograph`](data%20cartography/README.md) and [`model testing`](model_testing/README.md) have their own README files for more specific further information about the code that is involved in these processes.

`Requirements.txt` files can be found for each subpart within their respective folder.

<!-- It contains two Jupyter notebook files containing all the work we (group 3) did for the baseline.
The best model can be found in the *best_model.ipynb*, and the general decision process can be found in *Shared_task_IS.ipynb*.

## Working with the Jupyter notebook files

### Opening the files
To make use of the Jupyter notebook file, it is recommended that a proper Jupyter notebook viewer is used, options include:
- Jupyter notebook
- Google colab
- vscode

### Checking the results
For both aforementioned files (see [General Information](##-General-information)), outcomes have already been generated as part of the Jupyter notbooks. These results were used by the team as the final output.

### Running The Code
In case one wants to regenerate the outcomes of the Jupyter notebook (or make changes), one can simply *run all* code. More information about these features can be found in the guides for the specific tool that is being used.

After this, the results should be the same as before (unless changes have been made). -->