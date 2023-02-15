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

[`Data Cartograph`](data%20cartography/) and [`model testing`](model_testing/) have their own README files for more specific further information about the code that is involved in these processes.

***A `Requirements.txt` file can be found within this folder for the required Python Packages***.