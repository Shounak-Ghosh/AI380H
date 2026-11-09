# MIMIC NLP Tutorial
There are two notebooks associated with this assignment along with a `data` folder (not committed per [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) usage guidelines). The dataset preparation notebook (`01_prepare_sepsis_dataset.ipynb`) is meant to run locally while the NLP notebook (`02_sepsis_nlp_analysis.ipynb`)runs on Google Colab. 

First, install the `uv` package manager: 
```
# If `uv` is not installed yet
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Restart Terminal or run:
```
source $HOME/.local/bin/env
```
Check: 
```
uv --version
```
Have `uv` install Python 3.11:
```
uv python install 3.11
```
Create a virtual environment specifically using 3.11: 
```
uv venv --python 3.11 .venv
```
Activate it: 
```
source .venv/bin/activate
```
Verify Python version `3.11.x`:
```
python --version
```
Install only what the dataset preparation notebook needs: 
```
uv pip install \
    jupyterlab \
    ipykernel \
    pandas \
    tqdm
```
Register this environment as a Jupyter kernel:
```
python -m ipykernel install \
    --user \
    --name mimic-nlp \
    --display-name "Python 3.11 - MIMIC NLP"
```
Now start Jupyter: 
```
jupyter lab
```
and choose: Kernel → Change Kernel → Python 3.11 - MIMIC NLP

Create the following directories: 
```
mkdir -p ~/MIMIC_NLP/data/raw
mkdir -p ~/MIMIC_NLP/data/processed
```
So that we have the following folder structure: 
```
MIMIC_NLP/
├── .venv/
├── data/
│   ├── raw/
│   │   ├── DIAGNOSES_ICD.csv.gz
│   │   └── NOTEEVENTS.csv.gz
│   └── processed/
│       └── sepsis_discharge_notes.csv.gz
└── notebooks/
    ├── 01_prepare_sepsis_dataset.ipynb
    └── 02_sepsis_nlp_analysis.ipynb
```
Both `NOTEEVENTS.csv.gz` and `DIAGNOSES_ICD.csv.gz` can be downloaded as part of the MIMIC-III dateset (requires credentialed access).`NOTEEVENTS.csv.gz` is a large file (1.1GB) and may take some time to not only download, but also upload to a cloud environment. That is why we are processing it locally to produce `sepsis_discharge_notes.csv.gz` (2.24MB). 

The NLP notebook has been executed on Google Colab with a runtime version of `2025.07` instead of the `Latest (recommended)` for compatability with the NLP/scientific packages required. The runtime can be changed by selecting "Python 3" in the bottom right corner of the Colab UI and then selected "Change runtime type". 

The following installs may work for running the NLP notebook locally but have not been tested: 
```
uv pip install \
    "spacy==3.7.5" \
    "scispacy==0.5.5" \
    "medspacy==1.3.1" \
    gensim \
    scikit-learn \
    matplotlib
```
Regular spaCy model:
```
python -m spacy download en_core_web_md
```
SciSpaCy model:
```
uv pip install \
  "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
```
Biomedical NER model: 
```
uv pip install \
  "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
```