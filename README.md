# [Computational Semantics for Natural Language Processing](https://github.com/codingFerryman/computational_semantics_for_NLP)
This is our project of Computational Semantics for Natural Language Processing course in ETH Zurich.

In this project, our goal is to extract socio-political events from raw text. We use manually annotated data, namely the [CASE dataset](https://emw.ku.edu.tr/case-2021/), which has been used to organize a shared task and a workshop event at ACL-IJCNLP 2021. 

We have four subtasks in total that we want to work on 
(we only tackle the English version of the subtask -- the dataset offers the same task across different langauges and provides another two tasks which we did not conisder because this might go beyond the scope of this project). 
- Document classification (Does a news article contain information about a past or ongoing event?)
- Sentence classification (Does a sentence contain information about a past or ongoing event?)
- Event sentence coreference identification (Which event sentences (subtask 2) are about the same event?)
- Event extraction (What is the event trigger and its arguments?)

## Contents
- [Setup](#Setup)
- [Repo structure](#repo-structure)
- [Execution](#execution)
- [Developers](#developers)
- [Acknowledgements](#acknowledgements)

## Setup
We would recommend using venv (after loading a python version linked to GPUs if execute on ETHZ Leonhard cluster)

Load a python (cluster only)
```bash
module load gcc/6.3.0 python_gpu/3.8.5
```

Create a venv virtual environment and install required packages
```bash
python -m venv ./venv
source ./venv/bin/activate
python -m pip install -r requirements.txt
```

Generally we need 1 GPU with 10GB+ VRAM and 25GB+ RAM to execute our program. 
The command examples for running on the cluster are given in the code.

## Repo structure

├── [data](./data/) Here is the dataset

├── [models](./models/) Here the training of each model are stored to a specific folder. 

├── [report](./report/) Here are the reports we submitted in the course

├── [scripts](./scripts/) Here are the scripts for executing subtask 1, 2 and 4

├── [src](./src) Here is all the source code of the group's solution

│   ├── [subtask 1 and 2](./src/subtask_1_2/) Here is our solution for subtask 1 and 2

│   ├── [subtask 3](./src/subtask_3/)  Here are our explorations for subtask 3

│   ├── [subtask 4](./src/subtask_4/) Here is our solution for subtask 4

│   ├── [conlleval.py](./src/conlleval.py/) Here is the code for evaluation

## Developers

- Dominik Stammbach
- He Liu
- Didem Durukan

## Acknowledgements

- ETHZ for providing the leonhard cluster nodes to us
- Huggingface for their transformers library models
