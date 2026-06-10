## Download data and perform feature engineering
Due to space constraints, we have created new folders named "train" and "test" under the "data" folder to separately store the datasets provided by the official source.

Since data engineering can take up a significant amount of time, we have provided saved files of completed feature engineering checkpoints.

The files includes table data after feature engineering (train and test), and the download link is:

Link: https://pan.baidu.com/s/1Nub-klu30DgIILCZ3N9mcg?pwd=qey1

Extraction code: qey1

## Run code

This script trains a LightGBM model using 5-fold cross-validation on pre-processed feature data and generates submission files.

## Usage
```bash
python code.py
```

## Output: Submission file ./sub/sub_v1.csv and trained models in ./model_save/
