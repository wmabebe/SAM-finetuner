<!-- wget https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/training.tif
wget https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/training_groundtruth.tif
wget https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/testing.tif
wget https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/testing_groundtruth.tif
mkdir datasets/mitochondria
mv *.tif ./datasets/mitochondria

pip install -r requirements.txt

sbatch --gres=gpu:1 --wrap="python3 finetune.py" -->

# Finetuning SAM on the Mitochondria dataset

## Description
This repository contains commands to download the Electron Microscopy dataset for training and testing purposes. The dataset include mitochondria images, along with corresponding ground truth images. The commands provided facilitate the download and organization of the dataset.

### Download Data
- `wget https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/training.tif`
- `wget https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/training_groundtruth.tif`
- `wget https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/testing.tif`
- `wget https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/testing_groundtruth.tif`

## Move data to dataset
To maintain a structured dataset, the following commands can be used:


`mkdir datasets/mitochondria`
`mv *.tif ./datasets/mitochondria`

### Install dependancies

`pip install -r requirements.txt`

### Run finetuning code

`sbatch --gres=gpu:1 --wrap="python3 finetune.py"`
