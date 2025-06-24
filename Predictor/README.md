# Software implementation
The software is located in the Predictor folder of the repository and supports two different ways of usage:
* run the prediction through the terminal with a python script
* run predictions in a IPython Notebook interactive environment

## 1. Run from terminal
In order to run the software from terminal the Predictor folder must be downloaded and placed in the current working directory. In that same directory the input tsv files must be present.
This software is supposed to be executed after the calc_feature.py and cal_3di.py scripts are run on the input pdb file, as it takes in input the tsv files created by the first two scripts.
Additionally, the input file can either be a signle tsv file, a folder with multiple tsv files, a zip archive with multiple tsv files.

The expected directory structure is:
working_dir/
├── your_input.tsv
├── Predictor/
│   ├── calc_additional_features.py
│   ├── onehot_encoder.pkl
│   └── models/
│       └── xgboost/
│           ├── HBOND.joblib
│           ├── IONIC.joblib
│           └── ... (other model files)

The python script that will automatically run the predictions on the input is run_prediction.py
An example of usage is the following:




## 2. Predict in notebook
In the Predictor folder, the Predictor.ipynb IPython Notebook is present. It can be directly run in the Google Colab envirionment. Otherwise we suggest creating a virtual envirionment 
and run a Jupyter Lab notebbok in it. The environment dependencies are in the yml file.
