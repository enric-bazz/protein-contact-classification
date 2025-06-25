# Protein Contact Type Classification

This project is part of the **Structural Bioinformatics course at the University of Padova**. It focuses on the **classification of residue-residue contacts** in protein structures using **machine learning** methods trained on geometric and physico-chemical features.

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Quick-start](#quick-start)
- [Authors](#authors)

## Project Overview

Protein structures contain a variety of intramolecular interactions critical for their function and stability. The goal of this project is to **predict the type of contact** between amino acid residues, such as hydrogen bonds, van der Waals interactions, and salt bridges, using statistical or supervised learning models. The classification mimics the one provided by the [RING software](https://ring.biocomputingup.it/), but is implemented independently and does not rely on its geometric rules.

The objectives of the project are:

- Extract features from protein structures using Python tools and third-party software (e.g., DSSP).
- Train predictive models to classify contacts into types defined by RING.
- Evaluate model performance using standard metrics.
- Provide comprehensive documentation and reproducible code.



## Repository Structure
 - **Notebooks**: Contains all the notebooks used for feature extraction, data pre-preocessing, model developement and additional material with detailed step-by-step explanations.
 - **Predictor**: Contains ready-to-use software and documentation
 - **Report**: Contains the report of the project, including methodologies, final results, and discussions.
 - **models_dev_results**: Contains the developed models, validtion metrics, parameters and result plots organized as per-model and per-dataset experiments.
 - `features_ring.zip`: zip archive with model developement raw data.
 - `environment.yml` and `requirements.txt`: Files for managing project dependencies.
 - `output`: Software output folder example, obtained with the `1i27.tsv` file.
 - `calc_features.py` and `calc_3di-py`: Scripts that produce the files fed in input to the software.

```bash
├── Notebooks/                      # Jupyter notebooks for experimentation and development
│   ├── Experiments Missing Interactions.ipynb
│   ├── Feature Extraction.ipynb
│   ├── MultiLabel NN Model.ipynb
│   ├── PerLabel NN Model.ipynb
│   ├── Pre-Processing New Data.ipynb
│   ├── Pre-Processing Original Data.ipynb
│   └── XGBoost Model.ipynb

├── Predictor/                      # Prediction pipeline and pretrained models
│   ├── calc_additional_features.py        # Additional feature computation
│   ├── onehot_encoder.pkl                 # Pretrained encoder for categorical features
│   ├── Predictor.ipynb                    # Interactive prediction workflow
│   ├── README.md                          # Module-level documentation
│   ├── run_prediction.py                  # Main script for predicting contact types
│   ├── states.txt                         # States definition for feature extraction
│   └── models/
│       └── xgboost/                       # Trained XGBoost models (one per class)
│           ├── HBOND.joblib
│           ├── IONIC.joblib
│           ├── PICATION.joblib
│           ├── PIHBOND.joblib
│           ├── PIPISTACK.joblib
│           ├── SSBOND.joblib
│           └── VDW.joblib

├── models_dev_results/        # Training results and exported models
│   ├── perlabel_nn/           # Results for neural networks
│   └── xgboost/               # Results for XGBoost runs (SMOTE and non-SMOTE)

├── output/                    # Example predictions and intermediate data
│   ├── pdb_id.tsv             # pre-processed tsv file
│   ├── pdb_id_predictions.csv # software prediction (FINAL OUTPUT)
│   └── features_ring_extended.zip

├── Report/                    # Final project report
│   └── Report.pdf

├── features_ring.zip              # Raw structural interaction data for training
├── environment.yml                # Conda environment definition
├── requirements.txt               # pip-based dependency list
├── 1i27.tsv                       # Example input
├── calc_3di.py                    # 3Di descriptor extractor
├── calc_features.py               # Feature assembly logic
└── README.md                      # Main project documentation

```

## Quick-start
To use the software, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/enric-bazz/protein-contact-classification.git
    ```

2. Navigate to the repository directory and use it as your working directory:
    ```sh
    cd protein-contact-classification
    ```
    Move your input data into this directory. Once the data is in place, you can run the prediction script directly:
    ```sh
   python Predictor/run_prediction.py pdb_id.tsv
    ```
   Alternatively, navigate to your preferred working directory and copy the Predictor/ folder into it:
   - On Unix/macOS:
   ```bash
   cd /path/to/your/working_directory
   cp -r /path/to/protein-contact-classification/Predictor .
   ```
   - On Windows:
   ```cmd
   cd path\to\your\working_directory
   xcopy /E /I /Y path\to\protein-contact-classification\Predictor .
   ```
   Then place your .tsv input files in the same directory and run:
   ```sh
   python Predictor/run_prediction.py pdb_id.tsv
   ```


4. For usage details and extended documentation consult the `README.md` file in the `Predictor` folder.

## Authors
This project is developed by the following authors:

- Enrico Bazzacco: enrico.bazzacco@studenti.unipd.it
- Valentina Signor: valentina.signor@studenti.unipd.it
- Filip Trajkovski: filip.trajkovski@studenti.unipd.it
