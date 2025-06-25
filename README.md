# Protein Contact Type Classification

This project is part of the **Structural Bioinformatics course at the University of Padova**. It focuses on the **classification of residue-residue contacts** in protein structures using **machine learning** methods trained on geometric and physico-chemical features.

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
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
```bash
├── data/
│   ├── raw/                  # Raw input data and PDB files
│   ├── processed/            # Feature matrices and cleaned data
├── features/
│   ├── calc_features.py      # Extracts structural features using BioPython and DSSP
│   ├── calc_3di.py           # Converts sequences to 3Di representation
├── models/
│   ├── train.py              # Training script for ML models
│   ├── model.pkl             # Example saved model
├── predictor/
│   └── predict.py            # Contact type prediction from new PDB files
├── config/
│   └── config.json           # Paths to tools, data, and model hyperparameters
├── notebooks/
│   └── classifier.ipynb      # Naive Bayes baseline example
├── docs/
│   └── report.pdf            # Final report
├── README.md
└── requirements.txt
```

## Usage
To use the software, follow these initial steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/protein-contact-classification.git
    cd protein-contact-classification
    ```

2. Navigate to the repository directory (suggested working directory):
    ```sh
    cd protein-contact-classification
    ```

3. Follow the folder `README.md` in `Predictor`

## Authors
This project is developed by the following authors:

- Enrico Bazzacco: enrico.bazzacco@studenti.unipd.it
- Valentina Signor: valentina.signor@studenti.unipd.it
- Filip Trajkovski: filip.trajkovski@studenti.unipd.it
