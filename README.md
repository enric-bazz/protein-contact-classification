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
 - `features_ring.zip`: zip archive with model developement raw data.
 - `environment.yml` and `requirements.txt`: Files for managing project dependencies.

```bash
├── Notebooks/
│   └── classifier.ipynb      # Naive Bayes baseline example
├── Predictor/
│   ├── models/
│   │   ├── xgboost/
│   └── predict.py            # Contact type prediction from new PDB files
├── Report/
│   └── report.pdf            # Final report
├── features_ring.zip         # Raw data for model developement
├── README.md
├── environment.yml
└── requirements.txt
```

## Usage
To use the software, follow these initial steps:

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
   python Predictor/run_prediction.py
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
   python Predictor/run_prediction.py
   ```


4. For usage details and extended documentation consult the `README.md` file in the `Predictor` folder.

## Authors
This project is developed by the following authors:

- Enrico Bazzacco: enrico.bazzacco@studenti.unipd.it
- Valentina Signor: valentina.signor@studenti.unipd.it
- Filip Trajkovski: filip.trajkovski@studenti.unipd.it
