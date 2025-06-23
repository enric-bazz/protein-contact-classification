# Protein Contact Type Classification

This project is part of the **Structural Bioinformatics course at the University of Padova**. It focuses on the **classification of residue-residue contacts** in protein structures using **machine learning** methods trained on geometric and physico-chemical features.

## 🧬 Project Overview

Protein structures contain a variety of intramolecular interactions critical for their function and stability. The goal of this project is to **predict the type of contact** between amino acid residues, such as hydrogen bonds, van der Waals interactions, and salt bridges, using statistical or supervised learning models. The classification mimics the one provided by the [RING software](https://ring.biocomputingup.it/), but is implemented independently and does not rely on its geometric rules.

## 📊 Objectives

- Extract features from protein structures using Python tools and third-party software (e.g., DSSP).
- Train predictive models to classify contacts into types defined by RING.
- Evaluate model performance using standard metrics.
- Provide comprehensive documentation and reproducible code.

## 📁 Dataset

- **Training Data**: Extracted from ~3,900 PDB structures using custom feature extraction scripts.
- **Features**: Include secondary structure, solvent accessibility, backbone angles, Atchley factors, and 3Di encoding.
- **Labels**: Contact types such as HBOND, VDW, PIPISTACK, IONIC, etc.

## 📦 Repository Structure (Template)

```bash
.
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
