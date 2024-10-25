# Data Sources Documentation

This directory contains datasets used for the multimodal early detection model for breast cancer. The datasets are organized into three main categories: Clinical Data, Genomic Data, and Imaging Data.

## Data Categories

### Clinical Data
**Google Drive Location:** [Clinical Data Folder](https://drive.google.com/drive/folders/1SbxErFl1QOwuXqDGlzUSplEsex7U93hJ?usp=sharing)

Contains patient data including:
- Cancer Data
- Real Breast Cancer Data 
- Surveillance, Epidemiology, and End Results (SEER) Program Data
- Breast Cancer Wisconsin (Diagnostic) Dataset

#### Kaggle Sources:
- Cancer Data: https://www.kaggle.com/datasets/erdemtaha/cancer-data/data
- Real Breast Cancer Data: https://www.kaggle.com/datasets/amandam1/breastcancerdataset
- SEER Program Data: https://www.kaggle.com/datasets/reihanenamdari/breast-cancer
- Breast Cancer Wisconsin Dataset: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

### Genomic Data  
**Google Drive Location:** [Genomic Data Folder](https://drive.google.com/drive/folders/1mFVKaY_8LenfkLZPZ8r4JQqnZOei9SUP?usp=sharing)

Contains genetic expression and mutation data from:
- CuMiDa Dataset
- METABRIC Dataset

#### Kaggle Sources:
- CuMiDa: https://www.kaggle.com/datasets/brunogrisci/breast-cancer-gene-expression-cumida
- METABRIC: https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric

### Imaging Data
**Google Drive Location:** [Imaging Data Folder](https://drive.google.com/drive/folders/16ztO8oL9b-5Fnsdf0O8XTxYT1PQbcsDJ?usp=sharing)

Contains medical imaging data from:
- BreakHis Database
- CBIS-DDSM Dataset

#### Kaggle Sources:
- BreakHis Database: https://www.kaggle.com/datasets/ambarish/breakhis
- CBIS-DDSM: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

## File Organization

```
data/
├── clinical/
│   ├── Breast Cancer/
│   ├── Cancer Data/
│   ├── Real Breast Cacer Data/
│   └── Wisconsin Dataset/
├── genomic/
│   ├── CuMiDa/
│   └── Metabric/
└── imaging/
    ├── BreakHis Images/
    └── CBIS-DDSM Images/
```
