## 🩺 Uncertainty-aware Glucose Prediction using Evidential Regression

### 📖 Overview

This repository implements evidential regression for blood glucose prediction using continuous glucose monitoring (CGM) data.


---

### 🌳 Repository Structure

```
├── configs.py     # Experiment configuration (hyperparameters, paths, settings)
├── models.py      # Model definitions 
├── utils.py       # Utility functions (loss functions, metrics)
├── data.py        # Data-related helper functions 
├── train.py       # Training script
├── evaluate.py    # Evaluation script

```

---

### ⚙️ Configuration

All experiment settings are defined in:

```
configs.py
```

Users can specify:

* model parameters
* training hyperparameters
* choice of input features

---

### 🚀 Usage

#### 1. Set up configuration

Edit `configs.py` to specify your experiment settings and various paths for data, saved model, etc.

---

#### 2. Train the model

```bash
python train.py
```
This will:

* load the dataset
* initialize the model
* train using specified parameters
* save model 

---

#### 3. Evaluate the model

```bash
python evaluate.py
```

This will:

* load model
* generate predictions
* compute evaluation metrics

---

### 📙 Dataset

This project uses publicly available CGM data (preprocessed folder):

* https://data.mendeley.com/datasets/3hbcscwz44/1

The dataset should be placed locally, and the path must be set through specifying 'data_dir' in configs.py

---

### 🌱 Notes

* The HUPA-UCM dataset is not included and must be downloaded separately.
* The sample model file 'H_S_transf_evid_kl_model.pth' corresponds to a (trained) evidential transformer-based model trained on heart-rate-included inputs (with 30-min predictive horizon). 

---

### ✍️ Authors

Hai Siong Tan and Rafe McBeth 

Preprint: https://arxiv.org/abs/2603.04955 

### 📝 License

This project is released under the Apache License 2.0.

---
