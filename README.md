# Bearing Fault Diagnosis Using Machine Learning and Deep Learning

## 1. Project Overview

This project implements a modular and reproducible pipeline for bearing fault diagnosis using the Paderborn University bearing dataset.

The main objective is to classify bearing operating conditions into three categories:

- **Healthy**
- **Outer Race Fault**
- **Inner Race Fault**

The proposed pipeline processes current and vibration signals, extracts temporal and frequency-domain features, and evaluates different Machine Learning (ML) and Deep Learning (DL) models.

The project is designed with modularity and reproducibility in mind. Data processing, feature extraction, model training, evaluation, experiment management, and visualization are implemented in separate modules.

## 2. Dataset and Signal Processing

### 2.1 Dataset

The project uses the Paderborn University bearing dataset.

The selected bearings are:

| Class | Bearing IDs |
|---|---|
| Healthy | K001, K002, K003, K004, K005 |
| Outer Race | KA04, KA15, KA16, KA22, KA30 |
| Inner Race | KI04, KI14, KI16, KI18, KI21 |

A total of 15 bearings are considered. Bearing K006 is reserved and is not included in the current experiments.

### 2.2 Signal Configuration

Two signals are extracted from each MAT file:

- `phase_current_1`
- `vibration_1`

The signal processing configuration is:

| Parameter | Value |
|---|---:|
| Sampling frequency | 64 kHz |
| Segment duration | 100 ms |
| Samples per segment | 6,400 |
| Overlap | None |

Current and vibration signals are segmented jointly to preserve temporal correspondence.

### 2.3 Feature Extraction

A total of 22 features are extracted from each segment:

**Temporal features (16):**

Eight features are calculated for each signal:

- Mean
- Standard deviation
- Minimum
- Maximum
- Root Mean Square (RMS)
- Peak-to-peak
- Energy
- Crest factor

**Frequency-domain features (6):**

Three features are calculated for each signal:

- Dominant magnitude
- Spectral energy
- Spectral centroid

The resulting dataset contains 22 numerical features and 5 metadata columns:

- `bearing_id`
- `class_id`
- `class_name`
- `experiment_id`
- `segment_id`

The generated dataset currently contains 48,042 segments from 1,200 experiments and 15 bearings.

## 3. Machine Learning and Deep Learning Models

The project currently implements five models:

1. Support Vector Machine (SVM)
2. K-Nearest Neighbors (KNN)
3. Random Forest
4. Multi-Layer Perceptron (MLP) using Scikit-learn
5. Multi-Layer Perceptron (MLP) using PyTorch

The Scikit-learn models and the PyTorch implementation are defined independently to facilitate model comparison and future modifications.

The PyTorch network uses two hidden layers with 50 neurons each and ReLU activation.

Feature scaling is performed using training data only. The scaler is fitted within each training fold and subsequently applied to its corresponding test fold.

## 4. Validation Strategy

The models are evaluated using five-fold Stratified Group Cross-Validation.

The validation strategy uses:

- `StratifiedGroupKFold`
- 5 folds
- Random state: 42
- Grouping variable: `bearing_id`
- Target variable: `class_id`

Grouping by bearing ensures that segments from the same physical bearing are not simultaneously included in the training and test subsets of a fold.

The same cross-validation folds are reused across all models to ensure a consistent comparison.

## 5. Project Structure

```text
bearing_fault/
│
├── main.py
├── environment.yml
├── README.md
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── dataset_config.py
│   │   ├── data_loader.py
│   │   ├── segmentation.py
│   │   └── preprocessing.py
│   │
│   ├── feature_extraction/
│   │   ├── __init__.py
│   │   ├── time_features.py
│   │   ├── frequency_features.py
│   │   ├── features.py
│   │   └── dataset_builder.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── validation.py
│   │   ├── model_builders.py
│   │   └── pytorch_model.py
│   │
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   ├── results.py
│   │   └── experiment_runner.py
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── metrics_visualization.py
│       ├── confusion_matrix_visualization.py
│       ├── training_visualization.py
│       └── experiment_comparison.py
│
├── data/
│   ├── raw/
│   │   └── paderborn/
│   └── processed/
│       └── features/
│           └── features.csv
│
├── models/
│
├── results/
│   ├── configs/
│   ├── metrics/
│   │   ├── folds/
│   │   └── training_history/
│   ├── predictions/
│   └── figures/
│
├── tests/
│
└── notebooks/
```

## 6. Module Responsibilities

### `src/data_processing/`

Contains the modules responsible for dataset configuration, MAT file loading, signal segmentation, and processed dataset validation.

### `src/feature_extraction/`

Implements temporal and frequency-domain feature extraction, feature combination, and dataset generation.

### `src/models/`

Contains model validation strategies, Scikit-learn model builders, and the PyTorch neural network architecture.

### `src/experiments/`

Manages model training, predictions, metric calculation, experiment registration, configuration storage, and execution of the cross-validation experiments.

### `src/visualization/`

Contains independent functions for visualizing experiment metrics, confusion matrices, training histories, and model comparisons.

### `data/`

Stores the original dataset and the processed feature dataset.

### `models/`

Reserved for trained model artifacts.

### `results/`

Stores experiment configurations, fold metrics, summary metrics, predictions, training histories, and generated figures.

### `notebooks/`

Reserved for exploratory analysis, individual experiments, code demonstrations, and documentation of the processing and modeling procedures.

## 7. Installation and Environment Setup

The project uses Conda to manage its Python environment.

Create the environment using:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate bearing_fault
```

The environment includes NumPy, SciPy, Pandas, Matplotlib, Scikit-learn, Joblib, Jupyter, and Pytest.

PyTorch is installed separately using the CPU build:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
```

## 8. Dataset Preparation

Place the Paderborn MAT files inside:

```text
data/raw/paderborn/
```

The dataset builder searches recursively for MAT files and processes the selected bearings.

The processed feature dataset is stored at:

```text
data/processed/features/features.csv
```

The main pipeline automatically reuses the existing feature dataset when available.

To generate the dataset for the first time, execute:

```bash
python main.py
```

If the processed dataset does not exist, it is generated automatically before the experiments begin.

To force dataset reconstruction:

```bash
python main.py --rebuild-dataset
```

**Note:** Forced reconstruction replaces the existing `features.csv` file.

## 9. Running the Experiments

From the project root, activate the Conda environment and execute:

```bash
conda activate bearing_fault
python main.py
```

The main script coordinates the following stages:

1. Dataset preparation.
2. Loading the processed feature dataset.
3. Generation of cross-validation folds.
4. Model training and evaluation.
5. Saving experiment metrics and predictions.
6. Registration of experiment configurations and results.

The enabled models and their main training parameters are configured in:

```text
src/experiments/experiment_runner.py
```

## 10. Results and Experiment Management

The project automatically stores the experiment outputs under `results/`.

| Directory | Contents |
|---|---|
| `results/configs/` | Experiment configurations |
| `results/metrics/` | Fold metrics and summary statistics |
| `results/metrics/folds/` | Individual experiment fold results |
| `results/metrics/training_history/` | PyTorch training histories |
| `results/predictions/` | Test predictions and associated metadata |
| `results/figures/` | Generated visualizations |

The file `results/experiments.csv` acts as the central experiment registry.

Each experiment receives a unique identifier, allowing results from different models and executions to be tracked.

## 11. Visualization

Visualization functions are implemented independently in `src/visualization/`.

The available functions support:

- Comparison of experiment metrics.
- Analysis of metrics across cross-validation folds.
- Confusion matrix visualization.
- Normalized confusion matrices.
- PyTorch training loss and accuracy curves.
- Comparison of selected experiments.

These functions can be called independently when visual analysis is required. They are not automatically executed by the main experiment pipeline.

## 12. Reproducibility and Development

The project follows a modular development approach:

- Processing and feature extraction are separated from model training.
- The same cross-validation folds are reused across models.
- Feature scaling is fitted exclusively on training data.
- Experiment configurations and results are stored systematically.
- Random states are configured to support reproducibility.

Exploratory analysis and individual experiments will be developed in Jupyter notebooks. Stable implementations will remain in the `src/` modules and will be imported by the notebooks to avoid duplicated code.

## 13. Future Work

Potential extensions include:

- Additional temporal and frequency-domain features.
- Evaluation of alternative ML and DL architectures.
- Additional signal analysis and visualization.
- Model optimization and complexity analysis.
- Hardware-oriented model development and eventual FPGA implementation.
