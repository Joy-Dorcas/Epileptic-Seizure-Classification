# Epileptic Seizure Classification Using BEED Dataset

## Project Overview

This project aims to implement automated epileptic seizure detection and classification using the Bangalore EEG Epilepsy Dataset (BEED). The goal is to build a machine learning pipeline that can distinguish between different types of seizures (Focal, Generalized) and healthy brain activity using EEG signal analysis.

## Dataset Information

**BEED (Bangalore EEG Epilepsy Dataset)**
- **Total Samples**: 8,000 EEG recordings
- **Recording Duration**: 20 seconds per sample
- **Sampling Rate**: 256 Hz
- **Channels**: 16 EEG channels (X1-X16) following the 10-20 electrode placement system
- **Categories**:
  - Class 0: Healthy Subjects (2,000 samples)
  - Class 1: Generalized Seizures (2,000 samples)
  - Class 2: Focal Seizures (2,000 samples)
  - Class 3: Seizure Events with physical movements (2,000 samples)

## Project Goals

1. Explore and visualize EEG signal patterns across different seizure types
2. Extract meaningful features from time-domain and frequency-domain representations
3. Implement dimensionality reduction techniques
4. Build classification models to detect and classify seizures
5. Evaluate model performance and compare with baseline approaches

## Planned Methodology

### 1. Data Preprocessing
- Load and explore the BEED dataset
- Check for missing values and data quality issues
- Normalize/standardize EEG signals
- Split data into training, validation, and test sets

### 2. Feature Engineering

#### Temporal Features
- Apply UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction
- Preserve temporal dynamics and relationships in the data
- Reduce from 16 dimensions to 3 dimensions

#### Spectral Features
- Apply Fast Fourier Transform (FFT) to extract frequency domain information
- Analyze brain wave frequencies (delta, theta, alpha, beta, gamma)
- Capture oscillatory patterns characteristic of seizures

#### Combined Features
- Merge temporal and spectral features
- Create comprehensive feature set for classification

### 3. Model Development

#### Baseline Models
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

#### Advanced Models
- Long Short-Term Memory (LSTM) networks
- XGBoost (Extreme Gradient Boosting)
- Gradient Boosting

#### Ensemble Approach (SeqBoostNet)
- Implement stacking ensemble with multiple base models
- Use AdaBoost as meta-learner
- Combine predictions for improved accuracy

### 4. Classification Tasks

Plan to evaluate models on multiple binary and multiclass scenarios:
- **A1**: Generalized vs Focal
- **A2**: Generalized vs Healthy
- **A3**: Focal vs Healthy
- **A4**: Focal vs Seizure Events
- **A5**: Generalized vs Seizure Events
- **A6**: Seizure Events vs Healthy

## Expected Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
- Confusion Matrix
- Sensitivity and Specificity
- Cohen's Kappa
- Matthews Correlation Coefficient (MCC)

## Technology Stack

### Required Libraries
```
- Python 3.8+
- NumPy - numerical computing
- Pandas - data manipulation
- Scikit-learn - machine learning algorithms
- TensorFlow/Keras - deep learning (LSTM)
- XGBoost - gradient boosting
- Matplotlib/Seaborn - visualization
- SciPy - signal processing and FFT
- UMAP-learn - dimensionality reduction
```

## Project Structure

```
epilepsy-seizure-detection/
│
├── data/
│   └── beed_dataset.csv
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   └── 04_advanced_models.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── models.py
│   └── evaluation.py
│
├── results/
│   ├── figures/
│   └── metrics/
│
├── requirements.txt
└── README.md
```

## Installation Steps

1. Clone this repository
```bash
git clone <repository-url>
cd epilepsy-seizure-detection
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download BEED dataset and place in `data/` directory

## Implementation Roadmap

### Phase 1: Data Exploration 
- [ ] Load and inspect dataset
- [ ] Visualize EEG signals for each class
- [ ] Statistical analysis of features
- [ ] Check class distribution

### Phase 2: Feature Engineering 
- [ ] Implement FFT for spectral features
- [ ] Apply UMAP for temporal features
- [ ] Create combined feature set
- [ ] Visualize feature distributions

### Phase 3: Baseline Models 
- [ ] Train simple classifiers
- [ ] Evaluate performance metrics
- [ ] Establish baseline accuracy

### Phase 4: Advanced Models 
- [ ] Implement LSTM network
- [ ] Train XGBoost and Gradient Boosting
- [ ] Compare with baselines

### Phase 5: Ensemble Learning 
- [ ] Build SeqBoostNet stacking model
- [ ] Optimize hyperparameters
- [ ] Final evaluation and comparison

### Phase 6: Documentation 
- [ ] Document findings
- [ ] Create visualizations
- [ ] Write final report

## Target Performance Benchmarks

Based on reference research, aiming for:
- Overall accuracy: >95%
- Focal vs Generalized: >95%
- Binary classifications: >99%
- Multiclass classification: >90%

## Challenges to Address

1. **High dimensionality**: 16 channels × 5,120 time points per recording
2. **Class imbalance**: Ensuring equal representation in training
3. **Signal noise**: EEG signals can be noisy and contain artifacts
4. **Computational complexity**: Large dataset and complex models
5. **Generalization**: Model should work across different subjects

## Future Enhancements

- Real-time seizure prediction
- Patient-specific model customization
- Integration with wearable EEG devices
- Web application for clinical use
- Extended validation on additional datasets (BONN, CHB-MIT)

## References

1. Najmusseher & Nizar Banu P K (2025). "Feature Engineering for Epileptic Seizure Classification Using SeqBoostNet". International Journal of Computing and Digital Systems, Vol. 17, No. 1.

2. Original BEED Dataset: [Kaggle/UCI Repository]


## Contributors
Joy Dorcas=joymanyara55@gmail.com

## Acknowledgments

Special thanks to the Bangalore EEG clinic for providing the BEED dataset and contributing to epilepsy research.
