# Telco Churn Prediction MLOps Project Structure

## Project Overview

This is a comprehensive MLOps project for predicting customer churn in a telecommunications company. The project follows best practices for machine learning operations, including:

1. **Data Processing Pipeline**: Modular code for data ingestion, cleaning, feature engineering, and preprocessing
2. **Model Development**: Multiple models developed, including baseline models and advanced ensemble methods
3. **Model Evaluation**: Comprehensive evaluation metrics and comparison of model performance
4. **Production Deployment**: Production-ready pipeline with serialized models
5. **Business Impact Analysis**: ROI assessment and identification of high-risk customers
6. **Testing**: Unit tests for critical components of the pipeline

The project structure follows a clean, modular architecture that separates concerns and promotes maintainability and reproducibility.

---

## Project Structure

```bash
├── 01_telco_churn_eda_comprehensive.ipynb     # Exploratory Data Analysis notebook
├── 02_preprocessing_ensemble_models_pipeline.ipynb  # Data preprocessing and model pipeline notebook
├── 03_model_evaluation.ipynb                  # Model evaluation notebook
├── 04_production_ready_pipeline.ipynb         # Production pipeline notebook
├── 05_business_impact_analysis.ipynb          # Business impact analysis notebook
├── README.md                                  # Project documentation
├── requirements.txt                           # Python dependencies
├── artifacts/                                 # Preprocessed data artifacts
│   ├── X_test.npz                             # Test features
│   ├── X_train.npz                            # Training features
│   ├── Y_test.npz                             # Test labels
│   └── Y_train.npz                            # Training labels
├── catboost_info/                             # CatBoost model information
│   └── learn/
│       └── events.out.tfevents                # Training events log
├── configs/
│   └── config.yaml                            # Configuration parameters
├── models/                                    # Trained model files
│   ├── pipeline_artifacts/                    # Pipeline metadata
│   │   └── catboost_info/                     # CatBoost model information
│   │       ├── catboost_training.json         # Training parameters
│   │       ├── learn_error.tsv                # Learning error metrics
│   │       ├── time_left.tsv                  # Training time tracking
│   │       ├── learn/
│   │       │   └── events.out.tfevents        # Training events log
│   │       └── tmp/
│   └── trained_models/                        # Serialized trained models
│       ├── baseline_dt_pipeline.joblib        # Decision Tree baseline model
│       ├── baseline_lr_pipeline.joblib        # Logistic Regression baseline model
│       ├── cat_best_pipeline.joblib           # Best CatBoost model
│       ├── RandomForestClassifier_pipeline.joblib  # Random Forest model
│       ├── rf_best_pipeline.joblib            # Best Random Forest model
│       ├── stacking_ensemble_pipeline.joblib  # Stacking ensemble model
│       └── xgb_best_pipeline.joblib           # Best XGBoost model
├── reports/                                   # Generated reports and outputs
│   ├── roi_scenarios_high_segment.csv         # ROI analysis for high-risk segments
│   ├── scored_customers.csv                   # Customer churn predictions
│   └── top100_high_risk_customers.csv         # Top 100 customers at risk of churning
├── src/                                       # Source code
│   ├── __init__.py                            # Package initialization
│   ├── data_ingestion.py                      # Data loading modules
│   ├── data_pipeline.py                       # Data processing pipeline
│   ├── data_splitter.py                       # Train/test split functionality
│   ├── feature_binning.py                     # Feature binning transformations
│   ├── feature_encoding.py                    # Feature encoding modules
│   ├── feature_scaling.py                     # Feature scaling modules
│   ├── handle_missing_values.py               # Missing value handling
│   ├── model_building.py                      # Model architecture construction
│   ├── model_evaluation.py                    # Model evaluation metrics
│   ├── model_inference.py                     # Inference functionality
│   ├── model_training.py                      # Model training modules
│   ├── outlier_detection.py                   # Outlier detection and handling
│   ├── streaming_inference_pipeline.py        # Real-time inference pipeline
│   ├── utils.py                               # Utility functions
│   ├── __pycache__/                           # Compiled Python files
│   └── data/                                  # Data directory
│       ├── external/                          # External data sources
│       │   ├── models_performance_summary.csv # Model performance metrics
│       │   └── part4_performance_summary.csv  # Phase 4 performance metrics
│       ├── processed/                         # Processed dataset
│       │   └── telco-customer-churn_cleaned.csv  # Cleaned dataset
│       └── raw/                               # Raw dataset
│           └── telco-customer-churn-raw.csv   # Original dataset
└── tests/                                     # Unit and integration tests
    ├── test_data_ingestion.py                 # Tests for data ingestion
    ├── test_handle_missing_values.py          # Tests for missing value handling
    └── __pycache__/                           # Compiled test files
```


