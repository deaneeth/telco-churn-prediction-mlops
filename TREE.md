├── .gitignore
├── .pytest_cache/
│   ├── .gitignore
│   ├── CACHEDIR.TAG
│   ├── README.md
│   └── v/
│       └── cache/
│           ├── lastfailed
│           ├── nodeids
│           └── stepwise
├── 01_telco_churn_eda_comprehensive.ipynb
├── 02_preprocessing_ensemble_models_pipeline.ipynb
├── 03_model_evaluation.ipynb
├── 04_production_ready_pipeline.ipynb
├── README.md
├── requirements.txt
├── artifacts/
│   ├── X_test.npz
│   ├── X_train.npz
│   ├── Y_test.npz
│   └── Y_train.npz
├── catboost_info/
│   ├── catboost_training.json
│   ├── learn_error.tsv
│   ├── time_left.tsv
│   ├── learn/
│   │   └── events.out.tfevents
│   └── tmp/
│       └── cat_feature_index.df883914-8f840f6b-fc31ca81-b0e1c302.tmp
├── configs/
│   └── config.yaml
├── models/
│   ├── pipeline_artifacts/
│   │   └── catboost_info/
│   │       ├── catboost_training.json
│   │       ├── learn_error.tsv
│   │       ├── time_left.tsv
│   │       └── learn/
│   │           └── events.out.tfevents
│   └── trained_models/
│       ├── baseline_dt_pipeline.joblib
│       ├── baseline_lr_pipeline.joblib
│       ├── cat_best_pipeline.joblib
│       ├── RandomForestClassifier_pipeline.joblib
│       ├── rf_best_pipeline.joblib
│       ├── stacking_ensemble_pipeline.joblib
│       └── xgb_best_pipeline.joblib
├── reports/
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── data_pipeline.py
│   ├── data_splitter.py
│   ├── feature_binning.py
│   ├── feature_encoding.py
│   ├── feature_scaling.py
│   ├── handle_missing_values.py
│   ├── model_building.py
│   ├── model_evaluation.py
│   ├── model_inference.py
│   ├── model_training.py
│   ├── outlier_detection.py
│   ├── streaming_inference_pipeline.py
│   ├── utils.py
│   ├── __pycache__/
│   │   ├── __init__.cpython-311.pyc
│   │   ├── data_ingestion.cpython-311.pyc
│   │   ├── data_pipeline.cpython-311.pyc
│   │   ├── data_splitter.cpython-311.pyc
│   │   ├── feature_encoding.cpython-311.pyc
│   │   ├── feature_scaling.cpython-311.pyc
│   │   ├── handle_missing_values.cpython-311.pyc
│   │   ├── model_building.cpython-311.pyc
│   │   ├── model_evaluation.cpython-311.pyc
│   │   ├── model_inference.cpython-311.pyc
│   │   ├── model_training.cpython-311.pyc
│   │   ├── outlier_detection.cpython-311.pyc
│   │   └── utils.cpython-311.pyc
│   └── data/
│       ├── external/
│       │   ├── models_performance_summary.csv
│       │   └── part4_performance_summary.csv
│       ├── processed/
│       │   └── telco-customer-churn_cleaned.csv
│       └── raw/
│           └── telco-customer-churn-raw.csv
└── tests/
    ├── test_data_ingestion.py
    ├── test_handle_missing_values.py
    └── __pycache__/
        ├── test_data_ingestion.cpython-313-pytest-8.3.4.pyc
        └── test_handle_missing_values.cpython-313-pytest-8.3.4.pyc