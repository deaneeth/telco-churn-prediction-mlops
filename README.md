# 📊 Telco Customer Churn Prediction - MLOps Project

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.0-orange)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.14.0-green)

> ⚠️ **Note**: This project requires scikit-learn 1.7.0 and imbalanced-learn 0.14.0. Using different versions may cause compatibility issues.

## Project Overview

This project implements an end-to-end machine learning pipeline for predicting customer churn in a telecommunications company. By identifying customers likely to cancel their services, the company can implement targeted retention strategies.

### Key Features

- **Data Analysis**: Exploratory data analysis with visualizations
- **Machine Learning Pipeline**: Modular preprocessing and model training
- **Multiple Models**: Comparison of models from baseline to advanced ensemble methods
- **Production-Ready**: FastAPI implementation for real-time inference
- **Business Insights**: ROI calculations and customer segmentation

## 🏢 Business Context 

This project helps telecommunications companies:

- 🔍 Identify high-risk customers before they churn
- 🧩 Segment customers for targeted retention efforts
- 💰 Estimate the financial impact of retention campaigns
- ⚡ Deploy real-time predictions for new customer data

## 📁 Project Structure 

Main components:

- 📓 **Notebooks**: Step-by-step workflow from EDA to business impact analysis
- 💻 **Source Code**: Modular Python code for all pipeline components
- 🤖 **Models**: Trained machine learning models
- 📊 **Data**: Raw and processed datasets
- 📈 **Reports**: Generated outputs and business insights

For a detailed directory structure of the project, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## 🚀 Installation 

1. Clone the repository

   ```bash
   git clone https://github.com/deaneeth/telco-churn-prediction-mlops.git
   cd telco-churn-prediction-mlops
   ```

2. Create a virtual environment and install dependencies

   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   # OR
   source venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   ```

## 📋 Dataset 

The dataset contains customer information including:

- 👤 Demographics (gender, age, partner status)
- 📝 Account details (tenure, contract type, payment method)
- 📞 Services subscribed (phone, internet, add-ons)
- 💵 Financial information (monthly charges, total charges)
- 🎯 Target variable: customer churn status

## 🔄 Workflow 

1. 🔍 **Data Exploration**: Analysis of customer churn patterns and feature relationships
2. 🧹 **Preprocessing**: Data cleaning, feature engineering, and preprocessing pipeline
3. 🛠️ **Model Development**: Training multiple models and optimizing performance
4. 📊 **Evaluation**: Comprehensive metrics and model comparison
5. 📦 **Production Pipeline**: Packaging for deployment
6. 💼 **Business Analysis**: Customer segmentation and ROI calculations

## 📈 Results 

### Model Evaluation Results

| Model              | ROC-AUC   | PR-AUC    | Precision  | Recall    | F1        |
|--------------------|-----------|-----------|------------|-----------|-----------|
| LogisticRegression | 0.847129  | 0.666142  | 0.500840   | 0.796791  | 0.615067  |
| RandomForest       | 0.841550  | 0.650611  | 0.518389   | 0.791444  | 0.626455  |
| XGBoost            | 0.846979  | 0.660561  | 0.685714   | 0.513369  | 0.587156  |
| CatBoost           | 0.845748  | 0.666034  | 0.513605   | 0.807487  | 0.627859  |
| StackingEnsemble   | 0.842646  | 0.644170  | 0.667832   | 0.510695  | 0.578788  |
| DecisionTree       | 0.627193  | 0.352190  | 0.461972   | 0.438503  | 0.449931  |

> 🚀 **Improvement Note**: These results can be further improved by implementing:
>
> - Advanced feature engineering techniques (RFM analysis, customer behavior patterns)
> - Deep learning approaches (Neural networks for complex pattern recognition)
> - Hyperparameter tuning with more extensive search space
> - Better handling of class imbalance using advanced sampling techniques
>
> This project will be enhanced with these improvements in the near future.

### 💡 Key Insights 

- 🥇 CatBoost achieved the highest F1 score (0.627859) with the best balance of precision and recall
- 📊 LogisticRegression shows strong performance with high ROC-AUC (0.847129)
- 🎯 XGBoost provides the highest precision (0.685714) but with lower recall
- 📈 The ensemble models generally outperform the baseline models (Decision Tree)

### 🔑 Feature Importance 

- 📝 Contract type, tenure, and service issues are the strongest predictors
- ⚠️ Month-to-month contracts with technical issues show highest churn rates
- 💰 Targeted interventions show 3-5x return on investment

## 🚀 Deployment 

The model will be deployed as a real-time prediction service using FastAPI:

```bash
will be updated...
```

## 🔮 Future Work 

- 🔄 Model monitoring and retraining pipeline
- 🗃️ Feature store for reproducibility
- 🔍 Advanced model interpretability
- 🧪 A/B testing framework for retention strategies
