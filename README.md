# Personal Loan Acceptance Prediction Using Machine Learning

## Project Overview
This project uses machine learning to predict which bank customers are likely to accept personal loan offers. The goal is to optimize marketing campaigns by targeting customers with the highest probability of acceptance, thereby improving conversion rates and reducing marketing costs.
## Problem Statement
Banks offer personal loans to their existing customers, but acceptance rates are typically low (9.7% in this dataset). By accurately predicting which customers will accept loan offers, banks can:
-Focus marketing efforts on high-probability customers
-Reduce wasted marketing spend
-Improve customer experience by sending relevant offers only
-Increase overall loan portfolio growth

## Target Variable
**Personal Loan** - Binary classification indicating whether a customer accepted (1) or declined (0) a personal loan offer.

## Dataset Features
The dataset contains 4,948 customer records with the following attributes:
-Demographics: Age, Experience, Family size, ZIP code
-Financial Data: Income, Mortgage amount
-Banking Behavior: Credit card average spending (CCAvg), Securities account, CD account
-Service Usage: Online banking, Credit card ownership
-Education Level: Categorical variable (1, 2, 3)

## Key Challenge: Severe class imbalance - only 9.7% of customers accepted loan offers.

## Methodology
**1. Data Preprocessing**
-Removed records with negative experience values
-Converted credit card spending from monthly to annual (multiplied by 12)
-Standardized column names
-Converted appropriate features to categorical/boolean types

**2. Handling Class Imbalance**
-Applied SMOTETomek (hybrid oversampling + undersampling)
-Resampled distribution: 67% negative class, 33% positive class
-Split: 60% train, 20% validation, 20% test

**3. Model Development**
Implemented and compared three models:
-Logistic Regression (baseline)
-Random Forest (ensemble method)
-XGBoost (gradient boosting)

**4. Hyperparameter Tuning**
-Used RandomizedSearchCV with 5-fold cross-validation on XGBoost
-Optimized parameters: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma

**5. Model Evaluation**
Metrics used: Accuracy, Precision, Recall, F1-score, Confusion Matrix

Results
| Model | Test Accuracy | Precision (Has Loan) | Recall (Has Loan) | F1-Score (Has Loan) |
|-------|--------------|---------------------|------------------|---------------------|
| Logistic Regression | 89.70% | 85.89% | 79.86% | 82.77% |
| Random Forest | 97.89% | 97.71% | 96.61% | 97.16% |
| XGBoost (baseline) | 97.97% | 97.94% | 96.61% | 97.27% |
| **XGBoost (tuned)** | **99.16%** | **98.62%** | **98.85%** | **98.74%** |

## Final Model Performance
Test Accuracy: 99.16%
False Positives: 6 (predicted acceptance but declined)
False Negatives: 4 (predicted decline but accepted)
Total Test Cases: 1,318

## Feature Importance
Top predictive features identified by XGBoost:
-Income
-Education level
-Credit card average spending (CCAvg)
-Family size
-CD account ownership

## Dependencies
pip install numpy pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn
**Required libraries:**
NumPy
Pandas
Scikit-learn
XGBoost
imbalanced-learn
Matplotlib
Seaborn

## Running the Project
-Clone the repository and install dependencies
-Load the dataset (Bank.csv)
-Run the preprocessing pipeline
-Train models and evaluate performance
-Use the final XGBoost model for predictions

## Business Impact
-99.16% accuracy enables confident customer targeting
-Reduces wasted marketing efforts by ~90%
-Improves customer satisfaction through relevant offers
-Increases loan portfolio growth through optimized conversions



## Author
Esther Osikoya



