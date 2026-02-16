# Used Car Price Prediction – Kaggle Competition

**Regression | Feature Engineering | Target & Label Encoding | XGBoost Tuning**

## Overview
Independent, from-scratch solution for the Kaggle Used Car Price Prediction competition.  
Goal: Predict used car selling price using features like brand, model, year, mileage, engine specs, etc.  
Built custom parsing, imputation, and encoding pipelines to handle real-world messy data.

- Competition: [Kaggle link – https://www.kaggle.com/competitions/hackathon-qualification/overview
- Final training RMSE (tuned XGBoost, target-encoded): ≈73,146
- Focus: Learning robust preprocessing, feature creation, and model comparison.

## Data
- Source: train.csv & test.csv from Kaggle  
- Rows: \~121k train, \~126k test  
- Target: price (continuous)  
- Key features: brand, model, model_year, milage, fuel_type, engine (parsed), transmission (parsed), ext_col/int_col, accident, clean_title  
- Download: https://www.kaggle.com/competitions/hackathon-qualification/data 
- Sample: sample_train.csv included (first 1000 rows for quick runs)

## Approach
- EDA: Distributions, scatter plots (milage vs price by year), categorical value counts  
- Preprocessing: Mode imputation for missing values; regex parsing for engine (HP, displacement, cylinders, type) & transmission (speeds, type)  
- Feature Engineering: car_age (2023 - model_year), milage_age_interaction, extracted engine/transmission components  
- Encoding:  
  - Low-cardinality: One-hot (fuel_type, accident, transmission_Type)  
  - High-cardinality: Target encoding (brand, model, ext_col, int_col, HP, etc.) vs Label encoding comparison  
- Models: LinearRegression (baseline), XGBRegressor  
- Evaluation: 5-fold cross-validation with RMSE  
- Tuning: GridSearchCV on XGBoost (learning_rate, max_depth, n_estimators, subsample)

## Results
| Model / Encoding     | Avg CV RMSE     | Notes                          |
|----------------------|-----------------|--------------------------------|
| LinearRegression (target) | 74,152       | Baseline                       |
| XGBoost (target)     | \~79,910 (untuned) → 73,146 (tuned) | Best overall                  |
| LinearRegression (label)  | 75,842       |                               |
| XGBoost (label)      | 78,491          | Worse than target encoding     |

Best params (tuned XGBoost): {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 300, 'subsample': 0.9}

## Notebook
[used_car_price_prediction.ipynb](used_car_price_prediction.ipynb)

## How to Run
1. Download full data from Kaggle or use sample_train.csv  
2. Open notebook in Colab/Jupyter  
3. Run all cells (includes data loading, processing, modeling)

## Visuals

![Mil mileage vs Price by Year](mileage%20vs%20Price%20by%20Year.png) 
)  
![Pairplot Numerical Features](Pairplot%20Numerical%20Features.png)

MSc Financial Engineering student | Part of personal data science portfolio
