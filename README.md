# House Price Prediction — ML Regression Model

An end-to-end machine learning pipeline to predict house prices using the Ames Housing dataset. Built with scikit-learn, Pandas, and NumPy.

## Tech Stack

- **Language:** Python 3
- **Libraries:** scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Environment:** Google Colab / Jupyter Notebook
- **Dataset:** Ames Housing Dataset (Kaggle)

## Pipeline Overview

1. **Exploratory Data Analysis (EDA)** — distribution plots, correlation heatmap, outlier detection
2. **Data Preprocessing** — IQR-based outlier removal, log transformation of skewed features, encoding categorical variables
3. **Model Training** — Random Forest Regressor with GridSearchCV for hyperparameter tuning
4. **Model Comparison** — compared Linear Regression, Decision Tree, and Random Forest
5. **Evaluation** — RMSE and R² score on held-out test set

## Results

| Model | RMSE | R² Score |
|-------|------|----------|
| Linear Regression | - | - |
| Decision Tree | - | - |
| Random Forest | - | - |

*(Results updated after running the notebook)*

## Key Concepts Applied

- **IQR outlier removal** — removed data points beyond 1.5×IQR to reduce noise
- **Log transformation** — applied to right-skewed target variable (SalePrice) to improve model fit
- **GridSearchCV** — exhaustive hyperparameter search with 5-fold cross-validation
- **Feature encoding** — one-hot encoding for nominal categories, label encoding for ordinal

## How to Run

```bash
# Option 1: Open in Google Colab
# Upload the .ipynb file and run all cells

# Option 2: Local
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook house_price_prediction.ipynb
```

## What I Learned

- Why log transformation helps with skewed regression targets
- How Random Forest reduces overfitting compared to a single Decision Tree
- What GridSearchCV actually does under the hood — cross-validated grid search
- How to evaluate regression models beyond just accuracy
Copy
