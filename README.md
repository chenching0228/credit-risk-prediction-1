# Credit Risk Prediction using Lending Club Data

## Project Overview

This project aims to predict loan default risk using Lending Club’s open dataset, combining structured data preprocessing, feature engineering, and machine learning model optimization. The workflow guides users from exploratory data analysis (EDA) to model fine-tuning and performance evaluation.

## Dataset

- **Source**: Kaggle - Lending Club Dataset
- **Period**: 2007-2018
- **Size**: 2.26 million loan applications
- **Features**: 151 attributes

## Project Structure

```
credit-risk-prediction/
├── data/
│   ├── raw/                 # Original data from Kaggle
│   └── processed/           # Cleaned and engineered features
├── notebooks/               # Jupyter notebooks for analysis
│   └── data_exploration.ipynb
│   └── feature_engineering.ipynb
│   └── modelling.ipynb
│   └── fine_tuning.ipynb
│ 
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation and Setup

## Installation & Environment Setup

### 1. Clone the repository:
```bash
   git clone https://github.com/chenching0228/credit-risk-prediction-1
   cd credit-risk-prediction
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

### 4. Run notebooks sequentially:

1. `data_exploration.ipynb`
2. `feature_engineering.ipynb`
3. `modelling.ipynb`
4. `fine_tuning.ipynb`

## Exploratory Data Analysis

### Analysis Workflow

The notebook follows a structured approach to data exploration and preparation for modeling.

1.  **Data Loading & Initial Assessment**: Loads the accepted  and rejected loan datasets to assess their size, structure, and column overlap.

2.  **Data Quality Assessment**: Performs an initial check for missing values and duplicate records. Columns are categorized by the percentage of missing data, with those having over 80% missingness flagged for removal.

3.  **Stratified Sampling**: To enable faster iteration, a representative sample of 500,000 records is created from the accepted loans dataset. Stratification is performed using `loan_status` and `issue_year` to ensure the sample maintains the original data's temporal patterns and outcome distributions.

4.  **Target Variable Definition**: A binary target variable (`target`) is engineered for the classification task:
    * **Good Loans (0)**: `Fully Paid`
    * **Defaulted Loans (1)**: `Charged Off`, `Default`, `Late (31-120 days)`
    * Loans with indeterminate statuses (e.g., `Current`, `In Grace Period`) are excluded from the final modeling dataset to prevent label noise.

5.  **Feature Engineering & Cleaning**: Raw data is transformed into model-ready features. Key steps include:
    * Parsing date fields (`issue_d`, `earliest_cr_line`) to calculate features like `credit_history_years`.
    * Cleaning financial metrics (`int_rate`, `revol_util`) by removing symbols and converting them to numeric types.
    * Standardizing categorical features like `emp_length` into a numeric format.
    * Creating new features like `loan_to_income_ratio`.

6.  **Missing Value Treatment**: After dropping columns with >80% missing data, remaining missing values in key numeric features (`emp_length_numeric`, `mort_acc`, `dti`, etc.) are imputed using the median.

7.  **Exploratory Data Analysis (EDA)**: The core analysis phase investigates relationships and patterns in the data.
    * **Bivariate Analysis**: Examines the relationship between individual features and the default rate. For example, it confirms a clear increase in default rate as the loan `grade` worsens (from A to G).
    * **Correlation Analysis**: A heatmap is used to identify multicollinearity. A high correlation (>0.95) is found between `loan_amnt` and `installment`.
    * **Segmentation Analysis**: Default rates are analyzed across different segments:
        * **Temporal**: A clear trend shows default rates increasing in the years following the 2008 financial crisis.
        * **Geographic**: Default rates are compared across the top 10 states by loan volume.


### Key Findings

* **Risk Stratification**: There is a clear risk stratification across loan grades, with grade 'G' loans having a default rate (~51.6%) significantly higher than grade 'A' loans (~6.8%).
* **Temporal Patterns**: Default rates show clear temporal trends, with a noticeable spike in the years following the 2008-2010 financial crisis.
* **Predictive Features**: Features like interest rate (`int_rate`), debt-to-income ratio (`dti`), and credit history length (`credit_history_years`) show strong relationships with loan default outcomes.
* **Class Imbalance**: The modeling dataset has a class imbalance of approximately 3.7:1 (Good Loans vs. Defaulted Loans), which must be addressed during modeling.
* **Multicollinearity**: A high correlation exists between `loan_amnt` and `installment`, suggesting one might be redundant for some models.

---

### Modelling Recommendations

Based on the EDA, the following recommendations are made for model development:
* **Model Selection**: Start with a Logistic Regression baseline for interpretability, and explore more powerful ensemble models like Random Forest, XGBoost, and LightGBM.
* **Class Imbalance**: Use techniques such as SMOTE, class weighting in models, or adjusting the decision threshold to handle the imbalance.
* **Validation Strategy**: Employ a time-based split (e.g., train on 2007-2016, test on 2017-2018) to prevent data leakage and ensure the model generalizes to future data.
* **Evaluation Metrics**: Focus on ROC-AUC for overall performance, but also heavily consider the Precision-Recall curve and F1-Score due to the class imbalance.


## Feature Engineering

**Methodology**

The analysis employed several feature selection techniques to provide a comprehensive evaluation:

* **Variance Analysis:** Features with low variability were identified to eliminate those providing minimal predictive information.
* **Correlation Analysis:** A correlation matrix was generated to identify highly correlated feature pairs (e.g., `loan_amnt` and `installment`). This helps reduce multicollinearity. The correlation of each feature with the target variable was also assessed.
* **Univariate Feature Importance:** Statistical tests, including Mutual Information and ANOVA F-statistic ($F$-statistic), were used to measure the strength of the relationship between individual features and the loan default target.
* **Tree-Based Feature Importance:** A `RandomForestClassifier` was trained on a sample of the data to calculate feature importance scores, capturing non-linear relationships and interactions.
* **Principal Component Analysis (PCA):** PCA was applied to the scaled numeric features to explore dimensionality reduction. The analysis showed that just 2 principal components could explain over 90% of the variance in the dataset.

**Recommendations & Output**

Based on a consolidated ranking from all methods, the analysis produced several recommended feature sets for different modeling scenarios:

* **`minimal_set` (3 features):** Contains only the most critical predictors like `int_rate_clean` and `grade` for rapid prototyping.
* **`recommended_set` (4 features):** A balanced set including critical and strong features, ideal for production models.
* **`comprehensive_set` (4 features):** A more extensive set for models where maximizing accuracy is the primary goal.

The final feature importance rankings and the recommended feature sets were exported to `data/processed/` for use in the modeling phase.

## Modelling

This section covers the data preparation, model training, evaluation, and comparison for predicting credit risk.

---

### Model Preparation

The dataset is split into three sets for training, validation, and testing to ensure a robust evaluation process.

* **Training Set**: 211,610 samples (70%)
* **Validation Set**: 45,345 samples (15%)
* **Test Set**: 45,346 samples (15%)

Preprocessing steps include using `LabelEncoder` for categorical variables and extracting year/month components from datetime features, resulting in a total of 26 features.

---

#### Model Training & Evaluation

Three different classification models were trained and evaluated:

1.  **Logistic Regression**: Class imbalance was handled using the Synthetic Minority Over-sampling Technique (SMOTE). Features were scaled using `StandardScaler`.
2.  **Random Forest**: This model was trained with the `class_weight='balanced'` parameter to manage class imbalance without resampling the data.
3.  **XGBoost**: The `scale_pos_weight` parameter was used to address the imbalanced classes, a common and effective technique for gradient boosting models.

The performance of each model on the validation set is summarized below:

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.6775 | 0.3257 | 0.4855 | 0.3899 | 0.6572 |
| **Random Forest** | 0.6580 | 0.3406 | 0.6534 | 0.4478 | 0.7171 |
| **XGBoost** | 0.6586 | 0.3435 | 0.6684 | 0.4538 | 0.7243 |

---

#### Model Comparison

Based on the validation results, **XGBoost** was selected as the best-performing model, achieving the highest ROC AUC score of **0.7243**.





---

#### Feature Importance

Feature importance analysis was conducted for all models. The XGBoost model identified `grade` as the most influential predictor of loan default, followed by `term_months` and `sub_grade`.


## Model Fine-Tuning and Evaluation

This section details the process of fine-tuning, evaluating, and comparing models for credit risk prediction. The key steps include feature set comparison, hyperparameter optimization for Random Forest and Logistic Regression, handling class imbalance with SMOTE, and a comprehensive final model evaluation.

---

### 1. Feature Set Comparison

To identify the most effective and efficient feature configuration, a baseline Random Forest model was evaluated across various pre-selected feature sets using 5-fold cross-validation.

**Results:**
The `minimal_set` and `tier1_features`, both containing only 3 features, achieved the highest mean AUC, indicating that a smaller, more focused feature set is most effective for the baseline model.

```text
         feature_set  n_features  mean_auc   std_auc
0        minimal_set           3  0.613980  0.011790
3     tier1_features           3  0.613980  0.011790
1    recommended_set           4  0.611486  0.013417
2  comprehensive_set           4  0.611486  0.013417
5     tier4_features          22  0.503461  0.008517
4     tier2_features           1  0.490788  0.008175
```

---

### 2. Hyperparameter Optimization

Different optimization methods are suited for different scenarios. This notebook uses Randomized Search for the Random Forest and Grid Search for the Logistic Regression.

| Method                | Mechanism                                        | Advantages                                     | Disadvantages                                  | Use When                                     |
| --------------------- | ------------------------------------------------ | ---------------------------------------------- | ---------------------------------------------- | -------------------------------------------- |
| **Grid Search** | Exhaustive search over specified parameter grid  | Guaranteed to find best combination in grid    | Computationally expensive, curse of dimensionality | Small parameter space, need comprehensive search |
| **Random Search** | Random sampling from parameter distributions     | More efficient than grid for large spaces      | May miss optimal combination                   | Large parameter space, limited compute budget    |
| **Bayesian Optimisation** | Builds probabilistic model of objective function | Efficient for expensive evaluations            | Complex implementation, requires careful tuning | Expensive model training, need sample efficiency |

#### Random Forest Tuning
`RandomizedSearchCV` was used for its efficiency over large parameter spaces.

- **Best AUC:** $0.6453$
- **Best Parameters:**
    ```json
    {
        "bootstrap": true,
        "class_weight": "balanced",
        "max_depth": 20,
        "max_features": "log2",
        "min_samples_leaf": 77,
        "min_samples_split": 142,
        "n_estimators": 411
    }
    ```

#### Logistic Regression Tuning
`GridSearchCV` was used for a comprehensive search of the smaller parameter space, with a focus on regularization (`C`) and solver choice. **Note:** Data was scaled using `StandardScaler` before tuning.

- **Best AUC:** $0.6487$
- **Best Parameters:**
    ```json
    {
        "C": 0.001,
        "class_weight": "balanced",
        "max_iter": 5000,
        "penalty": "l2",
        "solver": "lbfgs"
    }
    ```

---

### 3. Handling Class Imbalance with SMOTE

The Synthetic Minority Over-sampling Technique (SMOTE) was applied to the training data to address class imbalance by generating synthetic samples for the minority class.

- **Original Class Distribution:** `Counter({0: 5185, 1: 1815})` (Ratio: 2.86:1)
- **Resampled Class Distribution:** `Counter({0: 5185, 1: 2592})` (Ratio: 2.00:1)

---

### 4. Final Model Evaluation

The tuned models were evaluated on the held-out test set. A key finding was the importance of using appropriate data scaling for different model types.

#### Issue Resolution: Feature Scaling
- **Problem:** The Random Forest model initially showed poor performance.
- **Root Cause:** It was being evaluated on **scaled data**, but tree-based models like Random Forest do not require feature scaling. Linear models like Logistic Regression, however, do.
- **Solution:** The evaluation pipeline was corrected to use **unscaled data for the Random Forest** and **scaled data for the Logistic Regression**, leading to valid performance metrics.

#### Comparison Results

The tuned Logistic Regression model slightly outperformed the Random Forest model in overall AUC, precision, and F1-score on the test set.

| model    |   roc_auc |   precision |   recall |       f1 |
|:---------|----------:|------------:|---------:|---------:|
| lr_tuned |  0.621138 |    0.341060 | 0.529563 | 0.414904 |
| rf_tuned |  0.609553 |    0.335616 | 0.503856 | 0.402878 |

![Model Performance Comparison](attachment:image.png)

**Evaluation Summary:**
```text
LR_TUNED:
  ROC AUC:    0.6211
  Precision:  0.3411 (206/604 positive predictions correct)
  Recall:     0.5296 (206/389 actual positives identified)
  F1 Score:   0.4149 (harmonic mean of precision & recall)

RF_TUNED:
  ROC AUC:    0.6096
  Precision:  0.3356 (196/584 positive predictions correct)
  Recall:     0.5039 (196/389 actual positives identified)
  F1 Score:   0.4029 (harmonic mean of precision & recall)
```

---

### 5. Recommendations & Next Steps

Based on the evaluation, several paths for further improvement are recommended:
1.  **Try XGBoost:** This model often outperforms both LR and RF on tabular data.
2.  **Advanced Feature Engineering:** Create polynomial or interaction terms for LR and use feature importance from RF for selection.
3.  **Threshold Optimization:** The default 0.5 threshold is not always optimal for business needs. Adjusting the threshold can balance the trade-off between approving safer loans (higher precision) and catching more defaults (higher recall). An analysis showed that a threshold of **0.40** maximizes the F1 Score (0.4259).
4.  **Ensemble Methods:** Combine predictions from multiple models (e.g., voting, stacking) to improve robustness.
5.  **Advanced Imbalance Handling:** Experiment with other techniques like ADASYN or SMOTEENN.


## Conclusions and Recommendations

[To be completed]

## Future Work

## Future Work

While the current model provides a solid baseline, several avenues exist for further improvement and exploration.

---

### Model and Feature Enhancements

* **Full XGBoost Pipeline**: The initial modeling phase showed that XGBoost was a very promising candidate, achieving the highest validation AUC (0.7243). A key next step is to perform a full hyperparameter tuning and evaluation cycle on the XGBoost model to see if it can outperform the tuned Logistic Regression on the final test set.
* **Advanced Feature Engineering**: Incorporate external macroeconomic indicators (e.g., unemployment rates, GDP growth) corresponding to the loan's issue date. Given the temporal patterns observed in the EDA, this could significantly boost predictive power. Creating polynomial and interaction features could also capture more complex relationships, especially for the Logistic Regression model.
* **Alternative Imbalance Handling**: Experiment with more advanced over/under-sampling techniques like ADASYN, which adaptively generates minority samples, or combination methods like SMOTEENN, which combines over-sampling with data cleaning.

## References

- Lending Club Dataset: https://www.kaggle.com/datasets/wordsforthewise/lending-club

## Author

[Ching Chen]
[2025/10/12]