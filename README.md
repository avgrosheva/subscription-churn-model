# Customer Churn Prediction

A machine learning project for predicting customer churn in a telecom service and identifying customers who may benefit from proactive retention actions.

The project combines exploratory data analysis, an interpretable Logistic Regression baseline, CatBoost classification, and decision-threshold optimization with a focus on detecting more customers at risk of churn.

## Business Problem

Customer churn directly affects recurring revenue and increases the need for customer acquisition.

The goal of this project is to identify customers with a high probability of leaving the service within the next month so that retention efforts can be targeted before they churn.

For this use case, missing an actual churner can be more costly than incorrectly flagging a customer as high-risk. Therefore, model evaluation focuses not only on overall predictive performance, but also on **recall for the churn class**.

## Dataset

The project uses the public **Telco Customer Churn** dataset.

- 7,043 customers
- 20 input features
- Target: `Churn`
- Churn rate: ~26.5%

The features describe customer demographics, subscribed services, contract characteristics, tenure, payment methods, and charges.

The target is imbalanced, with churners representing approximately one quarter of the dataset.

## Approach

The analysis follows five main stages:

1. **Exploratory Data Analysis**
   - examined feature distributions and churn patterns;
   - compared churn across customer groups;
   - identified characteristics associated with higher churn.

2. **Data Preparation**
   - prepared numerical and categorical variables for modeling;
   - handled the class imbalance during model evaluation.

3. **Baseline Model**
   - trained Logistic Regression as an interpretable benchmark.

4. **CatBoost Model**
   - trained CatBoostClassifier to capture nonlinear relationships and interactions between customer characteristics.

5. **Threshold Optimization**
   - evaluated classification thresholds rather than relying only on the default 0.50 cutoff;
   - selected a threshold of **0.35** to improve churn detection while maintaining reasonable overall model performance.

## Results

| Model | F1 | ROC-AUC | Recall — Churn |
|---|---:|---:|---:|
| Logistic Regression | 0.61 | 0.842 | 0.56 |
| **CatBoost — threshold 0.35** | **0.64** | **0.846** | **0.73** |

CatBoost provided a modest improvement in ROC-AUC and F1 compared with the Logistic Regression baseline.

The more important improvement for the retention use case came from threshold optimization: churn recall increased from **0.56 to 0.73**.

This means the final classification setup identifies a substantially larger share of customers who actually churn.

## Churn Patterns

Exploratory analysis showed higher churn among customers who:

- have shorter tenure;
- use month-to-month contracts;
- have higher monthly charges;
- do not subscribe to Online Security;
- do not subscribe to Tech Support;
- pay via electronic check;
- use fiber-optic internet service.

These patterns provide additional context for customer segmentation and retention strategies beyond individual model predictions.

## Feature Importance

The most influential CatBoost features include:

1. `Contract`
2. `InternetService`
3. `tenure`
4. `TotalCharges`
5. `MonthlyCharges`
6. `PaymentMethod`
7. `OnlineSecurity`

The results suggest that churn risk is associated not only with customer tenure and pricing, but also with contract structure and the combination of services used.

## Business Interpretation

The model can be used to rank customers by churn risk and support targeted retention campaigns.

Rather than treating the classification threshold as a purely technical parameter, it can be adjusted according to the economics of a retention campaign.

For example, when the cost of contacting an additional customer is relatively low compared with the cost of losing a subscriber, prioritizing recall can be preferable to using the default classification threshold.

The selected **0.35 threshold** reflects this trade-off by reducing false negatives and identifying more potential churners.

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- CatBoost
- Matplotlib
- Seaborn
- Jupyter Notebook

## Repository Structure

```text
subscription-churn-model/
├── data/
│   └── telco_churn.csv
├── notebooks/
│   ├── cat_boost_info/
│   └── 01_eda_baseline_catboost.ipynb
├── requirements.txt
└── README.md
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/avgrosheva/subscription-churn-model.git
cd subscription-churn-model
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Open the analysis notebook:

```text
notebooks/01_eda_baseline_catboost.ipynb
```

Run the notebook from top to bottom to reproduce the exploratory analysis, model training, evaluation, and threshold comparison.

## Key Takeaway

The project demonstrates how a churn model can be evaluated as a **business decision tool rather than only a classification problem**.

While CatBoost slightly improves overall predictive performance over the Logistic Regression baseline, the main practical gain comes from adapting the classification threshold to the retention use case, increasing churn recall from **0.56 to 0.73**.
