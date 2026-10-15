# Titanic Survival Prediction

Educational ML benchmark by Praveen Bala, extended from [my AI/ML capstone](https://github.com/praveensri2018/praveen-bala-capstone-project). The evaluation refactor and documentation were prepared with AI assistance; this is not a production system.

## Run

Use Python 3.11 or newer in a virtual environment:

```bash
pip install -r requirements.txt
python evaluate.py
```

The script prints model comparisons and writes `results/evaluation_clean.json`.

## Data and preprocessing

Source: [Seaborn Titanic dataset](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv), copied from the original capstone exercise. The checked-in CSV contains 906 rows: 891 source passengers plus 15 deliberately appended exercise duplicates. The script verifies and removes those 15 appended rows only. Identical original passenger records are retained because there is no passenger ID to establish duplicate identities.

Seven predictors: age, fare, sibsp, parch, pclass, sex and embarked. The target is survived. Target-equivalent alive and redundant derived fields are excluded. Numeric median imputation, categorical mode imputation, one-hot encoding and numeric scaling are fitted inside each training fold through ColumnTransformer and Pipeline. Scaling is shared for comparability; tree models do not require it.

## Evaluation protocol

- Seed 42; stratified split: 534 training, 178 validation, 179 final test passengers.
- Tune four model families using 5-fold stratified cross-validation on training data only, optimizing ROC-AUC.
- Search 3 Logistic Regression, 6 Decision Tree, 18 Random Forest and 8 Gradient Boosting configurations.
- Select the family using validation ROC-AUC, then refit its selected configuration on all 712 training-plus-validation rows.
- Evaluate the final model once on the held-out test partition, using the default classification threshold of 0.5.

## Recorded results

| Model | Best training CV ROC-AUC | Validation ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.8438 | 0.8641 |
| Decision Tree | 0.8485 | 0.8573 |
| Random Forest | 0.8663 | 0.8861 |
| Gradient Boosting | 0.8698 | 0.9050 |

Selected Gradient Boosting: 100 estimators, learning rate 0.1, max depth 2.

| Final test metric | Value |
|---|---:|
| ROC-AUC | 0.8193 |
| Accuracy | 79.33% |
| Precision, survival class | 0.8200 |
| Recall, survival class | 0.5942 |
| F1, survival class | 0.6891 |

## Interpretation and limitations

The lower final test AUC compared with validation AUC illustrates split uncertainty. Survival-class recall is only 59.4%; accuracy alone hides missed survivors. These are results from one fixed educational benchmark split, not evidence of production performance or causal explanations. CV scores are tuning estimates, not an independent final assessment. No deployment, model monitoring, original model research or production business impact is claimed. No final model is serialized in this version. Future threshold selection should use training/validation data only; further development should reserve a fresh final evaluation set.
