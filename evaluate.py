"""Reproducible Titanic evaluation with fold-local preprocessing and held-out test."""
import json
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / 'data/titanic.csv')
# prepare_dataset.py appends copies of the first 15 passengers as an exercise.
# Remove only those artificial rows; identical real passenger records can be valid.
if len(df) == 906 and df.iloc[-15:].reset_index(drop=True).equals(df.iloc[:15].reset_index(drop=True)):
    df = df.iloc[:-15].copy()
assert len(df) == 891, 'Unexpected source: check original Titanic passenger data.'
num = ['age', 'fare', 'sibsp', 'parch']
cat = ['pclass', 'sex', 'embarked']
X, y = df[num + cat].copy(), df['survived']
for c in num:
    X[c] = pd.to_numeric(X[c], errors='coerce')
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=.25, stratify=y_trainval, random_state=42)
def pipeline(model):
    pre = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat),
    ])
    return Pipeline([('preprocess', pre), ('model', model)])
models = {
    'Logistic Regression': (LogisticRegression(max_iter=2000, random_state=42), {'model__C': [.1, 1, 10]}),
    'Decision Tree': (DecisionTreeClassifier(random_state=42), {'model__max_depth': [3, 5, 8], 'model__min_samples_leaf': [1, 5]}),
    'Random Forest': (RandomForestClassifier(random_state=42, n_jobs=1), {'model__n_estimators': [50, 100, 200], 'model__max_depth': [5, 10, None], 'model__min_samples_leaf': [1, 5]}),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=42), {'model__n_estimators': [50, 100], 'model__learning_rate': [.05, .1], 'model__max_depth': [2, 3]}),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows, searches = [], {}
for name, (model, params) in models.items():
    search = GridSearchCV(pipeline(model), params, cv=cv, scoring='roc_auc', n_jobs=2)
    search.fit(X_train, y_train)
    row = {'model': name, 'cv_auc': search.best_score_, 'validation_auc': roc_auc_score(y_val, search.predict_proba(X_val)[:, 1]), 'parameters': search.best_params_}
    rows.append(row)
    searches[name] = search
    print(json.dumps(row), flush=True)
selected = max(rows, key=lambda row: row['validation_auc'])['model']
final = searches[selected].best_estimator_
final.fit(X_trainval, y_trainval)
pred, prob = final.predict(X_test), final.predict_proba(X_test)[:, 1]
report = {'dataset_rows': len(df), 'split': {'train': len(X_train), 'validation': len(X_val), 'test': len(X_test)}, 'comparison': rows, 'selected_model': selected, 'test': {'roc_auc': roc_auc_score(y_test, prob), 'accuracy': accuracy_score(y_test, pred), 'precision': precision_score(y_test, pred), 'recall': recall_score(y_test, pred), 'f1': f1_score(y_test, pred)}, 'limitations': 'Single seeded holdout; educational benchmark, not production validation. No passenger IDs available for duplicate identity checks.'}
(ROOT / 'results/evaluation_clean.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
