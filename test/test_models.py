#test model evaluator
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn.model_selection import train_test_split

from analysis.models.core_logistic_regression_lasso import LassoLogisticRegression
from analysis.models.evaluator import evaluate_classification

# load data
df = pd.read_csv('data/processed/train_set_with_features.csv')

exclude_cols = ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols].values
y = df['toxic'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LassoLogisticRegression()
model.fit(X_train, y_train)

metrics = evaluate_classification(
    y_true=y_test,
    y_pred=model.predict(X_test),
    y_score=model.predict_proba(X_test)[:, 1],
    name="Lasso"
)