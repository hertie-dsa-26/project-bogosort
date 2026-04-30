import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import pickle
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from analysis.pipeline_and_dispatch.data_pipeline import DataPipeline
from analysis.evaluation_code.evaluator import evaluate_classification
from analysis.features.build_features import DenseFeatureTransformer, TfidfTransformer, BertTransformer


OUTPUT_DIR = "analysis/models/all_outputs/baseline"


def run(data_path, save_predictions=True, save_model=True):
    dp = DataPipeline(data_path, label_columns=["toxic"])
    X_train, X_test, y_train, y_test = dp.get_data()

    model = DummyClassifier(strategy="stratified", random_state=42)

    pipeline = Pipeline([
        ("dense", DenseFeatureTransformer()),
        ("baseline_model", model)
    ])

    X_train_dummy = pd.DataFrame([[0]] * len(y_train), columns=['comment_text'])
    X_test_dummy  = pd.DataFrame([[0]] * len(y_test),  columns=['comment_text'])

    pipeline.fit(X_train_dummy, y_train)
    y_pred = pipeline.predict(X_test_dummy)

    metrics = evaluate_classification(
        y_test,
        y_pred,
        name="Dummy Baseline"
    )

    if save_predictions:
        os.makedirs(os.path.join(OUTPUT_DIR, "predictions"), exist_ok=True)
        pd.DataFrame({"true": y_test, "pred": y_pred}).to_csv(
            os.path.join(OUTPUT_DIR, "predictions", "dummy_predictions.csv"),
            index=False,
        )

    if save_model:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "dummy_baseline.pkl"), "wb") as f:
            pickle.dump(model, f)

    return metrics
