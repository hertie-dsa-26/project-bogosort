import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from analysis.pipeline_and_dispatch.data_pipeline import DataPipeline
from analysis.evaluation_code.evaluator import evaluate_classification
from analysis.features.build_features import FeatureBuilder, FeaturePreprocessor


OUTPUT_DIR = "analysis/models/all_outputs/lasso_log_reg"


def run(data_path, mode="train", save_predictions=True, save_model=True):
    fb = FeatureBuilder()

    if mode == "train":
        dp = DataPipeline(data_path, label_columns=["toxic"])
        X_train, X_test, y_train, y_test = dp.get_data()

        if os.path.exists(fb.tfidf_path):
            fb.load()
        else:
            print("Fitting TF-IDF vectorizers...")
            fb.fit(X_train)

        print("Transforming train features...")
        X_train_feat = fb.transform(X_train, split="train")
        print("Transforming test features...")
        X_test_feat = fb.transform(X_test, split="test")

        preprocessor = FeaturePreprocessor()
        print("Preprocessing features...")
        X_train_proc = preprocessor.fit_transform(X_train_feat)
        X_test_proc  = preprocessor.transform(X_test_feat)

        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=1.0,
            max_iter=1000,
            random_state=42,
        )

        pipeline = Pipeline([("model", model)])

        print("Fitting pipeline...")
        y_train = y_train.values.ravel()
        pipeline.fit(X_train_proc, y_train)
        y_pred = pipeline.predict(X_test_proc)

        y_test = y_test.values.ravel()

        metrics = evaluate_classification(
            y_test,
            y_pred,
            None,
            name="Lasso Baseline (LogReg L1)",
        )

        if save_predictions:
            os.makedirs(os.path.join(OUTPUT_DIR, "predictions"), exist_ok=True)
            pd.DataFrame({"true": y_test, "pred": y_pred}).to_csv(
                os.path.join(OUTPUT_DIR, "predictions", "lasso_sklearn_baseline_predictions.csv"),
                index=False,
            )

        if save_model:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(os.path.join(OUTPUT_DIR, "lasso_sklearn_baseline.pkl"), "wb") as f:
                pickle.dump(pipeline, f)

    elif mode == "infer":
        model_path = os.path.join(OUTPUT_DIR, "lasso_sklearn_baseline.pkl")
        if not os.path.exists(model_path):
            raise ValueError("No trained model found.")

        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)

        fb.load()

        dp = DataPipeline(data_path, label_columns=["toxic"])
        X, y_test = dp.get_infer_data(infer_path="data/processed/test_data.pkl")

        X_feat = fb.transform(X, split="test")
        print("Preprocessing...")
        preprocessor = FeaturePreprocessor()
        X_proc = preprocessor.transform(X_feat)
        y_pred = pipeline.predict(X_proc)
        y_test = y_test.values.ravel()

        if save_predictions:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            pd.DataFrame({"pred": y_pred}).to_csv(
                os.path.join(OUTPUT_DIR, "lasso_sklearn_baseline_infer.csv"),
                index=False,
            )
    else:
        raise ValueError("mode must be 'train' or 'infer'")
