import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)
from src.modeling.metrics import ClassificationMetrics


class ModelEvaluator:

    @staticmethod
    def evaluate(model, X_valid, y_valid, threshold=0.5):
        # probability of default (class 1)
        y_prob = model.predict_proba(X_valid)[:, 1]

        # Convert probabilities into binary predictions
        y_pred = (y_prob >= threshold).astype(int)

        metrics = ClassificationMetrics.calculate(
            y_valid,
            y_pred,
            y_prob
        )
        
        # evaluation artifacts for this specific threshold
        return {
            "metrics": metrics,
            "confusion_matrix": confusion_matrix(y_valid, y_pred, labels=[0, 1]),
            "probabilities": y_prob,
            "predictions": y_pred
        }

    @staticmethod
    def threshold_analysis(y_true, y_prob, thresholds=None):
        if thresholds is None:
            thresholds = np.arange(0.10, 0.81, 0.05)

        results = []

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_true, y_pred, labels=[0, 1]
            ).ravel()

            results.append({
                "Threshold": round(threshold, 2), # Keeps the DataFrame clean
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0),
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
                "Specificity": tn / (tn + fp) if (tn + fp) else 0,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn
            })
            
        # Return the sweeping analysis as a DataFrame
        return pd.DataFrame(results)

    @staticmethod
    def select_threshold(y_true, y_prob, min_recall=0.60):

        import numpy as np

        thresholds = np.arange(0.10, 0.801, 0.01)

        results = ModelEvaluator.threshold_analysis(
            y_true,
            y_prob,
            thresholds
        )

        eligible = results[
            results["Recall"] >= min_recall
        ]

        if eligible.empty:
            raise ValueError(
                f"No threshold satisfies minimum recall: {min_recall}"
            )

        selected = eligible.sort_values(
            by=["Precision", "Balanced_Accuracy"],
            ascending=False
        ).iloc[0]

        return selected
