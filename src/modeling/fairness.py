import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix
)


class FairnessAuditor:

    @staticmethod
    def group_metrics(
        y_true,
        y_prob,
        groups,
        threshold=0.57,
        min_group_size=30
    ):

        data = pd.DataFrame({
            "actual": np.asarray(y_true),
            "probability": np.asarray(y_prob),
            "group": np.asarray(groups)
        })

        data["prediction"] = (
            data["probability"] >= threshold
        ).astype(int)

        results = []

        for group, subset in data.groupby("group"):

            actual = subset["actual"]
            predicted = subset["prediction"]

            tn, fp, fn, tp = confusion_matrix(
                actual,
                predicted,
                labels=[0, 1]
            ).ravel()

            results.append({
                "Group": group,
                "Applicants": len(subset),
                "Actual_Default_Rate": actual.mean(),
                "Predicted_Default_Rate": predicted.mean(),
                "Precision": precision_score(
                    actual, predicted, zero_division=0
                ),
                "Recall": recall_score(
                    actual, predicted, zero_division=0
                ),
                "False_Positive_Rate": (
                    fp / (fp + tn) if (fp + tn) else 0
                ),
                "True_Positive_Rate": (
                    tp / (tp + fn) if (tp + fn) else 0
                ),
                "Small_Group": len(subset) < min_group_size
            })

        return pd.DataFrame(results)

    @staticmethod
    def disparity_report(group_results):

        # Exclude groups below the minimum sample-size requirement
        eligible = group_results[
            ~group_results["Small_Group"]
        ]

        if len(eligible) < 2:
            raise ValueError(
                "At least two sufficiently large groups are required."
            )

        metrics = [
            "Actual_Default_Rate",
            "Predicted_Default_Rate",
            "Precision",
            "Recall",
            "False_Positive_Rate"
        ]

        return (
            eligible[metrics].max()
            - eligible[metrics].min()
        ).rename("Absolute_Group_Difference")