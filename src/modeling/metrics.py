from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)


class ClassificationMetrics:

    @staticmethod
    def calculate(y_true, y_pred, y_prob):

        return {
            "ROC_AUC": roc_auc_score(y_true, y_prob),
            "PR_AUC": average_precision_score(y_true, y_prob),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Balanced_Accuracy": balanced_accuracy_score(
                y_true, y_pred
            )
        }