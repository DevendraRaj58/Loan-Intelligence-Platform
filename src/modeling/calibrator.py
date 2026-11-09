from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss, log_loss


class ProbabilityCalibrator:

    @staticmethod
    def fit(model, X_train, y_train, method="sigmoid", cv=5):

        cross_validation = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=42
        )

        calibrated_model = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv=cross_validation,
            ensemble=True,
            n_jobs=1
        )

        calibrated_model.fit(X_train, y_train)

        return calibrated_model

    @staticmethod
    def evaluate(y_true, y_prob):

        return {
            "Brier_Score": brier_score_loss(y_true, y_prob),
            "Log_Loss": log_loss(y_true, y_prob)
        }