from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class ModelFactory:

    @staticmethod
    def create(model_name, y_train=None):

        # Calculate imbalance ratio from training labels
        scale_pos_weight = 1.0

        if y_train is not None:
            positive_count = int((y_train == 1).sum())
            negative_count = int((y_train == 0).sum())

            if positive_count > 0:
                scale_pos_weight = negative_count / positive_count

        if model_name == "logistic_regression":

            return LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )

        elif model_name == "random_forest":

            return RandomForestClassifier(
                n_estimators=200,
                max_depth=18,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            )

        elif model_name == "xgboost":

            return XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                tree_method="hist",
                random_state=42,
                n_jobs=-1
            )

        elif model_name == "lightgbm":

            return LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )

        elif model_name == "catboost":

            return CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="AUC",
                auto_class_weights="Balanced",
                random_seed=42,
                verbose=False,
                thread_count=-1
            )

        else:
            raise ValueError(f"Unsupported model: {model_name}")