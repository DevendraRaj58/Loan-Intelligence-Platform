import numpy as np
import pandas as pd
import shap


class SHAPExplainer:

    def __init__(self, model, feature_names):

        self.model = model
        self.feature_names = list(feature_names)

        # Initialize Tree SHAP for LightGBM
        self.explainer = shap.TreeExplainer(
            model,
            feature_names=self.feature_names,
            feature_perturbation="tree_path_dependent"
        )

        self.explanation = None

    @staticmethod
    def _select_positive_class(shap_values):

        # Handle older SHAP versions
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        # Handle multi-output classification
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        return shap_values

    def _get_base_value(self):

        base_value = np.asarray(self.explainer.expected_value)

        if base_value.size > 1:
            base_value = base_value.reshape(-1)[-1]

        return float(base_value)

    def explain(self, X, sample_size=1000, random_state=42):

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("Input must be a 2D feature matrix.")

        if X.shape[1] != len(self.feature_names):
            raise ValueError("Feature count does not match feature names.")

        # Use a representative sample to control computation
        if sample_size is not None and len(X) > sample_size:

            rng = np.random.default_rng(random_state)

            indices = rng.choice(
                len(X),
                size=sample_size,
                replace=False
            )

            X = X[indices]

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(
            X,
            check_additivity=True
        )

        shap_values = self._select_positive_class(shap_values)

        # Store explanations with feature names
        self.explanation = shap.Explanation(
            values=shap_values,
            base_values=np.full(
                len(X),
                self._get_base_value()
            ),
            data=X,
            feature_names=self.feature_names
        )

        return self.explanation

    def feature_importance(self):

        if self.explanation is None:
            raise ValueError("Run explain() before calculating importance.")

        importance = np.abs(
            self.explanation.values
        ).mean(axis=0)

        return (
            pd.DataFrame({
                "feature": self.feature_names,
                "mean_abs_shap": importance
            })
            .sort_values(
                by="mean_abs_shap",
                ascending=False
            )
            .reset_index(drop=True)
        )

    def plot_summary(self, max_display=20):

        if self.explanation is None:
            raise ValueError("Run explain() before plotting.")

        shap.plots.beeswarm(
            self.explanation,
            max_display=max_display
        )

    def plot_bar(self, max_display=20):

        if self.explanation is None:
            raise ValueError("Run explain() before plotting.")

        shap.plots.bar(
            self.explanation,
            max_display=max_display
        )

    def explain_instance(self, X_row):

        X_row = np.asarray(X_row)

        if X_row.ndim == 1:
            X_row = X_row.reshape(1, -1)

        if X_row.shape != (1, len(self.feature_names)):
            raise ValueError("Expected exactly one applicant with all features.")

        shap_values = self.explainer.shap_values(
            X_row,
            check_additivity=True
        )

        shap_values = self._select_positive_class(shap_values)

        return shap.Explanation(
            values=shap_values[0],
            base_values=self._get_base_value(),
            data=X_row[0],
            feature_names=self.feature_names
        )

    @staticmethod
    def plot_waterfall(instance_explanation, max_display=15):

        shap.plots.waterfall(
            instance_explanation,
            max_display=max_display
        )