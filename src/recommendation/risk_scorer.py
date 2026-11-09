import numpy as np
import pandas as pd


class LoanRiskScorer:

    def __init__(
        self,
        model,
        preprocessor,
        feature_mask,
        input_columns,
        threshold
    ):

        self.model = model
        self.preprocessor = preprocessor
        self.feature_mask = np.asarray(
            feature_mask, dtype=bool
        )
        self.input_columns = list(input_columns)
        self.threshold = float(threshold)

        if not 0 < self.threshold < 1:
            raise ValueError("Threshold must be between 0 and 1.")

    def _validate_input(self, applicant_features):

        if not isinstance(applicant_features, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if applicant_features.empty:
            raise ValueError("Applicant data cannot be empty.")

        # Target must never be used as a prediction input
        if "TARGET" in applicant_features.columns:
            raise ValueError(
                "TARGET must not be included in prediction inputs."
            )

        # Remove applicant identifier if supplied
        applicant_features = applicant_features.drop(
            columns=["SK_ID_CURR"],
            errors="ignore"
        )

        missing_columns = [
            col for col in self.input_columns
            if col not in applicant_features.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required features: {missing_columns[:10]}"
            )

        # Keep exactly the training input columns in the correct order
        return applicant_features[self.input_columns]

    def score(self, applicant_features):

        # Validate and align incoming features
        X = self._validate_input(applicant_features)

        # Apply the fitted training preprocessor
        X_processed = self.preprocessor.transform(X)

        if X_processed.shape[1] != len(self.feature_mask):
            raise ValueError(
                "Transformed feature count does not match feature mask."
            )

        # Remove gender-derived features
        X_selected = X_processed[:, self.feature_mask]

        # Obtain model-estimated default scores
        scores = self.model.predict_proba(X_selected)[:, 1]

        # Apply operating threshold
        predictions = (scores >= self.threshold).astype(int)

        return pd.DataFrame(
    {
        "estimated_default_probability": scores,
        "decision_threshold": self.threshold,
        "above_threshold": predictions.astype(bool),
        "model_signal": np.where(
            predictions == 1,
            "Above threshold",
            "Below threshold"
        )
    },
    index=X.index
)