import numpy as np
import pandas as pd


class RiskCategorizer:

    def __init__(
        self,
        low_risk_upper=0.05,
        elevated_risk_threshold=0.11
    ):

        if not 0 <= low_risk_upper < elevated_risk_threshold <= 1:
            raise ValueError("Invalid risk category boundaries.")

        self.low_risk_upper = low_risk_upper
        self.elevated_risk_threshold = elevated_risk_threshold

    def categorize(self, probabilities):

        values = np.asarray(probabilities, dtype=float).reshape(-1)

        if not np.isfinite(values).all():
            raise ValueError("Probabilities must be finite.")

        if ((values < 0) | (values > 1)).any():
            raise ValueError("Probabilities must be between 0 and 1.")

        categories = np.select(
            [
                values < self.low_risk_upper,
                values < self.elevated_risk_threshold
            ],
            [
                "LOW",
                "MODERATE"
            ],
            default="ELEVATED"
        )

        return categories

    def transform(self, scores):

        probability_column = "estimated_default_probability"

        if probability_column not in scores.columns:
            raise ValueError(
                f"Missing column: {probability_column}"
            )

        result = scores.copy()

        result["risk_category"] = self.categorize(
            result[probability_column]
        )

        return result