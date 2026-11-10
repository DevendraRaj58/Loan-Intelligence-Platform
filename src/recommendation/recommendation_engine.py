
import numpy as np
import pandas as pd


class RecommendationEngine:

    FEATURE_ACTIONS = {
        "LATE_PAYMENT_RATIO":
            "Review repayment history and investigate recurring payment delays.",

        "DEBT_TO_CREDIT_RATIO":
            "Review existing debt relative to available credit and assess repayment capacity.",

        "CREDIT_UTILIZATION_RATIO":
            "Review outstanding credit utilization and existing financial obligations.",

        "MAXIMUM_CREDIT_UTILIZATION":
            "Investigate periods of high credit utilization and outstanding balances.",

        "REMAINING_INSTALLMENT_RATIO":
            "Review remaining installment obligations and ongoing loan commitments.",

        "AMT_ANNUITY":
            "Verify that the proposed installment is affordable given the applicant's verified income.",

        "AMT_CREDIT":
            "Review the requested loan amount against verified repayment capacity.",

        "DAYS_EMPLOYED":
            "Verify employment stability and continuity of income.",

        "EXT_SOURCE_1":
            "Verify the supporting external credit information.",

        "EXT_SOURCE_2":
            "Review the applicant's external credit assessment and supporting records.",

        "EXT_SOURCE_3":
            "Review the external credit assessment and verify its supporting information.",

        "REFUSAL_RATIO":
            "Review previous application outcomes in context rather than treating refusals as standalone evidence.",

        "EARLY_PAYMENT_RATIO":
            "Review historical repayment behaviour and confirm the completeness of payment records."
    }

    def __init__(
        self,
        risk_scorer,
        risk_categorizer,
        shap_explainer,
        top_n=5
    ):
        self.risk_scorer = risk_scorer
        self.risk_categorizer = risk_categorizer
        self.shap_explainer = shap_explainer
        self.top_n = int(top_n)

        if self.top_n < 1:
            raise ValueError("top_n must be at least 1.")

        if not np.isclose(
            self.risk_scorer.threshold,
            self.risk_categorizer.elevated_risk_threshold
        ):
            raise ValueError(
                "Scoring threshold and categorization threshold must match."
            )

    @staticmethod
    def _prepare_input(applicant_features, input_columns):

        if not isinstance(applicant_features, pd.DataFrame):
            raise TypeError("Applicant input must be a pandas DataFrame.")

        if len(applicant_features) != 1:
            raise ValueError(
                "The recommendation engine currently supports one applicant at a time."
            )

        if "TARGET" in applicant_features.columns:
            raise ValueError("TARGET must not be supplied as an input feature.")

        data = applicant_features.drop(
            columns=["SK_ID_CURR"],
            errors="ignore"
        )

        missing_columns = set(input_columns) - set(data.columns)

        if missing_columns:
            raise ValueError(
                f"Missing applicant features: {sorted(missing_columns)}"
            )

        return data.loc[:, input_columns]

    @staticmethod
    def _clean_feature_name(feature_name):

        name = str(feature_name)

        name = name.replace("numeric__", "")
        name = name.replace("categorical__", "")

        return name

    def _extract_factors(self, explanation):

        values = np.asarray(
            explanation.values,
            dtype=float
        )

        # Flatten a single-applicant explanation if necessary
        if values.ndim == 2 and values.shape[0] == 1:
            values = values[0]

        if values.ndim != 1:
            raise ValueError(
                "Expected SHAP explanation for exactly one applicant."
            )

        feature_names = explanation.feature_names

        if feature_names is None:
            feature_names = self.shap_explainer.feature_names

        if len(feature_names) != len(values):
            raise ValueError(
                "SHAP feature names and contribution values do not match."
            )

        contributions = pd.DataFrame({
            "feature": [
                self._clean_feature_name(name)
                for name in feature_names
            ],
            "shap_value": values
        })

        risk_increasing = (
            contributions[contributions["shap_value"] > 0]
            .nlargest(self.top_n, "shap_value")
        )

        risk_reducing = (
            contributions[contributions["shap_value"] < 0]
            .nsmallest(self.top_n, "shap_value")
        )

        return (
            risk_increasing.to_dict("records"),
            risk_reducing.to_dict("records")
        )

    def _generate_actions(self, risk_increasing_factors):

        actions = []

        for factor in risk_increasing_factors:

            feature = factor["feature"]

            action = self.FEATURE_ACTIONS.get(
                feature,
                f"Review {feature} and verify its supporting application or credit records."
            )

            actions.append({
                "related_feature": feature,
                "recommended_action": action
            })

        if not actions:
            actions.append({
                "related_feature": None,
                "recommended_action":
                    "Review the applicant's supporting financial and credit records."
            })

        return actions

    @staticmethod
    def _generate_summary(risk_category):

        summaries = {
            "LOW":
                "The estimated default probability is below the configured risk threshold. Standard verification is still required.",

            "MODERATE":
                "The applicant falls within the moderate-risk band. Review repayment capacity and supporting credit information.",

            "ELEVATED":
                "The estimated default probability is at or above the configured elevated-risk threshold. A more detailed manual review is recommended."
        }

        return summaries[risk_category]

    def recommend(self, applicant_features):

        # 1. Generate calibrated risk score
        scoring_result = self.risk_scorer.score(
            applicant_features
        )

        # 2. Assign risk category
        categorized_result = self.risk_categorizer.transform(
            scoring_result
        )

        applicant_result = categorized_result.iloc[0]

        # 3. Prepare the same features used by the scoring pipeline
        X = self._prepare_input(
            applicant_features,
            self.risk_scorer.input_columns
        )

        X_processed = self.risk_scorer.preprocessor.transform(X)

        X_selected = X_processed[
            :,
            self.risk_scorer.feature_mask
        ]

        # 4. Generate individual SHAP explanation
        explanation = self.shap_explainer.explain_instance(
            X_selected
        )

        risk_increasing, risk_reducing = self._extract_factors(
            explanation
        )

        # 5. Generate review actions
        recommended_actions = self._generate_actions(
            risk_increasing
        )

        risk_category = applicant_result["risk_category"]

        # 6. Assemble final assessment
        return {
            "estimated_default_probability": float(
                applicant_result["estimated_default_probability"]
            ),
            "decision_threshold": float(
                applicant_result["decision_threshold"]
            ),
            "above_threshold": bool(
                applicant_result["above_threshold"]
            ),
            "model_signal": applicant_result["model_signal"],
            "risk_category": risk_category,
            "risk_summary": self._generate_summary(risk_category),
            "risk_increasing_factors": risk_increasing,
            "risk_reducing_factors": risk_reducing,
            "recommended_actions": recommended_actions,
            "explanation_scale":
                "Underlying LightGBM raw model output, not calibrated probability",
            "decision_support_only": True
        }
