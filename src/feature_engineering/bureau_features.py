"""
Feature engineering for bureau.csv
"""

import pandas as pd

from src.feature_engineering.base_feature_engineer import BaseFeatureEngineer


class BureauFeatureEngineer(BaseFeatureEngineer):

    def transform(self, bureau: pd.DataFrame) -> pd.DataFrame:

        features = pd.DataFrame()

        features["SK_ID_CURR"] = bureau["SK_ID_CURR"].unique()

        loan_count = (
        bureau
        .groupby("SK_ID_CURR")
        .size()
        .reset_index(name="TOTAL_EXTERNAL_LOANS"))

        status_counts = (
        bureau
        .groupby(["SK_ID_CURR", "CREDIT_ACTIVE"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

        status_counts = status_counts.rename(
            columns={
                "Active": "ACTIVE_EXTERNAL_LOANS",
                "Closed": "CLOSED_EXTERNAL_LOANS",
                "Sold": "SOLD_EXTERNAL_LOANS",
                "Bad debt": "BAD_DEBT_LOANS",
            }
        )

        total_debt = (
        bureau
        .groupby("SK_ID_CURR")["AMT_CREDIT_SUM_DEBT"]
        .sum()
        .reset_index(name="TOTAL_OUTSTANDING_DEBT")
        )

        total_credit = (
            bureau
            .groupby("SK_ID_CURR")["AMT_CREDIT_SUM"]
            .sum()
            .reset_index(name="TOTAL_CREDIT_AMOUNT")
            )

        features = self.merge_feature(
                features,
                total_credit
            )
        
        features = self.merge_feature(
        features,
        total_debt
        )

        features = self.merge_feature(
        features,
        status_counts
        )


        features = self.merge_feature(
        features,
        loan_count
        )
        
        return features
