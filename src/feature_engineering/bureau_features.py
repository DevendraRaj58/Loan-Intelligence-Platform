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

        features = features.merge(
            loan_count,
            on="SK_ID_CURR",
            how="left")
        
        return features
