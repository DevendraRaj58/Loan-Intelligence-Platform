"""
Feature Engineering for Bureau Balance Dataset.
"""

import pandas as pd
import numpy as np
from src.feature_engineering.base_feature_engineer import BaseFeatureEngineer
from src.utils.aggregation import Aggregation


class BureauBalanceFeatureEngineer(BaseFeatureEngineer):

    def transform(self, bureau_balance: pd.DataFrame) -> pd.DataFrame:

        features = pd.DataFrame()

        features["SK_ID_BUREAU"] = (
            bureau_balance["SK_ID_BUREAU"].unique()
        )
        KEY = 'SK_ID_BUREAU'



        on_time_months = (
        bureau_balance[bureau_balance["STATUS"] == "0"]
        .groupby("SK_ID_BUREAU")
        .size()
        .reset_index(name="ON_TIME_MONTHS")
        )

        features = self.merge_feature(
            features,
            on_time_months,
            key=KEY
        )

        features["ON_TIME_MONTHS"] = (
            features["ON_TIME_MONTHS"]
            .fillna(0)
            .astype(int)
        )



        delinquent_months = (
        bureau_balance[
            bureau_balance["STATUS"].isin(["1", "2", "3", "4", "5"])
        ]
        .groupby("SK_ID_BUREAU")
        .size()
        .reset_index(name="DELINQUENT_MONTHS")
        )

        features = self.merge_feature(
            features,
            delinquent_months,
            key=KEY
        )

        features["DELINQUENT_MONTHS"] = (
            features["DELINQUENT_MONTHS"]
            .fillna(0)
            .astype(int)
        )


        worst_delinquency = (
        bureau_balance[
            bureau_balance["STATUS"].isin(["1", "2", "3", "4", "5"])
        ]
        .assign(
            STATUS=lambda df: df["STATUS"].astype(int)
        )
        .groupby("SK_ID_BUREAU")["STATUS"]
        .max()
        .reset_index(name="WORST_DELINQUENCY_LEVEL")
        )

        features = self.merge_feature(
            features,
            worst_delinquency,
            key=KEY
        )

        features["WORST_DELINQUENCY_LEVEL"] = (
            features["WORST_DELINQUENCY_LEVEL"]
            .fillna(0)
            .astype(int)
        )


        closed_months = (
        bureau_balance[bureau_balance["STATUS"] == "C"]
        .groupby("SK_ID_BUREAU")
        .size()
        .reset_index(name="CLOSED_MONTHS")
        )

        features = self.merge_feature(
            features,
            closed_months,
            key=KEY
        )

        features["CLOSED_MONTHS"] = (
            features["CLOSED_MONTHS"]
            .fillna(0)
            .astype(int)
        )

        features["ON_TIME_PAYMENT_RATIO"] = (
        features["ON_TIME_MONTHS"] /
        (
            features["ON_TIME_MONTHS"] +
            features["DELINQUENT_MONTHS"]
        )
        ).replace(
            [np.inf, -np.inf],
            0
        ).fillna(0)


        features["DELINQUENCY_RATIO"] = (
        features["DELINQUENT_MONTHS"] /
        (
            features["ON_TIME_MONTHS"] +
            features["DELINQUENT_MONTHS"]
        )
        ).replace(
            [np.inf, -np.inf],
            0
        ).fillna(0)



        return features