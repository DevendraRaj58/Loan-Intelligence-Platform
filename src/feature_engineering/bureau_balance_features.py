"""
Feature Engineering for Bureau Balance Dataset.
"""

import pandas as pd
import numpy as np
from src.feature_engineering.base_feature_engineer import BaseFeatureEngineer
from src.utils.aggregation import Aggregation


class BureauBalanceFeatureEngineer(BaseFeatureEngineer):


    """
    Creates loan-level repayment features from bureau_balance
    and aggregates them to customer level.
    """

    KEY = 'SK_ID_BUREAU'
    def transform(self, bureau_balance: pd.DataFrame) -> pd.DataFrame:

        features = pd.DataFrame()

        features["SK_ID_BUREAU"] = (
            bureau_balance["SK_ID_BUREAU"].unique()
        )



        on_time_months = (
        bureau_balance[bureau_balance["STATUS"] == "0"]
        .groupby("SK_ID_BUREAU")
        .size()
        .reset_index(name="ON_TIME_MONTHS")
        )

        features = self.merge_feature(
            features,
            on_time_months,
            key=self.KEY
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
            key=self.KEY
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
            key=self.KEY
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
            key=self.KEY
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

    def aggregate_to_customer(self,bureau_enriched: pd.DataFrame) -> pd.DataFrame:

        features = pd.DataFrame()

        features["SK_ID_CURR"] = (
            bureau_enriched["SK_ID_CURR"].unique()
        )

        total_on_time_months = Aggregation.aggregate(
        df=bureau_enriched,
        groupby_col="SK_ID_CURR",
        target_col="ON_TIME_MONTHS",
        agg_func="sum",
        feature_name="TOTAL_ON_TIME_MONTHS"
        )

        features = self.merge_feature(
            features,
            total_on_time_months
        )

        total_delinquent_months = Aggregation.aggregate(
        df=bureau_enriched,
        groupby_col="SK_ID_CURR",
        target_col="DELINQUENT_MONTHS",
        agg_func="sum",
        feature_name="TOTAL_DELINQUENT_MONTHS"
        )

        features = self.merge_feature(
            features,
            total_delinquent_months
        )


        worst_delinquency = Aggregation.aggregate(
        df=bureau_enriched,
        groupby_col="SK_ID_CURR",
        target_col="WORST_DELINQUENCY_LEVEL",
        agg_func="max",
        feature_name="WORST_DELINQUENCY_LEVEL"
        )

        features = self.merge_feature(
            features,
            worst_delinquency
        )


        total_closed_months = Aggregation.aggregate(
        df=bureau_enriched,
        groupby_col="SK_ID_CURR",
        target_col="CLOSED_MONTHS",
        agg_func="sum",
        feature_name="TOTAL_CLOSED_MONTHS"
        )

        features = self.merge_feature(
            features,
            total_closed_months
        )

        average_on_time_ratio = Aggregation.aggregate(
        df=bureau_enriched,
        groupby_col="SK_ID_CURR",
        target_col="ON_TIME_PAYMENT_RATIO",
        agg_func="mean",
        feature_name="AVERAGE_ON_TIME_PAYMENT_RATIO"
        )

        features = self.merge_feature(
            features,
            average_on_time_ratio
        )


        average_delinquency_ratio = Aggregation.aggregate(
        df=bureau_enriched,
        groupby_col="SK_ID_CURR",
        target_col="DELINQUENCY_RATIO",
        agg_func="mean",
        feature_name="AVERAGE_DELINQUENCY_RATIO"
        )

        features = self.merge_feature(
            features,
            average_delinquency_ratio
        )

        features = features.fillna(0)




        return features