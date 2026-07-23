"""
Feature engineering for bureau.csv
"""

import pandas as pd
import numpy as np
from src.feature_engineering.base_feature_engineer import BaseFeatureEngineer
from src.utils.aggregation import Aggregation

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

        total_debt = Aggregation.aggregate(
        bureau,
        "SK_ID_CURR",
        "AMT_CREDIT_SUM_DEBT",
        "sum",
        "TOTAL_OUTSTANDING_DEBT"
        )

        total_credit = Aggregation.aggregate(
        bureau,
        "SK_ID_CURR",
        "AMT_CREDIT_SUM",
        "sum",
        "TOTAL_CREDIT_AMOUNT"
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
        
        features["DEBT_TO_CREDIT_RATIO"] = (
        features["TOTAL_OUTSTANDING_DEBT"] /
        features["TOTAL_CREDIT_AMOUNT"]
        ).replace(
        [np.inf,-np.inf],0
        ).fillna(0)

        average_credit = Aggregation.aggregate(
        df=bureau,
        groupby_col="SK_ID_CURR",
        target_col="AMT_CREDIT_SUM",
        agg_func="mean",
        feature_name="AVERAGE_CREDIT_AMOUNT"
        )

        features = self.merge_feature(
        features,
        average_credit
        )


        max_credit = Aggregation.aggregate(
        bureau,
        "SK_ID_CURR",
        "AMT_CREDIT_SUM",
        "max",
        "MAXIMUM_CREDIT_AMOUNT"
        )
    
        features = self.merge_feature(
            features,
            max_credit
        )

        minimum_credit = Aggregation.aggregate(
        bureau,
        "SK_ID_CURR",
        "AMT_CREDIT_SUM",
        "min",
        "MINIMUM_CREDIT_AMOUNT"
        )

        features = self.merge_feature(
            features,
            minimum_credit
        )


        total_overdue = Aggregation.aggregate(
        df=bureau,
        groupby_col="SK_ID_CURR",
        target_col="AMT_CREDIT_SUM_OVERDUE",
        agg_func="sum",
        feature_name="TOTAL_OVERDUE_AMOUNT"
        )

        features = self.merge_feature(
            features,
            total_overdue
        )

        average_overdue = Aggregation.aggregate(
        df=bureau,
        groupby_col="SK_ID_CURR",
        target_col="AMT_CREDIT_SUM_OVERDUE",
        agg_func="mean",
        feature_name="AVERAGE_OVERDUE_AMOUNT"
        )

        features = self.merge_feature(
            features,
            average_overdue
        )

        maximum_overdue = Aggregation.aggregate(
        df=bureau,
        groupby_col="SK_ID_CURR",
        target_col="AMT_CREDIT_SUM_OVERDUE",
        agg_func="max",
        feature_name="MAXIMUM_OVERDUE_AMOUNT"
        )

        features = self.merge_feature(
            features,
            maximum_overdue
        )


        features["OVERDUE_RATIO"] = (
        features["TOTAL_OVERDUE_AMOUNT"]/ features["TOTAL_OUTSTANDING_DEBT"]
        ).replace(
            [np.inf, -np.inf],
            0
        ).fillna(0)

        recent_credit = Aggregation.aggregate(
        df=bureau,
        groupby_col="SK_ID_CURR",
        target_col="DAYS_CREDIT",
        agg_func="max",
        feature_name="DAYS_SINCE_LAST_EXTERNAL_LOAN"
        )

        features = self.merge_feature(
            features,
            recent_credit
        )

        oldest_credit = Aggregation.aggregate(
        df=bureau,
        groupby_col="SK_ID_CURR",
        target_col="DAYS_CREDIT",
        agg_func="min",
        feature_name="DAYS_SINCE_FIRST_EXTERNAL_LOAN"
        )

        features = self.merge_feature(
            features,
            oldest_credit
        )


        features["EXTERNAL_CREDIT_HISTORY_DAYS"] = (
        features["DAYS_SINCE_LAST_EXTERNAL_LOAN"]
        - features["DAYS_SINCE_FIRST_EXTERNAL_LOAN"]
        )

        longest_remaining_loan = Aggregation.aggregate(
        df=bureau,
        groupby_col="SK_ID_CURR",
        target_col="DAYS_CREDIT_ENDDATE",
        agg_func="max",
        feature_name="LONGEST_REMAINING_LOAN_DAYS"
        )

        features = self.merge_feature(
            features,
            longest_remaining_loan
        )

        active_future_loans = (
        bureau[bureau["DAYS_CREDIT_ENDDATE"] > 0]
        .groupby("SK_ID_CURR")
        .size()
        .reset_index(name="ACTIVE_FUTURE_LOANS")
        )

        features = self.merge_feature(
            features,
            active_future_loans
        )

        features["ACTIVE_FUTURE_LOANS"] = (
            features["ACTIVE_FUTURE_LOANS"]
            .fillna(0)
            .astype(int)
        )

        total_credit_limit = Aggregation.aggregate(
        df=bureau,
        groupby_col="SK_ID_CURR",
        target_col="AMT_CREDIT_SUM_LIMIT",
        agg_func="sum",
        feature_name="TOTAL_CREDIT_LIMIT"
        )

        features = self.merge_feature(
            features,
            total_credit_limit
        )

        features["CREDIT_UTILIZATION_RATIO"] = (
        features["TOTAL_OUTSTANDING_DEBT"]
        / features["TOTAL_CREDIT_LIMIT"]
        ).replace(
            [np.inf, -np.inf],
            0
        ).fillna(0)


        total_credit_prolong = Aggregation.aggregate(
        df=bureau,
        groupby_col="SK_ID_CURR",
        target_col="CNT_CREDIT_PROLONG",
        agg_func="sum",
        feature_name="TOTAL_CREDIT_PROLONGATIONS"
        )

        features = self.merge_feature(
            features,
            total_credit_prolong
        )

        numeric_columns = features.columns.drop("SK_ID_CURR")

        features[numeric_columns] = features[numeric_columns].fillna(0)


        return features
