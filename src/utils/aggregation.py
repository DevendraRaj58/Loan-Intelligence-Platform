"""
Aggregation Utility

Provides reusable aggregation functions for feature engineering.
"""

import pandas as pd


class Aggregation:

    @staticmethod
    def aggregate(
        df: pd.DataFrame,
        groupby_col: str,
        target_col: str,
        agg_func: str,
        feature_name: str
    ) -> pd.DataFrame:
        """
        Aggregate a column and return a feature dataframe.
        """

        return (
            df
            .groupby(groupby_col)[target_col]
            .agg(agg_func)
            .reset_index(name=feature_name)
        )