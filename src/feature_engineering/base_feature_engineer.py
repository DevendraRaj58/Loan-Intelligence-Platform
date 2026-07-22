"""
Base Feature Engineering Class

Defines the common interface for all feature engineering modules.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseFeatureEngineer(ABC):
    """
    Base class for all feature engineering modules.
    """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate engineered features.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.

        Returns
        -------
        pd.DataFrame
            Engineered feature dataframe.
        """
        pass
    
    @staticmethod
    def merge_feature(
        features: pd.DataFrame,
        new_feature: pd.DataFrame,
        key: str = "SK_ID_CURR"
    ) -> pd.DataFrame:
        """
        Merge a newly created feature dataframe into the master feature dataframe.

        Parameters
        ----------
        features : pd.DataFrame
            Master feature dataframe.

        new_feature : pd.DataFrame
            Newly engineered feature dataframe.

        key : str
            Merge key.

        Returns
        -------
        pd.DataFrame
            Updated feature dataframe.
        """

        return features.merge(
            new_feature,
            on=key,
            how="left",
            validate="one_to_one"
        )
    