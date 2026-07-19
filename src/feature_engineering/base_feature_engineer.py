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

    