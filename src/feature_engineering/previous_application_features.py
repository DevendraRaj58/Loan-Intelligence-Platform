import pandas as pd
import numpy as np

from src.feature_engineering.base_feature_engineer import BaseFeatureEngineer
from src.utils.aggregation import Aggregation

class PreviousApplicationFeatureEngineer(BaseFeatureEngineer):

    def transform(
        self,
        previous_application: pd.DataFrame
    ) -> pd.DataFrame:

        features = pd.DataFrame()

        features["SK_ID_CURR"] = (
            previous_application["SK_ID_CURR"].unique()
        )

        status_counts = (
        previous_application
        .groupby(["SK_ID_CURR", "NAME_CONTRACT_STATUS"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        )

        status_counts = status_counts.rename(
        columns={
            "Approved": "TOTAL_APPROVED_APPLICATIONS",
            "Refused": "TOTAL_REFUSED_APPLICATIONS",
            "Canceled": "TOTAL_CANCELED_APPLICATIONS",
            "Unused offer": "TOTAL_UNUSED_OFFERS"
        }
        )






        return features