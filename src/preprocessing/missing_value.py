import pandas as pd
import numpy as np


class MissingValueHandler:

    @staticmethod
    def transform(
        dataset: pd.DataFrame
    ) -> pd.DataFrame:

        data = dataset.copy()

        numeric_columns = data.select_dtypes(
            include=np.number
        ).columns

        categorical_columns = data.select_dtypes(
            exclude=np.number
        ).columns

        for column in numeric_columns:
            median = data[column].median()
            data[column] = data[column].fillna(median)

        for column in categorical_columns:
            data[column] = data[column].fillna("Unknown")

        return data