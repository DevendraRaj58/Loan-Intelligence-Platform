import pandas as pd
from src.utils.config import config
import numpy as np


class FeatureSelector:

    @staticmethod
    def remove_constant_features(
        dataset: pd.DataFrame
    ) -> pd.DataFrame:

        data = dataset.copy()

        constant_columns = [
            column
            for column in data.columns
            if data[column].nunique(dropna=False) == 1
        ]

        print(f"Removed {len(constant_columns)} constant features.")

        data = data.drop(
            columns=constant_columns
        )

        return data


    @staticmethod
    def remove_duplicate_features(
        dataset: pd.DataFrame
    ) -> pd.DataFrame:

        data = dataset.copy()

        duplicate_columns = []

        columns = data.columns

        for i in range(len(columns)):

            for j in range(i + 1, len(columns)):

                if data[columns[i]].equals(data[columns[j]]):

                    duplicate_columns.append(columns[j])

        print(
            f"Removed {len(duplicate_columns)} duplicate features."
        )

        data = data.drop(
            columns=duplicate_columns
        )

        return data


    @staticmethod
    def remove_correlated_features(
        dataset: pd.DataFrame
    ) -> pd.DataFrame:

        data = dataset.copy()

        correlation_matrix = data.corr().abs()

        upper_triangle = correlation_matrix.where(
            np.triu(
                np.ones(correlation_matrix.shape),
                k=1
            ).astype(bool)
        )

        threshold = config["feature_selection"]["correlation_threshold"]

        correlated_columns = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > threshold)
        ]

        print(
        f"Removed {len(correlated_columns)} highly correlated features:"
        )

        for column in correlated_columns:
            print(f" - {column}")

        data = data.drop(
            columns=correlated_columns
        )

        return data











    @staticmethod
    def transform(
        dataset: pd.DataFrame
    ) -> pd.DataFrame:

        data = dataset.copy()

        data = FeatureSelector.remove_constant_features(data)

        data = FeatureSelector.remove_duplicate_features(data)

        data = FeatureSelector.remove_correlated_features(data)

        return data