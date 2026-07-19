"""
Data Dictionary Utility

Generates a structured summary of a dataframe.
"""

import pandas as pd


class DataDictionary:

    @staticmethod
    def generate(df: pd.DataFrame,
                 cardinality_threshold : int =50
                ) -> pd.DataFrame:

        summary = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Feature Type": [
            "Categorical" if df[col].dtype == "object"
            else "Numerical"
            for col in df.columns
        ],
        "Non Null Count": df.notnull().sum().values,
        "Missing Values": df.isnull().sum().values,
        "Missing %": (
            df.isnull().sum() / len(df) * 100
        ).round(2).values,
        "Unique Values": df.nunique().values
        })

        summary["Constant Feature"] = summary["Unique Values"] == 1

        summary["High Cardinality"] = summary["Unique Values"] > cardinality_threshold

        return summary