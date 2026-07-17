"""
Dataset Profiling Utility

Provides a quick overview of any tabular dataset.
"""

import pandas as pd


class DatasetProfiler:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def profile(self):

        print("=" * 60)
        print("DATASET PROFILE")
        print("=" * 60)

        print(f"\nShape : {self.df.shape}")

        print("\nColumn Names")
        print(self.df.columns.tolist())

        print("\nData Types")
        print(self.df.dtypes)

        print("\nMissing Values")
        print(self.df.isnull().sum())

        print("\nDuplicate Rows")
        print(self.df.duplicated().sum())

        print("\nMemory Usage")
        print(self.df.memory_usage(deep=True).sum() / 1024**2, "MB")

        print("\nNumerical Summary")
        print(self.df.describe())

        print("\nCategorical Summary")
        print(self.df.describe(include="object"))