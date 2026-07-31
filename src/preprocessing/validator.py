import pandas as pd


class DatasetValidator:

    @staticmethod
    def validate(
        dataset: pd.DataFrame,
        target_column: str = "TARGET"
    ) -> None:

        # Empty dataset
        if dataset.empty:
            raise ValueError("Dataset is empty.")

        # Duplicate columns
        if dataset.columns.duplicated().sum() > 0:
            raise ValueError("Duplicate columns detected.")

        # Duplicate customers
        if dataset["SK_ID_CURR"].duplicated().sum() > 0:
            raise ValueError("Duplicate SK_ID_CURR values detected.")

        # Target column exists
        if target_column not in dataset.columns:
            raise ValueError(f"{target_column} column not found.")

        # Missing target values
        if dataset[target_column].isnull().sum() > 0:
            raise ValueError(f"{target_column} contains missing values.")

        print("Dataset validation passed.")