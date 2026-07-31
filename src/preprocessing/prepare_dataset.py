import pandas as pd

from src.preprocessing.validator import DatasetValidator
from src.preprocessing.missing_value import MissingValueHandler
from src.preprocessing.encoder import CategoricalEncoder


class DatasetPreprocessor:

    @staticmethod
    def prepare(
        dataset: pd.DataFrame,
        target_column: str = "TARGET",
        encoding_method: str = "label"
    ):

        data = dataset.copy()

        # Validate dataset
        DatasetValidator.validate(
            data,
            target_column
        )

        # Handle missing values
        data = MissingValueHandler.transform(data)

        # Encode categorical variables
        data = CategoricalEncoder.transform(
            data,
            method=encoding_method
        )

        # Drop identifier column
        from src.utils.config import config

        if config["preprocessing"]["drop_identifier_columns"]:
            data = data.drop(
                columns=config["preprocessing"]["identifier_columns"],
                errors="ignore"
            )

        # Split features and target
        X = data.drop(
            columns=[target_column]
        )

        y = data[target_column]

        return X, y