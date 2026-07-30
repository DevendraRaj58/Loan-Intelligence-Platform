import pandas as pd


class DatasetPreprocessor:

    @staticmethod
    def prepare(
        dataset: pd.DataFrame
    ) -> pd.DataFrame:

        data = dataset.copy()

        return data