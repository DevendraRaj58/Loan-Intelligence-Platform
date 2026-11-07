import pandas as pd

from sklearn.model_selection import train_test_split

from src.utils.config import config


class DataSplitter:

    @staticmethod
    def split(
        X: pd.DataFrame,
        y: pd.Series
    ):
        test_size = config["data"]["test_size"]
        random_state = config["data"]["random_state"]

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        return X_train, X_valid, y_train, y_valid