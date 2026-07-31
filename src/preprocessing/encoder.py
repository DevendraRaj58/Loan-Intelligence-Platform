import pandas as pd
from sklearn.preprocessing import LabelEncoder


class CategoricalEncoder:

    @staticmethod
    def transform(
        dataset: pd.DataFrame,
        method: str = "label"
    ):

        data = dataset.copy()

        categorical_columns = data.select_dtypes(
            include=["object", "category"]
        ).columns

        if method == "none":
            return data

        elif method == "label":

            for column in categorical_columns:
                encoder = LabelEncoder()
                data[column] = encoder.fit_transform(
                    data[column].astype(str)
                )

        elif method == "onehot":

            data = pd.get_dummies(
                data,
                columns=categorical_columns,
                drop_first=True
            )

        else:
            raise ValueError(f"Unknown encoding method: {method}")

        return data