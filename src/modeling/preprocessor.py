import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ModelPreprocessor:

    @staticmethod
    def create(X_train):

        numeric_columns = X_train.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_columns = X_train.select_dtypes(
            include=["object", "category", "string", "bool"]
        ).columns.tolist()

        # Numerical preprocessing
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Categorical preprocessing
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float32
            ))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_columns),
                ("categorical", categorical_pipeline, categorical_columns)
            ],
            remainder="drop",
            sparse_threshold=0.0
        )

        return preprocessor