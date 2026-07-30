import pandas as pd


class FeatureMerger:

    @staticmethod
    def merge(
        application_train: pd.DataFrame,
        bureau_features: pd.DataFrame,
        bureau_balance_features: pd.DataFrame,
        previous_application_features: pd.DataFrame,
        pos_cash_features: pd.DataFrame,
        installment_features: pd.DataFrame,
        credit_card_features: pd.DataFrame
    ) -> pd.DataFrame:

        dataset = application_train.copy()

        dataset = dataset.merge(
            bureau_features,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one"
        )

        dataset = dataset.merge(
            bureau_balance_features,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one"
        )

        dataset = dataset.merge(
            previous_application_features,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one"
        )

        dataset = dataset.merge(
            pos_cash_features,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one"
        )

        dataset = dataset.merge(
            installment_features,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one"
        )

        dataset = dataset.merge(
            credit_card_features,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one"
        )



        return dataset

