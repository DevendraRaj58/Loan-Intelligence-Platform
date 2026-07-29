import pandas as pd
import numpy as np

from src.feature_engineering.base_feature_engineer import BaseFeatureEngineer
from src.utils.aggregation import Aggregation


class InstallmentsFeatureEngineer(BaseFeatureEngineer):

    def transform(
        self,
        installments_payments: pd.DataFrame
    ) -> pd.DataFrame:

        features = pd.DataFrame()
    
        features["SK_ID_CURR"] = (
            installments_payments["SK_ID_CURR"].unique()
        )


        installments_payments["PAYMENT_DELAY"] = (
            installments_payments["DAYS_ENTRY_PAYMENT"] -
            installments_payments["DAYS_INSTALMENT"]
        )

        installments_payments["PAYMENT_DIFFERENCE"] = (
            installments_payments["AMT_PAYMENT"] -
            installments_payments["AMT_INSTALMENT"]
        )

        installments_payments["PAYMENT_RATIO"] = (
            installments_payments["AMT_PAYMENT"] /
            installments_payments["AMT_INSTALMENT"]
        ).replace([np.inf, -np.inf], np.nan)


        total_payment_delay = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "PAYMENT_DELAY",
            "sum",
            "TOTAL_PAYMENT_DELAY"
        )

        average_payment_delay = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "PAYMENT_DELAY",
            "mean",
            "AVERAGE_PAYMENT_DELAY"
        )


        maximum_payment_delay = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "PAYMENT_DELAY",
            "max",
            "MAXIMUM_PAYMENT_DELAY"
        )

        minimum_payment_delay = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "PAYMENT_DELAY",
            "min",
            "MINIMUM_PAYMENT_DELAY"
        )

        total_payment_difference = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "PAYMENT_DIFFERENCE",
            "sum",
            "TOTAL_PAYMENT_DIFFERENCE"
        )


        average_payment_difference = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "PAYMENT_DIFFERENCE",
            "mean",
            "AVERAGE_PAYMENT_DIFFERENCE"
        )

        maximum_payment_difference = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "PAYMENT_DIFFERENCE",
            "max",
            "MAXIMUM_PAYMENT_DIFFERENCE"
        )

        minimum_payment_difference = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "PAYMENT_DIFFERENCE",
            "min",
            "MINIMUM_PAYMENT_DIFFERENCE"
        )

        average_payment_ratio = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "PAYMENT_RATIO",
            "mean",
            "AVERAGE_PAYMENT_RATIO"
        )

        minimum_payment_ratio = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "PAYMENT_RATIO",
            "min",
            "MINIMUM_PAYMENT_RATIO"
        )

        maximum_payment_ratio = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "PAYMENT_RATIO",
            "max",
            "MAXIMUM_PAYMENT_RATIO"
        )

        features = self.merge_feature(features, total_payment_delay)
        features = self.merge_feature(features, average_payment_delay)
        features = self.merge_feature(features, maximum_payment_delay)
        features = self.merge_feature(features, minimum_payment_delay)

        features = self.merge_feature(features, total_payment_difference)
        features = self.merge_feature(features, average_payment_difference)
        features = self.merge_feature(features, maximum_payment_difference)
        features = self.merge_feature(features, minimum_payment_difference)

        features = self.merge_feature(features, average_payment_ratio)
        features = self.merge_feature(features, minimum_payment_ratio)
        features = self.merge_feature(features, maximum_payment_ratio)


        installments_payments["LATE_PAYMENT"] = (
            installments_payments["PAYMENT_DELAY"] > 0
        ).astype(int)


        late_payments = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "LATE_PAYMENT",
            "sum",
            "TOTAL_LATE_PAYMENTS"
        )

        features = self.merge_feature(features, late_payments)

        installments_payments["EARLY_PAYMENT"] = (
            installments_payments["PAYMENT_DELAY"] < 0
        ).astype(int)

        early_payments = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "EARLY_PAYMENT",
            "sum",
            "TOTAL_EARLY_PAYMENTS"
        )

        features = self.merge_feature(features, early_payments)

        installments_payments["UNDER_PAYMENT"] = (
            installments_payments["PAYMENT_DIFFERENCE"] < 0
        ).astype(int)

        under_payments = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "UNDER_PAYMENT",
            "sum",
            "TOTAL_UNDERPAYMENTS"
        )

        features = self.merge_feature(features, under_payments)

        installments_payments["OVER_PAYMENT"] = (
            installments_payments["PAYMENT_DIFFERENCE"] > 0
        ).astype(int)

        over_payments = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "OVER_PAYMENT",
            "sum",
            "TOTAL_OVERPAYMENTS"
        )

        features = self.merge_feature(features, over_payments)

        total_installments = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "NUM_INSTALMENT_NUMBER",
            "count",
            "TOTAL_INSTALLMENTS"
        )

        features = self.merge_feature(features, total_installments)

        features["LATE_PAYMENT_RATIO"] = (
            features["TOTAL_LATE_PAYMENTS"] /
            features["TOTAL_INSTALLMENTS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["EARLY_PAYMENT_RATIO"] = (
            features["TOTAL_EARLY_PAYMENTS"] /
            features["TOTAL_INSTALLMENTS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["UNDERPAYMENT_RATIO"] = (
            features["TOTAL_UNDERPAYMENTS"] /
            features["TOTAL_INSTALLMENTS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["OVERPAYMENT_RATIO"] = (
            features["TOTAL_OVERPAYMENTS"] /
            features["TOTAL_INSTALLMENTS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        total_expected_amount = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "AMT_INSTALMENT",
            "sum",
            "TOTAL_EXPECTED_AMOUNT"
        )

        total_expected_amount = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "AMT_INSTALMENT",
            "sum",
            "TOTAL_EXPECTED_AMOUNT"
        )

        total_paid_amount = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "AMT_PAYMENT",
            "sum",
            "TOTAL_PAID_AMOUNT"
        )

        
        average_expected_amount = Aggregation.aggregate(
        installments_payments,
        "SK_ID_CURR",
        "AMT_INSTALMENT",
        "mean",
        "AVERAGE_EXPECTED_AMOUNT"
        )


        average_paid_amount = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "AMT_PAYMENT",
            "mean",
            "AVERAGE_PAID_AMOUNT"
        )

        features = self.merge_feature(features, total_expected_amount)
        features = self.merge_feature(features, total_paid_amount)

        features = self.merge_feature(features, average_expected_amount)
        features = self.merge_feature(features, average_paid_amount)


        latest_installment = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "DAYS_INSTALMENT",
            "max",
            "LATEST_INSTALLMENT"
        )

        oldest_installment = Aggregation.aggregate(
            installments_payments,
            "SK_ID_CURR",
            "DAYS_INSTALMENT",
            "min",
            "OLDEST_INSTALLMENT"
        )

        features = self.merge_feature(features, latest_installment)
        features = self.merge_feature(features, oldest_installment)

        features["PAYMENT_COVERAGE_RATIO"] = (
            features["TOTAL_PAID_AMOUNT"] /
            features["TOTAL_EXPECTED_AMOUNT"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["AVERAGE_PAYMENT_COVERAGE"] = (
            features["AVERAGE_PAID_AMOUNT"] /
            features["AVERAGE_EXPECTED_AMOUNT"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["INSTALLMENT_HISTORY_LENGTH"] = (
            features["LATEST_INSTALLMENT"] -
            features["OLDEST_INSTALLMENT"]
        )

        return features