import pandas as pd
import numpy as np

from src.feature_engineering.base_feature_engineer import BaseFeatureEngineer
from src.utils.aggregation import Aggregation


class POSCashBalanceFeatureEngineer(BaseFeatureEngineer):

    def transform(
        self,
        pos_cash_balance: pd.DataFrame
    ) -> pd.DataFrame:

        features = pd.DataFrame()

        features["SK_ID_CURR"] = (
            pos_cash_balance["SK_ID_CURR"].unique()
        )


        total_dpd = Aggregation.aggregate(
            pos_cash_balance,
            "SK_ID_CURR",
            "SK_DPD",
            "sum",
            "POS_TOTAL_DPD"
        )

        average_dpd = Aggregation.aggregate(
            pos_cash_balance,
            "SK_ID_CURR",
            "SK_DPD",
            "mean",
            "POS_AVERAGE_DPD"
        )

        maximum_dpd = Aggregation.aggregate(
            pos_cash_balance,
            "SK_ID_CURR",
            "SK_DPD",
            "max",
            "POS_MAXIMUM_DPD"
        )

        total_default_dpd = Aggregation.aggregate(
            pos_cash_balance,
            "SK_ID_CURR",
            "SK_DPD_DEF",
            "sum",
            "POS_TOTAL_DEFAULT_DPD"
        )

        average_default_dpd = Aggregation.aggregate(
            pos_cash_balance,
            "SK_ID_CURR",
            "SK_DPD_DEF",
            "mean",
            "POS_AVERAGE_DEFAULT_DPD"
        )

        maximum_default_dpd = Aggregation.aggregate(
            pos_cash_balance,
            "SK_ID_CURR",
            "SK_DPD_DEF",
            "max",
            "POS_MAXIMUM_DEFAULT_DPD"
        )

        average_installments = Aggregation.aggregate(
            pos_cash_balance,
            "SK_ID_CURR",
            "CNT_INSTALMENT",
            "mean",
            "POS_AVERAGE_INSTALLMENTS"
        )

        maximum_installments = Aggregation.aggregate(
            pos_cash_balance,
            "SK_ID_CURR",
            "CNT_INSTALMENT",
            "max",
            "POS_MAXIMUM_INSTALLMENTS"
        )

        average_future_installments = Aggregation.aggregate(
            pos_cash_balance,
            "SK_ID_CURR",
            "CNT_INSTALMENT_FUTURE",
            "mean",
            "AVERAGE_FUTURE_INSTALLMENTS"
        )

        maximum_future_installments = Aggregation.aggregate(
            pos_cash_balance,
            "SK_ID_CURR",
            "CNT_INSTALMENT_FUTURE",
            "max",
            "MAXIMUM_FUTURE_INSTALLMENTS"
        )

        features = self.merge_feature(features, total_dpd)
        features = self.merge_feature(features, average_dpd)
        features = self.merge_feature(features, maximum_dpd)

        features = self.merge_feature(features, total_default_dpd)
        features = self.merge_feature(features, average_default_dpd)
        features = self.merge_feature(features, maximum_default_dpd)

        features = self.merge_feature(features, average_installments)
        features = self.merge_feature(features, maximum_installments)

        features = self.merge_feature(features, average_future_installments)
        features = self.merge_feature(features, maximum_future_installments)


        features["REMAINING_INSTALLMENT_RATIO"] = (
            features["AVERAGE_FUTURE_INSTALLMENTS"] /
            features["POS_AVERAGE_INSTALLMENTS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["DEFAULT_RATIO"] = (
            features["POS_TOTAL_DEFAULT_DPD"] /
            features["POS_TOTAL_DPD"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["DPD_PER_INSTALLMENT"] = (
            features["POS_AVERAGE_DPD"] /
            features["POS_AVERAGE_INSTALLMENTS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)


        status_counts = (
            pos_cash_balance
            .groupby(["SK_ID_CURR", "NAME_CONTRACT_STATUS"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        status_counts = status_counts.rename(
            columns={
                "Active": "ACTIVE_LOANS",
                "Completed": "COMPLETED_LOANS",
                "Signed": "SIGNED_LOANS",
                "Demand": "DEMAND_LOANS",
                "Returned to the store": "RETURNED_LOANS",
                "Approved": "APPROVED_LOANS",
                "Amortized debt": "AMORTIZED_LOANS",
                "Canceled": "CANCELED_LOANS",
                "XNA": "UNKNOWN_STATUS_LOANS"
            }
        )

        features = self.merge_feature(
            features,
            status_counts
        )


        total_pos_loans = Aggregation.aggregate(
            df=pos_cash_balance,
            groupby_col="SK_ID_CURR",
            target_col="SK_ID_PREV",
            agg_func="nunique",
            feature_name="TOTAL_POS_LOANS"
        )

        features = self.merge_feature(
            features,
            total_pos_loans
        )



        features["ACTIVE_LOAN_RATIO"] = (
            features["ACTIVE_LOANS"] /
            features["TOTAL_POS_LOANS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["COMPLETED_LOAN_RATIO"] = (
            features["COMPLETED_LOANS"] /
            features["TOTAL_POS_LOANS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["SIGNED_LOAN_RATIO"] = (
            features["SIGNED_LOANS"] /
            features["TOTAL_POS_LOANS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["DEMAND_LOAN_RATIO"] = (
            features["DEMAND_LOANS"] /
            features["TOTAL_POS_LOANS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["RETURNED_LOAN_RATIO"] = (
            features["RETURNED_LOANS"] /
            features["TOTAL_POS_LOANS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)


        latest_pos_record = Aggregation.aggregate(
            pos_cash_balance,
            "SK_ID_CURR",
            "MONTHS_BALANCE",
            "max",
            "LATEST_POS_RECORD"
        )

        oldest_pos_record = Aggregation.aggregate(
            pos_cash_balance,
            "SK_ID_CURR",
            "MONTHS_BALANCE",
            "min",
            "OLDEST_POS_RECORD"
        )

        average_pos_record = Aggregation.aggregate(
            pos_cash_balance,
            "SK_ID_CURR",
            "MONTHS_BALANCE",
            "mean",
            "AVERAGE_POS_RECORD"
        )


        features = self.merge_feature(features, latest_pos_record)
        features = self.merge_feature(features, oldest_pos_record)
        features = self.merge_feature(features, average_pos_record)

        features["POS_HISTORY_LENGTH"] = (
            features["LATEST_POS_RECORD"] -
            features["OLDEST_POS_RECORD"]
        )

        total_monthly_records = Aggregation.aggregate(
            pos_cash_balance,
            "SK_ID_CURR",
            "MONTHS_BALANCE",
            "count",
            "TOTAL_MONTHLY_RECORDS"
        )

        features = self.merge_feature(
            features,
            total_monthly_records
        )

        features["AVERAGE_MONTHS_PER_LOAN"] = (
            features["TOTAL_MONTHLY_RECORDS"] /
            features["TOTAL_POS_LOANS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)


        return features