import pandas as pd
import numpy as np

from src.feature_engineering.base_feature_engineer import BaseFeatureEngineer
from src.utils.aggregation import Aggregation


class CreditCardFeatureEngineer(BaseFeatureEngineer):

    def transform(
        self,
        credit_card_balance: pd.DataFrame
    ) -> pd.DataFrame:

        features = pd.DataFrame()
    
        features["SK_ID_CURR"] = (
            credit_card_balance["SK_ID_CURR"].unique()
        )


        status_counts = (
            credit_card_balance
            .groupby(["SK_ID_CURR", "NAME_CONTRACT_STATUS"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        status_counts = status_counts.rename(columns={
            "Active": "ACTIVE_CREDIT_CARDS",
            "Completed": "COMPLETED_CREDIT_CARDS",
            "Signed": "SIGNED_CREDIT_CARDS",
            "Demand": "DEMAND_CREDIT_CARDS",
            "Sent proposal": "SENT_PROPOSAL_CARDS",
            "Refused": "REFUSED_CREDIT_CARDS",
            "Approved": "APPROVED_CREDIT_CARDS"
        })


        features = self.merge_feature(
            features,
            status_counts
        )


        total_credit_card_records = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "SK_ID_PREV",
            "count",
            "TOTAL_CREDIT_CARD_RECORDS"
        )

        features = self.merge_feature(
            features,
            total_credit_card_records
        )



        total_credit_cards = (
            credit_card_balance
            .groupby("SK_ID_CURR")["SK_ID_PREV"]
            .nunique()
            .reset_index(name="TOTAL_CREDIT_CARDS")
        )

        features = self.merge_feature(
            features,
            total_credit_cards
        )

        features["ACTIVE_CARD_RATIO"] = (
            features["ACTIVE_CREDIT_CARDS"] /
            features["TOTAL_CREDIT_CARD_RECORDS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["COMPLETED_CARD_RATIO"] = (
            features["COMPLETED_CREDIT_CARDS"] /
            features["TOTAL_CREDIT_CARD_RECORDS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)


        total_balance = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "AMT_BALANCE",
            "sum",
            "TOTAL_BALANCE"
        )

        average_balance = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "AMT_BALANCE",
            "mean",
            "AVERAGE_BALANCE"
        )

        maximum_balance = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "AMT_BALANCE",
            "max",
            "MAXIMUM_BALANCE"
        )

        average_credit_limit = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "AMT_CREDIT_LIMIT_ACTUAL",
            "mean",
            "AVERAGE_CREDIT_LIMIT"
        )

        maximum_credit_limit = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "AMT_CREDIT_LIMIT_ACTUAL",
            "max",
            "MAXIMUM_CREDIT_LIMIT"
        )

        features = self.merge_feature(features, total_balance)
        features = self.merge_feature(features, average_balance)
        features = self.merge_feature(features, maximum_balance)

        features = self.merge_feature(features, average_credit_limit)
        features = self.merge_feature(features, maximum_credit_limit)


        features["AVERAGE_CREDIT_UTILIZATION"] = (
            features["AVERAGE_BALANCE"] /
            features["AVERAGE_CREDIT_LIMIT"]
        ).replace([np.inf, -np.inf], 0).fillna(0)


        features["MAXIMUM_CREDIT_UTILIZATION"] = (
            features["MAXIMUM_BALANCE"] /
            features["MAXIMUM_CREDIT_LIMIT"]
        ).replace([np.inf, -np.inf], 0).fillna(0)


        features["BALANCE_TO_LIMIT_RATIO"] = (
            features["TOTAL_BALANCE"] /
            features["MAXIMUM_CREDIT_LIMIT"]
        ).replace([np.inf, -np.inf], 0).fillna(0)


        total_payment = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "AMT_PAYMENT_TOTAL_CURRENT",
            "sum",
            "TOTAL_PAYMENT"
        )

        average_payment = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "AMT_PAYMENT_TOTAL_CURRENT",
            "mean",
            "AVERAGE_PAYMENT"
        )

        total_drawings = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "AMT_DRAWINGS_CURRENT",
            "sum",
            "TOTAL_DRAWINGS"
        )

        average_drawings = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "AMT_DRAWINGS_CURRENT",
            "mean",
            "AVERAGE_DRAWINGS"
        )

        total_atm_drawings = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "AMT_DRAWINGS_ATM_CURRENT",
            "sum",
            "TOTAL_ATM_DRAWINGS"
        )


        total_pos_drawings = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "AMT_DRAWINGS_POS_CURRENT",
            "sum",
            "TOTAL_POS_DRAWINGS"
        )


        average_receivable = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "AMT_TOTAL_RECEIVABLE",
            "mean",
            "AVERAGE_TOTAL_RECEIVABLE"
        )

        features = self.merge_feature(features, total_payment)
        features = self.merge_feature(features, average_payment)

        features = self.merge_feature(features, total_drawings)
        features = self.merge_feature(features, average_drawings)

        features = self.merge_feature(features, total_atm_drawings)
        features = self.merge_feature(features, total_pos_drawings)

        features = self.merge_feature(features, average_receivable)


        features["PAYMENT_TO_BALANCE_RATIO"] = (
            features["TOTAL_PAYMENT"] /
            features["TOTAL_BALANCE"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["DRAWING_TO_LIMIT_RATIO"] = (
            features["TOTAL_DRAWINGS"] /
            features["MAXIMUM_CREDIT_LIMIT"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["ATM_DRAWING_RATIO"] = (
            features["TOTAL_ATM_DRAWINGS"] /
            features["TOTAL_DRAWINGS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["POS_DRAWING_RATIO"] = (
            features["TOTAL_POS_DRAWINGS"] /
            features["TOTAL_DRAWINGS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)


        features["POS_DRAWING_RATIO"] = (
            features["TOTAL_POS_DRAWINGS"] /
            features["TOTAL_DRAWINGS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)


        total_dpd = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "SK_DPD",
            "sum",
            "CC_TOTAL_DPD"
        )

        average_dpd = Aggregation.aggregate(
        credit_card_balance,
        "SK_ID_CURR",
        "SK_DPD",
        "mean",
        "CC_AVERAGE_DPD"
        )

        maximum_dpd = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "SK_DPD",
            "max",
            "CC_MAXIMUM_DPD"
        )

        total_default_dpd = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "SK_DPD_DEF",
            "sum",
            "CC_TOTAL_DEFAULT_DPD"
        )

        average_default_dpd = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "SK_DPD_DEF",
            "mean",
            "CC_AVERAGE_DEFAULT_DPD"
        )

        maximum_default_dpd = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "SK_DPD_DEF",
            "max",
            "CC_MAXIMUM_DEFAULT_DPD"
        )


        latest_record = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "MONTHS_BALANCE",
            "max",
            "LATEST_CC_RECORD"
        )

        oldest_record = Aggregation.aggregate(
            credit_card_balance,
            "SK_ID_CURR",
            "MONTHS_BALANCE",
            "min",
            "OLDEST_CC_RECORD"
        )

        features = self.merge_feature(features, total_dpd)
        features = self.merge_feature(features, average_dpd)
        features = self.merge_feature(features, maximum_dpd)

        features = self.merge_feature(features, total_default_dpd)
        features = self.merge_feature(features, average_default_dpd)
        features = self.merge_feature(features, maximum_default_dpd)

        features = self.merge_feature(features, latest_record)
        features = self.merge_feature(features, oldest_record)


        features["CC_HISTORY_LENGTH"] = (
            features["LATEST_CC_RECORD"] -
            features["OLDEST_CC_RECORD"]
        )

        features = features.fillna(0)

        return features
