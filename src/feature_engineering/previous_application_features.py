import pandas as pd
import numpy as np

from src.feature_engineering.base_feature_engineer import BaseFeatureEngineer
from src.utils.aggregation import Aggregation

class PreviousApplicationFeatureEngineer(BaseFeatureEngineer):

    def transform(
        self,
        previous_application: pd.DataFrame
    ) -> pd.DataFrame:

        features = pd.DataFrame()

        features["SK_ID_CURR"] = (
            previous_application["SK_ID_CURR"].unique()
        )

        status_counts = (
        previous_application
        .groupby(["SK_ID_CURR", "NAME_CONTRACT_STATUS"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        )

        status_counts = status_counts.rename(
        columns={
            "Approved": "TOTAL_APPROVED_APPLICATIONS",
            "Refused": "TOTAL_REFUSED_APPLICATIONS",
            "Canceled": "TOTAL_CANCELED_APPLICATIONS",
            "Unused offer": "TOTAL_UNUSED_OFFERS"
        }
        )

        features = self.merge_feature(
        features,
        status_counts
        )




        total_previous_applications = Aggregation.aggregate(
        df=previous_application,
        groupby_col="SK_ID_CURR",
        target_col="SK_ID_PREV",
        agg_func="count",
        feature_name="TOTAL_PREVIOUS_APPLICATIONS"
        )

        features = self.merge_feature(
            features,
            total_previous_applications
        )


        features["APPROVAL_RATIO"] = (
            features["TOTAL_APPROVED_APPLICATIONS"] /
            features["TOTAL_PREVIOUS_APPLICATIONS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)


        features["REFUSAL_RATIO"] = (
            features["TOTAL_REFUSED_APPLICATIONS"] /
            features["TOTAL_PREVIOUS_APPLICATIONS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)


        features["CANCELLATION_RATIO"] = (
            features["TOTAL_CANCELED_APPLICATIONS"] /
            features["TOTAL_PREVIOUS_APPLICATIONS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)


        features["UNUSED_OFFER_RATIO"] = (
            features["TOTAL_UNUSED_OFFERS"] /
            features["TOTAL_PREVIOUS_APPLICATIONS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)


        total_application_amount = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="AMT_APPLICATION",
            agg_func="sum",
            feature_name="TOTAL_APPLICATION_AMOUNT"
        )

        average_application_amount = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="AMT_APPLICATION",
            agg_func="mean",
            feature_name="AVERAGE_APPLICATION_AMOUNT"
        )

        maximum_application_amount = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="AMT_APPLICATION",
            agg_func="max",
            feature_name="MAXIMUM_APPLICATION_AMOUNT"
        )

        minimum_application_amount = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="AMT_APPLICATION",
            agg_func="min",
            feature_name="MINIMUM_APPLICATION_AMOUNT"
        )

        total_approved_credit = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="AMT_CREDIT",
            agg_func="sum",
            feature_name="TOTAL_APPROVED_CREDIT"
        )

        print(total_approved_credit.head())
        print(total_approved_credit.columns)

        average_approved_credit = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="AMT_CREDIT",
            agg_func="mean",
            feature_name="AVERAGE_APPROVED_CREDIT"
        )

        maximum_approved_credit = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="AMT_CREDIT",
            agg_func="max",
            feature_name="MAXIMUM_APPROVED_CREDIT"
        )

        minimum_approved_credit = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="AMT_CREDIT",
            agg_func="min",
            feature_name="MINIMUM_APPROVED_CREDIT"
        )


        total_down_payment = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="AMT_DOWN_PAYMENT",
            agg_func="sum",
            feature_name="TOTAL_DOWN_PAYMENT"
        )

        average_down_payment = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="AMT_DOWN_PAYMENT",
            agg_func="mean",
            feature_name="AVERAGE_DOWN_PAYMENT"
        )

        average_down_payment_rate = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="RATE_DOWN_PAYMENT",
            agg_func="mean",
            feature_name="AVERAGE_DOWN_PAYMENT_RATE"
        )

        features = self.merge_feature(features, total_application_amount)
        features = self.merge_feature(features, average_application_amount)
        features = self.merge_feature(features, maximum_application_amount)
        features = self.merge_feature(features, minimum_application_amount)

        features = self.merge_feature(features, total_approved_credit)
        features = self.merge_feature(features, average_approved_credit)
        features = self.merge_feature(features, maximum_approved_credit)
        features = self.merge_feature(features, minimum_approved_credit)

        features = self.merge_feature(features, total_down_payment)
        features = self.merge_feature(features, average_down_payment)
        features = self.merge_feature(features, average_down_payment_rate)


        print(features.columns.tolist())


        features["APPROVAL_AMOUNT_RATIO"] = (
            features["TOTAL_APPROVED_CREDIT"] /
            features["TOTAL_APPLICATION_AMOUNT"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["AVERAGE_APPROVAL_AMOUNT_RATIO"] = (
            features["AVERAGE_APPROVED_CREDIT"] /
            features["AVERAGE_APPLICATION_AMOUNT"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["DOWN_PAYMENT_TO_CREDIT_RATIO"] = (
            features["TOTAL_DOWN_PAYMENT"] /
            features["TOTAL_APPROVED_CREDIT"]
        ).replace([np.inf, -np.inf], 0).fillna(0)



        date_columns = [
            "DAYS_FIRST_DRAWING",
            "DAYS_FIRST_DUE",
            "DAYS_LAST_DUE_1ST_VERSION",
            "DAYS_LAST_DUE",
            "DAYS_TERMINATION"
        ]

        previous_application[date_columns] = (
            previous_application[date_columns]
            .replace(365243, np.nan)
        )




        latest_application = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="DAYS_DECISION",
            agg_func="max",
            feature_name="LATEST_APPLICATION_DAYS"
        )

        oldest_application = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="DAYS_DECISION",
            agg_func="min",
            feature_name="OLDEST_APPLICATION_DAYS"
        )

        average_application_time = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="DAYS_DECISION",
            agg_func="mean",
            feature_name="AVERAGE_APPLICATION_DAYS"
        )

        average_installments = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="CNT_PAYMENT",
            agg_func="mean",
            feature_name="AVERAGE_INSTALLMENTS"
        )

        maximum_installments = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="CNT_PAYMENT",
            agg_func="max",
            feature_name="MAXIMUM_INSTALLMENTS"
        )

        minimum_installments = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="CNT_PAYMENT",
            agg_func="min",
            feature_name="MINIMUM_INSTALLMENTS"
        )


        average_annuity = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="AMT_ANNUITY",
            agg_func="mean",
            feature_name="AVERAGE_ANNUITY"
        )

        maximum_annuity = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="AMT_ANNUITY",
            agg_func="max",
            feature_name="MAXIMUM_ANNUITY"
        )

        minimum_annuity = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="AMT_ANNUITY",
            agg_func="min",
            feature_name="MINIMUM_ANNUITY"
        )


        previous_application["LOAN_DURATION"] = (
            previous_application["DAYS_LAST_DUE"] -
            previous_application["DAYS_FIRST_DUE"]
        )

        average_loan_duration = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="LOAN_DURATION",
            agg_func="mean",
            feature_name="AVERAGE_LOAN_DURATION"
        )

        maximum_loan_duration = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="LOAN_DURATION",
            agg_func="max",
            feature_name="MAXIMUM_LOAN_DURATION"
        )

        minimum_loan_duration = Aggregation.aggregate(
            df=previous_application,
            groupby_col="SK_ID_CURR",
            target_col="LOAN_DURATION",
            agg_func="min",
            feature_name="MINIMUM_LOAN_DURATION"
        )


        features = self.merge_feature(features, latest_application)
        features = self.merge_feature(features, oldest_application)
        features = self.merge_feature(features, average_application_time)

        features = self.merge_feature(features, average_installments)
        features = self.merge_feature(features, maximum_installments)
        features = self.merge_feature(features, minimum_installments)

        features = self.merge_feature(features, average_annuity)
        features = self.merge_feature(features, maximum_annuity)
        features = self.merge_feature(features, minimum_annuity)

        features = self.merge_feature(features, average_loan_duration)
        features = self.merge_feature(features, maximum_loan_duration)
        features = self.merge_feature(features, minimum_loan_duration)




        consumer_loans = (
            previous_application
            .query("NAME_CONTRACT_TYPE == 'Consumer loans'")
            .groupby("SK_ID_CURR")
            .size()
            .reset_index(name="TOTAL_CONSUMER_LOANS")
        )

        features = self.merge_feature(features, consumer_loans)


        cash_loans = (
            previous_application
            .query("NAME_CONTRACT_TYPE == 'Cash loans'")
            .groupby("SK_ID_CURR")
            .size()
            .reset_index(name="TOTAL_CASH_LOANS")
        )

        features = self.merge_feature(features, cash_loans)


        revolving_loans = (
            previous_application
            .query("NAME_CONTRACT_TYPE == 'Revolving loans'")
            .groupby("SK_ID_CURR")
            .size()
            .reset_index(name="TOTAL_REVOLVING_LOANS")
        )

        features = self.merge_feature(features, revolving_loans)

        repeat_clients = (
            previous_application
            .query("NAME_CLIENT_TYPE == 'Repeater'")
            .groupby("SK_ID_CURR")
            .size()
            .reset_index(name="REPEATER_APPLICATIONS")
        )

        features = self.merge_feature(features, repeat_clients)

        new_clients = (
            previous_application
            .query("NAME_CLIENT_TYPE == 'New'")
            .groupby("SK_ID_CURR")
            .size()
            .reset_index(name="NEW_CLIENT_APPLICATIONS")
        )

        features = self.merge_feature(features, new_clients)



        refreshed_clients = (
            previous_application
            .query("NAME_CLIENT_TYPE == 'Refreshed'")
            .groupby("SK_ID_CURR")
            .size()
            .reset_index(name="REFRESHED_CLIENT_APPLICATIONS")
        )

        features = self.merge_feature(features, refreshed_clients)


        walkin_clients = (
            previous_application
            .query("NAME_CLIENT_TYPE == 'XNA'")
            .groupby("SK_ID_CURR")
            .size()
            .reset_index(name="UNKNOWN_CLIENT_TYPE_APPLICATIONS")
        )

        features = self.merge_feature(features, walkin_clients)


        insurance_ratio = (
            previous_application
            .groupby("SK_ID_CURR")["NFLAG_INSURED_ON_APPROVAL"]
            .mean()
            .reset_index(name="INSURANCE_APPROVAL_RATIO")
        )

        features = self.merge_feature(features, insurance_ratio)


        average_goods_price = Aggregation.aggregate(
            previous_application,
            "SK_ID_CURR",
            "AMT_GOODS_PRICE",
            "mean",
            "AVERAGE_GOODS_PRICE"
        )

        maximum_goods_price = Aggregation.aggregate(
            previous_application,
            "SK_ID_CURR",
            "AMT_GOODS_PRICE",
            "max",
            "MAXIMUM_GOODS_PRICE"
        )

        minimum_goods_price = Aggregation.aggregate(
            previous_application,
            "SK_ID_CURR",
            "AMT_GOODS_PRICE",
            "min",
            "MINIMUM_GOODS_PRICE"
        )

        features = self.merge_feature(features, average_goods_price)
        features = self.merge_feature(features, maximum_goods_price)
        features = self.merge_feature(features, minimum_goods_price)


        features["GOODS_TO_CREDIT_RATIO"] = (
            features["AVERAGE_GOODS_PRICE"] /
            features["AVERAGE_APPROVED_CREDIT"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["CONSUMER_LOAN_RATIO"] = (
            features["TOTAL_CONSUMER_LOANS"] /
            features["TOTAL_PREVIOUS_APPLICATIONS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)


        features["CASH_LOAN_RATIO"] = (
            features["TOTAL_CASH_LOANS"] /
            features["TOTAL_PREVIOUS_APPLICATIONS"]
        ).replace([np.inf, -np.inf], 0).fillna(0)



        previous_application["CREDIT_DIFFERENCE"] = (
            previous_application["AMT_CREDIT"] -
            previous_application["AMT_APPLICATION"]
        )

        average_credit_difference = Aggregation.aggregate(
            previous_application,
            "SK_ID_CURR",
            "CREDIT_DIFFERENCE",
            "mean",
            "AVERAGE_CREDIT_DIFFERENCE"
        )

        maximum_credit_difference = Aggregation.aggregate(
            previous_application,
            "SK_ID_CURR",
            "CREDIT_DIFFERENCE",
            "max",
            "MAXIMUM_CREDIT_DIFFERENCE"
        )

        minimum_credit_difference = Aggregation.aggregate(
            previous_application,
            "SK_ID_CURR",
            "CREDIT_DIFFERENCE",
            "min",
            "MINIMUM_CREDIT_DIFFERENCE"
        )

        previous_application["APPROVED_PERCENTAGE"] = (
            previous_application["AMT_CREDIT"] /
            previous_application["AMT_APPLICATION"]
        ).replace([np.inf, -np.inf], np.nan)



        average_approved_percentage = Aggregation.aggregate(
            previous_application,
            "SK_ID_CURR",
            "APPROVED_PERCENTAGE",
            "mean",
            "AVERAGE_APPROVED_PERCENTAGE"
        )

        minimum_approved_percentage = Aggregation.aggregate(
            previous_application,
            "SK_ID_CURR",
            "APPROVED_PERCENTAGE",
            "min",
            "MINIMUM_APPROVED_PERCENTAGE"
        )


        average_primary_interest = Aggregation.aggregate(
            previous_application,
            "SK_ID_CURR",
            "RATE_INTEREST_PRIMARY",
            "mean",
            "AVERAGE_PRIMARY_INTEREST"
        )

        average_privileged_interest = Aggregation.aggregate(
            previous_application,
            "SK_ID_CURR",
            "RATE_INTEREST_PRIVILEGED",
            "mean",
            "AVERAGE_PRIVILEGED_INTEREST"
        )


        average_seller_area = Aggregation.aggregate(
            previous_application,
            "SK_ID_CURR",
            "SELLERPLACE_AREA",
            "mean",
            "AVERAGE_SELLER_AREA"
        )

        maximum_seller_area = Aggregation.aggregate(
            previous_application,
            "SK_ID_CURR",
            "SELLERPLACE_AREA",
            "max",
            "MAXIMUM_SELLER_AREA"
        )

        features["AVERAGE_CREDIT_TO_ANNUITY_RATIO"] = (
            features["AVERAGE_APPROVED_CREDIT"] /
            features["AVERAGE_ANNUITY"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["AVERAGE_APPLICATION_TO_GOODS_RATIO"] = (
            features["AVERAGE_APPLICATION_AMOUNT"] /
            features["AVERAGE_GOODS_PRICE"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        features["AVERAGE_INSTALLMENT_TO_DURATION_RATIO"] = (
            features["AVERAGE_INSTALLMENTS"] /
            features["AVERAGE_LOAN_DURATION"]
        ).replace([np.inf, -np.inf], 0).fillna(0)


        features = self.merge_feature(features, average_credit_difference)
        features = self.merge_feature(features, maximum_credit_difference)
        features = self.merge_feature(features, minimum_credit_difference)

        features = self.merge_feature(features, average_approved_percentage)
        features = self.merge_feature(features, minimum_approved_percentage)

        features = self.merge_feature(features, average_primary_interest)
        features = self.merge_feature(features, average_privileged_interest)

        features = self.merge_feature(features, average_seller_area)
        features = self.merge_feature(features, maximum_seller_area)





        return features;