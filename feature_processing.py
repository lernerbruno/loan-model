import pandas as pd
import numpy as np


class FeatureEngineer:

    def binning(self, feature_column):
        pass

    def normalising(self, feature_column):
        pass

    def feature_engineering(self, df):
        processed = pd.DataFrame()

        # Target variable
        processed['Approve Loan'] = df['Approve Loan']

        # TODO: binning and normalising

        # Numerical Features
        # Feature #1 - Percentage of the annual Income the loan amount represents
        processed['Loan to Income Ratio'] = df['Loan Amount'] / df['Annual Income']  # TODO: take care of 0

        # Feature - Percentage of the loan the installment represents
        processed['Installment Amount'] = df['Installment Amount'] / df['Loan Amount']
        # processed['Installment Amount'] = pd.cut(df['Installment Amount']/df['Loan Amount'], bins=10, labels=np.arange(10))

        # Feature # 2 - Equal-width binning on Term
        processed['Term'] = df['Term']

        # Categorical Features

        # # Feature # 3 - If the income is verified
        processed["Is Verified"] = np.where(df["Income Verification Status"] == "Verified", 1, 0)

        # # Feature # 4 - If there is a due settlement
        processed['Due Settlement'] = np.where(df['Due Settlement'] == 'Y', 1, 0)

        # # Feature # 5 - If there is a payment plan
        processed['Payment Plan'] = np.where(df['Payment Plan'] == 'Y', 1, 0)

        return processed

