import pandas as pd
import numpy as np


class FeatureEngineer:

    def feature_engineering(self, df):
        processed = pd.DataFrame()

        # Target variable
        processed['Approve Loan'] = df['Approve Loan']

        # Binnning numerical Features TODO: Play with number of bins or even change binning strategy
        # self.processed['Loan Amount'] = pd.cut(df['Loan Amount'], bins=10, labels=np.arange(10))
        # self.processed['Annual Income'] = pd.cut(df['Annual Income'], bins=10, labels=np.arange(10))

        # Numerical Features
        # Feature #1 - Percentage of the annual Income the loan applicant needs to pay per month
        processed['Income to Loan Ratio'] = df['Installment Amount'] / df['Annual Income']  # TODO: take care of 0

        # Feature # 2 - Equal-width binning on Term
        processed['Term'] = pd.cut(df['Term'], bins=10, labels=np.arange(10))

        # Categorical Features

        # Feature # 3 - If the income is verified
        processed["Is Verified"] = np.where(df["Income Verification Status"] == "Verified", 1, 0)

        # Feature # 4 - If there is a due settlement
        processed['Due Settlement'] = np.where(df['Due Settlement'] == 'Y', 1, 0)

        # Feature # 5 - If there is a payment plan
        processed['Payment Plan'] = np.where(df['Payment Plan'] == 'Y', 1, 0)

        return processed

