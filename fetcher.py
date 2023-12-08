import pandas as pd


class Fetcher:
    def __init__(self):
        self.train_path = 'data/Train.csv'
        self.test_path = 'data/Test.csv'

    def fetch(self):
        return pd.read_csv(self.train_path), pd.read_csv(self.test_path)
