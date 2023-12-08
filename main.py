from fetcher import Fetcher
from feature_processing import FeatureEngineer
from model import Model


def run_pipeline():
    data_fetcher = Fetcher()
    feat_engineer = FeatureEngineer()

    # Fetch data
    train_raw, test_raw = data_fetcher.fetch()

    # Engineer the features
    train_featured, test_featured = (feat_engineer.feature_engineering(train_raw),
                                     feat_engineer.feature_engineering(test_raw))

    # Train a model
    model = Model(train_featured, test_featured)
    model.train_model()

    # Evaluate
    model.evaluate()


if __name__ == "__main__":
    run_pipeline()
