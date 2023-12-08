from fetcher import Fetcher
from feature_processing import FeatureEngineer
from model import Model


def run_pipeline():
    data_fetcher = Fetcher()
    feat_engineer = FeatureEngineer()

    # Fetch data
    train_raw, test_raw = data_fetcher.fetch()

    # Engineer the features
    train_featured, test_featured = (feat_engineer.feature_engineering(train_raw), feat_engineer.feature_engineering(test_raw))

    # Train a model
    # params = {"solver": "lbfgs", "max_iter": 1000, "multi_class": "auto", "random_state": 8888}
    trainer = Model(train_featured, test_featured)
    trainer.train_model()

    # Evaluate
    trainer.evaluate()


if __name__ == "__main__":
    run_pipeline()
