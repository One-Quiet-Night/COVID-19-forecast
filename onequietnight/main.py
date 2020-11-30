"""Retrain the model and generate real-time predictions.

This script downloads the data to memory, transforms the features, generates
a training data set using historical data of features and targets, retrains
a model, and makes predictions using the latest instance of the feature values.
"""
from pathlib import Path
from onequietnight.env import OneQuietNightEnvironment


def main():
    """Retrain the model and generate real-time predictions."""
    env = OneQuietNightEnvironment(Path.cwd())
    env.get_data()
    env.get_features()
    env.train_models()
    env.save_visualization_data()
    env.save_covidhub_data()


if __name__ == "__main__":
    main()
