import pandas as pd
import numpy as np


def load_qualtrics_csv(filepath):
    """
    Load a Qualtrics CSV file.
    The file may still contain metadata rows.
    """
    df = pd.read_csv(filepath)
    return df


def main():
    filepath = "data/raw/results.csv"
    df = load_qualtrics_csv(filepath)

    print("Loaded data shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    main()
