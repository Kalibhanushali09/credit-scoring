import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def engineer_feature(df):
    dataset = df.copy()

    if "DAYS_EMPLOYED" in dataset.columns:
        dataset["DAYS_EMPLOYED"] = dataset["DAYS_EMPLOYED"].replace(365243, np.nan)

    median_income = dataset["AMT_INCOME_TOTAL"].median()
    dataset["IS_TRUE_ENTREPRENEUR"] = (
        (dataset["FLAG_EMP_PHONE"] == 0)
        & (dataset["ORGANIZATION_TYPE"] == "Self-employed")
        & (dataset["AMT_INCOME_TOTAL"] > median_income)
    ).astype(int)

    return dataset

def handle_missing(df):
    dataset=df.copy()
    for col in dataset.columns:
        if dataset[col].isnull().sum() > 0:
            if dataset[col].dtype == "object":
                dataset[col] = dataset[col].fillna("unknown")
            else:
                dataset[col] = dataset[col].fillna(dataset[col].median())
    return dataset

def encode_categoricals(df):
    dataset = df.copy()
    dataset = pd.get_dummies(dataset, drop_first=True, dtype=int)
    return dataset

def build_static_features(df):
    X=df.drop("TARGET",axis=1)
    y=df["TARGET"]

    X=engineer_feature(X)
    X=handle_missing(X)
    X=encode_categoricals(X)

    return X,y