from src.features.static_features import build_static_features
import pandas as pd

def load_path(path):
    
    df=pd.read_csv(path)

    X,y=build_static_features(df)

    return X,y


def load_bureau(path):
    df = pd.read_csv(path)
    return df

def load_installments(path):
    df = pd.read_csv(path)
    return df

def load_previous_app(path):
    df = pd.read_csv(path)
    return df

def load_credit_card_bal_app(path):
    df = pd.read_csv(path)
    return df

def load_bureau_bal_app(path):
    df = pd.read_csv(path)
    return df


