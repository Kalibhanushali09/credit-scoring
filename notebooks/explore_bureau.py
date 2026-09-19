import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

bureau = pd.read_csv('data/raw/bureau.csv')
installments = pd.read_csv('data/raw/installments_payments.csv')
credits=pd.read_csv('data/raw/credit_card_balance.csv')
bureau_bal=pd.read_csv('data/raw/bureau_balance.csv')

print("=== BUREAU ===")
print("Shape:", bureau.shape)
print("\nColumns and dtypes:")
print(bureau.dtypes)
print("\nFirst 3 rows:")
print(bureau.head(3))
print("\nMissing values:")
print(bureau.isnull().sum())

print("\n=== INSTALLMENTS ===")
print("Shape:", installments.shape)
print("\nColumns and dtypes:")
print(installments.dtypes)
print("\nFirst 3 rows:")
print(installments.head(3))
print("\nMissing values:")
print(installments.isnull().sum())

prev = pd.read_csv('data/raw/previous_application.csv')
print(prev.shape)
print(prev.dtypes)
print(prev.head(3))


print("\n=== CREDIT CARD BALANCE ===")
print("Shape:", credits.shape)
print("\nColumns and dtypes:")
print(credits.dtypes)
print("\nFirst 3 rows:")
print(credits.head(3))
print("\nMissing values:")
print(credits.isnull().sum())

print("\n=== BUREAU BALANCE ===")
print("Shape:", bureau_bal.shape)
print("\nColumns and dtypes:")
print(bureau_bal.dtypes)
print("\nFirst 3 rows:")
print(bureau_bal.head(3))
print("\nMissing values:")
print(bureau_bal.isnull().sum())