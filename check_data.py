import pandas as pd

df = pd.read_csv("maharashtra_housing_data.csv")

print("First 5 Rows:\n")
print(df.head())

print("\nDataset Summary:\n")
print(df.describe())

print("\nLocation Distribution:\n")
print(df["location"].value_counts())