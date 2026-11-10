
import pandas as pd

# Step 1: Load the Netflix dataset
df = pd.read_csv("dataset/netflix_titles.csv")

# Step 2: Display first 5 records
print("\nFIRST 5 RECORDS:")
print(df.head())

# Step 3: Display dataset information
print("\nDATASET SHAPE:")
print(df.shape)

# Step 4: Display all column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# Step 5: Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Step 6: Check duplicate records
print("\nDUPLICATE RECORDS:")
print(df.duplicated().sum())
