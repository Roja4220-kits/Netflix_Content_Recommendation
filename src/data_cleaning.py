
import pandas as pd

# Step 1: Load the original dataset
df = pd.read_csv("dataset/netflix_titles.csv")

print("ORIGINAL DATASET SHAPE:", df.shape)

# Step 2: Select columns needed for recommendation
selected_columns = [
    "show_id",
    "type",
    "title",
    "director",
    "country",
    "rating",
    "listed_in"
]

df = df[selected_columns].copy()

# Step 3: Handle missing values and unwanted placeholders
text_columns = [
    "type",
    "title",
    "director",
    "country",
    "rating",
    "listed_in"
]

for col in text_columns:

    # Replace actual missing values with empty strings
    df[col] = df[col].fillna("").astype(str)

    # Remove unnecessary spaces
    df[col] = df[col].str.strip()

    # Treat placeholder values as missing information
    df[col] = df[col].mask(
        df[col].str.casefold().isin(["not given", "unknown", "nan"]),
        ""
    )

# Step 4: Remove duplicate records
df = df.drop_duplicates()

# Step 5: Remove records without a title
df = df[df["title"] != ""]

# Step 6: Check cleaned dataset
print("\nCLEANED DATASET:")
print(df.head())

print("\nCLEANED DATASET SHAPE:")
print(df.shape)

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nEMPTY VALUES:")
print((df[text_columns] == "").sum())

# Step 7: Save the cleaned dataset
df.to_csv("dataset/netflix_cleaned.csv", index=False)

print("\nDATA CLEANING COMPLETED SUCCESSFULLY!")
