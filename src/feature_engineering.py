
import pandas as pd

# Step 1: Load cleaned dataset
df = pd.read_csv("dataset/netflix_cleaned.csv")

# Step 2: Select features useful for recommendations
features = [
    "type",
    "director",
    "country",
    "rating",
    "listed_in"
]

# Step 3: Handle empty values
for col in features:
    df[col] = df[col].fillna("").astype(str).str.strip()

# Step 4: Combine all selected features
df["combined_features"] = (
    df["type"] + " " +
    df["director"] + " " +
    df["country"] + " " +
    df["rating"] + " " +
    df["listed_in"]
)

# Step 5: Display the result
print("\nFEATURE ENGINEERING RESULT:")

print(df[["title", "combined_features"]].head(10))

# Step 6: Save the prepared dataset
df.to_csv("dataset/netflix_prepared.csv", index=False)

print("\nPREPARED DATASET SHAPE:", df.shape)

print("\nFEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
