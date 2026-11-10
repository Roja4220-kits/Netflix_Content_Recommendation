
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv("dataset/netflix_titles.csv")

# Step 2: Analyze Movie and TV Show counts
print("\nCONTENT TYPE COUNT:")
print(df["type"].value_counts())

# Step 3: Calculate percentages
print("\nCONTENT TYPE PERCENTAGE:")
print(df["type"].value_counts(normalize=True) * 100)

# Step 4: Find top 10 countries
countries = (
    df["country"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
)

top_countries = countries.value_counts().head(10)

print("\nTOP 10 COUNTRIES:")
print(top_countries)

# Step 5: Find top 10 genres
genres = (
    df["listed_in"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
)

top_genres = genres.value_counts().head(10)

print("\nTOP 10 GENRES:")
print(top_genres)

# Step 6: Analyze content by release year
year_counts = df["release_year"].value_counts().sort_index()

print("\nLATEST RELEASE YEARS:")
print(year_counts.tail(10))

# Step 7: Visualize Movie vs TV Show
plt.figure(figsize=(7, 5))
df["type"].value_counts().plot(kind="bar")
plt.title("Movies vs TV Shows on Netflix")
plt.xlabel("Content Type")
plt.ylabel("Number of Titles")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("screenshots/content_type.png")
plt.show()

# Step 8: Visualize top 10 countries
plt.figure(figsize=(9, 5))
top_countries.sort_values().plot(kind="barh")
plt.title("Top 10 Countries by Netflix Titles")
plt.xlabel("Number of Titles")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig("screenshots/top_countries.png")
plt.show()

# Step 9: Visualize top 10 genres
plt.figure(figsize=(9, 5))
top_genres.sort_values().plot(kind="barh")
plt.title("Top 10 Netflix Genres")
plt.xlabel("Number of Titles")
plt.ylabel("Genre")
plt.tight_layout()
plt.savefig("screenshots/top_genres.png")
plt.show()

# Step 10: Visualize content by release year
plt.figure(figsize=(10, 5))
year_counts.plot(kind="line")
plt.title("Netflix Content by Release Year")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.savefig("screenshots/release_year.png")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")
