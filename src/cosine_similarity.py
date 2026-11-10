
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load prepared dataset
df = pd.read_csv("dataset/netflix_prepared.csv")

# Step 2: Load saved TF-IDF matrix
with open("dataset/tfidf_matrix.pkl", "rb") as file:
    tfidf_matrix = pickle.load(file)

# Step 3: Get a Netflix title from the user
movie_name = input("Enter a Netflix title: ").strip()

# Step 4: Find the title in the dataset
matches = df.index[
    df["title"].str.casefold() == movie_name.casefold()
].tolist()

if not matches:
    print("Title not found in the Netflix dataset.")

else:
    # Select the first matching title
    title_index = matches[0]

    # Step 5: Calculate similarity with all titles
    similarity_scores = cosine_similarity(
        tfidf_matrix[title_index],
        tfidf_matrix
    ).flatten()

    # Step 6: Sort titles by similarity score
    sorted_indices = similarity_scores.argsort()[::-1]

    # Step 7: Exclude the selected title itself
    recommended_indices = [
        i for i in sorted_indices if i != title_index
    ][:5]

    # Step 8: Display recommendations
    print("\nSELECTED TITLE:", df.iloc[title_index]["title"])

    print("\nTOP 5 SIMILAR TITLES:")

    for index in recommended_indices:
        print(
            df.iloc[index]["title"],
            "| Similarity:",
            round(similarity_scores[index], 3)
        )

print("\nCOSINE SIMILARITY COMPLETED!")
