
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load prepared dataset
df = pd.read_csv("dataset/netflix_prepared.csv")

# Step 2: Load TF-IDF matrix
with open("dataset/tfidf_matrix.pkl", "rb") as file:
    tfidf_matrix = pickle.load(file)


# Step 3: Create recommendation function
def recommend_movies(movie_name, number_of_recommendations=5):

    # Find the entered title
    matches = df.index[
        df["title"].str.casefold() == movie_name.strip().casefold()
    ].tolist()

    # Check whether title exists
    if not matches:
        print("Sorry! Title not found in Netflix dataset.")
        return

    # Select first matching title
    title_index = matches[0]

    # Calculate similarity scores
    similarity_scores = cosine_similarity(
        tfidf_matrix[title_index],
        tfidf_matrix
    ).flatten()

    # Sort scores in descending order
    sorted_indices = similarity_scores.argsort()[::-1]

    # Remove the selected title itself
    recommended_indices = [
        i for i in sorted_indices if i != title_index
    ][:number_of_recommendations]

    # Display selected title
    print("\nSELECTED TITLE:", df.iloc[title_index]["title"])

    # Display recommendations
    print("\nRECOMMENDED NETFLIX TITLES:")

    for index in recommended_indices:

        print(
            df.iloc[index]["title"],
            "| Similarity:",
            round(float(similarity_scores[index]), 3)
        )


# Step 4: Get input from user
movie_name = input("Enter a Netflix title: ")

# Step 5: Call recommendation function
recommend_movies(movie_name)

print("\nRECOMMENDATION ENGINE COMPLETED!")
